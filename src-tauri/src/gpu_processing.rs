use std::sync::Arc;
use std::time::Instant;

use bytemuck;
use half::f16;
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Rgba};
use wgpu::util::{DeviceExt, TextureDataOrder};

use crate::cubecl_processing;
use crate::image_processing::{AllAdjustments, GpuContext};
use crate::lut_processing::Lut;
use crate::{AppState, GpuImageCache};

pub fn get_or_init_gpu_context(state: &tauri::State<AppState>) -> Result<GpuContext, String> {
    let mut context_lock = state.gpu_context.lock().unwrap();
    if let Some(context) = &*context_lock {
        return Ok(context.clone());
    }
    let mut instance_desc = wgpu::InstanceDescriptor::from_env_or_default();

    #[cfg(target_os = "windows")]
    if std::env::var("WGPU_BACKEND").is_err() {
        instance_desc.backends = wgpu::Backends::PRIMARY;
    }

    let instance = wgpu::Instance::new(&instance_desc);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .map_err(|e| format!("Failed to find a wgpu adapter: {}", e))?;

    let mut required_features = wgpu::Features::empty();
    if adapter
        .features()
        .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
    {
        required_features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    }

    let limits = adapter.limits();

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Processing Device"),
        required_features,
        required_limits: limits.clone(),
        experimental_features: wgpu::ExperimentalFeatures::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| e.to_string())?;

    let new_context = GpuContext {
        device: Arc::new(device),
        queue: Arc::new(queue),
        limits,
    };
    *context_lock = Some(new_context.clone());
    Ok(new_context)
}

fn read_texture_data(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    size: wgpu::Extent3d,
) -> Result<Vec<u8>, String> {
    let unpadded_bytes_per_row = 4 * size.width;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) & !(align - 1);
    let output_buffer_size = (padded_bytes_per_row * size.height) as u64;

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(size.height),
            },
        },
        size,
    );

    queue.submit(Some(encoder.finish()));
    let buffer_slice = output_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(60)),
        })
        .unwrap();
    rx.recv().unwrap().map_err(|e| e.to_string())?;

    let padded_data = buffer_slice.get_mapped_range().to_vec();
    output_buffer.unmap();

    if padded_bytes_per_row == unpadded_bytes_per_row {
        Ok(padded_data)
    } else {
        let mut unpadded_data = Vec::with_capacity((unpadded_bytes_per_row * size.height) as usize);
        for chunk in padded_data.chunks(padded_bytes_per_row as usize) {
            unpadded_data.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
        }
        Ok(unpadded_data)
    }
}

fn to_rgba_f16(img: &DynamicImage) -> Vec<f16> {
    let rgba_f32 = img.to_rgba32f();
    rgba_f32.into_raw().into_iter().map(f16::from_f32).collect()
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurParams {
    radius: u32,
    tile_offset_x: u32,
    tile_offset_y: u32,
    input_width: u32,
    input_height: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FlareParams {
    amount: f32,
    is_raw: u32,
    exposure: f32,
    brightness: f32,
    contrast: f32,
    whites: f32,
    aspect_ratio: f32,
    _pad: f32,
}

pub struct GpuProcessor {
    context: GpuContext,
    blur_bgl: wgpu::BindGroupLayout,
    h_blur_pipeline: wgpu::ComputePipeline,
    v_blur_pipeline: wgpu::ComputePipeline,
    blur_params_buffer: wgpu::Buffer,

    flare_bgl_0: wgpu::BindGroupLayout,
    flare_bgl_1: wgpu::BindGroupLayout,
    flare_threshold_pipeline: wgpu::ComputePipeline,
    flare_ghosts_pipeline: wgpu::ComputePipeline,
    flare_params_buffer: wgpu::Buffer,
    flare_threshold_view: wgpu::TextureView,
    flare_ghosts_view: wgpu::TextureView,
    flare_final_view: wgpu::TextureView,
    flare_sampler: wgpu::Sampler,

    main_bgl: wgpu::BindGroupLayout,
    main_pipeline: wgpu::ComputePipeline,
    adjustments_buffer: wgpu::Buffer,
    dummy_blur_view: wgpu::TextureView,
    dummy_mask_view: wgpu::TextureView,
    dummy_lut_view: wgpu::TextureView,
    dummy_lut_sampler: wgpu::Sampler,
    ping_pong_view: wgpu::TextureView,
    sharpness_blur_view: wgpu::TextureView,
    clarity_blur_view: wgpu::TextureView,
    structure_blur_view: wgpu::TextureView,
    output_texture: wgpu::Texture,
    output_texture_view: wgpu::TextureView,
}

const FLARE_MAP_SIZE: u32 = 512;

impl GpuProcessor {
    pub fn new(context: GpuContext, max_width: u32, max_height: u32) -> Result<Self, String> {
        let device = &context.device;
        const MAX_MASKS: u32 = 9;

        let blur_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blur.wgsl").into()),
        });

        let blur_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blur BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blur Pipeline Layout"),
            bind_group_layouts: &[&blur_bgl],
            immediate_size: 0,
        });

        let h_blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Horizontal Blur Pipeline"),
            layout: Some(&blur_pipeline_layout),
            module: &blur_shader_module,
            entry_point: Some("horizontal_blur"),
            compilation_options: Default::default(),
            cache: None,
        });

        let v_blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Vertical Blur Pipeline"),
            layout: Some(&blur_pipeline_layout),
            module: &blur_shader_module,
            entry_point: Some("vertical_blur"),
            compilation_options: Default::default(),
            cache: None,
        });

        let blur_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Blur Params Buffer"),
            size: std::mem::size_of::<BlurParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let flare_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flare Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/flare.wgsl").into()),
        });

        let flare_bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flare BGL 0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let flare_bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flare BGL 1"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let flare_threshold_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Flare Threshold Layout"),
                bind_group_layouts: &[&flare_bgl_0],
                immediate_size: 0,
            });

        let flare_ghosts_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Flare Ghosts Layout"),
            bind_group_layouts: &[&flare_bgl_0, &flare_bgl_1],
            immediate_size: 0,
        });

        let flare_threshold_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Flare Threshold Pipeline"),
                layout: Some(&flare_threshold_layout),
                module: &flare_shader,
                entry_point: Some("threshold_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let flare_ghosts_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Flare Ghosts Pipeline"),
                layout: Some(&flare_ghosts_layout),
                module: &flare_shader,
                entry_point: Some("ghosts_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let flare_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Flare Params Buffer"),
            size: std::mem::size_of::<FlareParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let flare_tex_desc = wgpu::TextureDescriptor {
            label: Some("Flare Tex"),
            size: wgpu::Extent3d {
                width: FLARE_MAP_SIZE,
                height: FLARE_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        let flare_threshold_texture = device.create_texture(&flare_tex_desc);
        let flare_threshold_view = flare_threshold_texture.create_view(&Default::default());
        let flare_ghosts_texture = device.create_texture(&flare_tex_desc);
        let flare_ghosts_view = flare_ghosts_texture.create_view(&Default::default());
        let flare_final_texture = device.create_texture(&flare_tex_desc);
        let flare_final_view = flare_final_texture.create_view(&Default::default());

        let flare_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Flare Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Image Processing Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shader.wgsl").into()),
        });

        let mut bind_group_layout_entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        for i in 0..MAX_MASKS {
            bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 3 + i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3 + MAX_MASKS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 4 + MAX_MASKS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        });

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 5 + MAX_MASKS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 6 + MAX_MASKS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 7 + MAX_MASKS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });

        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 8 + MAX_MASKS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 9 + MAX_MASKS,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });

        let main_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Main BGL"),
            entries: &bind_group_layout_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&main_bgl],
            immediate_size: 0,
        });

        let main_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let adjustments_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Adjustments Buffer"),
            size: std::mem::size_of::<AllAdjustments>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dummy_texture_desc = wgpu::TextureDescriptor {
            label: Some("Dummy Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let dummy_blur_texture = device.create_texture(&dummy_texture_desc);
        let dummy_blur_view = dummy_blur_texture.create_view(&Default::default());

        let dummy_mask_texture = device.create_texture(&wgpu::TextureDescriptor {
            format: wgpu::TextureFormat::R8Unorm,
            ..dummy_texture_desc
        });
        let dummy_mask_view = dummy_mask_texture.create_view(&Default::default());

        let dummy_lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            dimension: wgpu::TextureDimension::D3,
            ..dummy_texture_desc
        });
        let dummy_lut_view = dummy_lut_texture.create_view(&Default::default());
        let dummy_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let max_tile_size = wgpu::Extent3d {
            width: max_width,
            height: max_height,
            depth_or_array_layers: 1,
        };

        let reusable_texture_desc = wgpu::TextureDescriptor {
            label: None,
            size: max_tile_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        let ping_pong_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Ping Pong Texture"),
            ..reusable_texture_desc
        });
        let ping_pong_view = ping_pong_texture.create_view(&Default::default());

        let sharpness_blur_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Sharpness Blur Texture"),
            ..reusable_texture_desc
        });
        let sharpness_blur_view = sharpness_blur_texture.create_view(&Default::default());

        let clarity_blur_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Clarity Blur Texture"),
            ..reusable_texture_desc
        });
        let clarity_blur_view = clarity_blur_texture.create_view(&Default::default());

        let structure_blur_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Structure Blur Texture"),
            ..reusable_texture_desc
        });
        let structure_blur_view = structure_blur_texture.create_view(&Default::default());

        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output Tile Texture"),
            size: max_tile_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_texture_view = output_texture.create_view(&Default::default());

        Ok(Self {
            context,
            blur_bgl,
            h_blur_pipeline,
            v_blur_pipeline,
            blur_params_buffer,
            flare_bgl_0,
            flare_bgl_1,
            flare_threshold_pipeline,
            flare_ghosts_pipeline,
            flare_params_buffer,
            flare_threshold_view,
            flare_ghosts_view,
            flare_final_view,
            flare_sampler,
            main_bgl,
            main_pipeline,
            adjustments_buffer,
            dummy_blur_view,
            dummy_mask_view,
            dummy_lut_view,
            dummy_lut_sampler,
            ping_pong_view,
            sharpness_blur_view,
            clarity_blur_view,
            structure_blur_view,
            output_texture,
            output_texture_view,
        })
    }

    pub fn run(
        &self,
        input_texture_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        adjustments: AllAdjustments,
        mask_bitmaps: &[ImageBuffer<Luma<u8>, Vec<u8>>],
        lut: Option<Arc<Lut>>,
    ) -> Result<Vec<u8>, String> {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let scale = (width.min(height) as f32) / 1080.0;
        const MAX_MASKS: u32 = 9;

        // ... [Textures and LUT setup remains identical] ...
        let full_texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let mask_views: Vec<wgpu::TextureView> = mask_bitmaps
            .iter()
            .map(|mask_bitmap| {
                let mask_texture = device.create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some("Full Mask Texture"),
                        size: full_texture_size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::R8Unorm,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                    TextureDataOrder::MipMajor,
                    mask_bitmap,
                );
                mask_texture.create_view(&Default::default())
            })
            .collect();

        let (lut_texture_view, lut_sampler) = if let Some(lut_arc) = &lut {
            let lut_data = &lut_arc.data;
            let size = lut_arc.size;
            let mut rgba_lut_data_f16 = Vec::with_capacity(lut_data.len() / 3 * 4);
            for chunk in lut_data.chunks_exact(3) {
                rgba_lut_data_f16.push(f16::from_f32(chunk[0]));
                rgba_lut_data_f16.push(f16::from_f32(chunk[1]));
                rgba_lut_data_f16.push(f16::from_f32(chunk[2]));
                rgba_lut_data_f16.push(f16::ONE);
            }
            let lut_texture = device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("LUT 3D Texture"),
                    size: wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: size,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                TextureDataOrder::MipMajor,
                bytemuck::cast_slice(&rgba_lut_data_f16),
            );
            let view = lut_texture.create_view(&Default::default());
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            (view, sampler)
        } else {
            (self.dummy_lut_view.clone(), self.dummy_lut_sampler.clone())
        };

        if adjustments.global.flare_amount > 0.0 {
            let mut encoder = device.create_command_encoder(&Default::default());

            let aspect_ratio = if height > 0 {
                width as f32 / height as f32
            } else {
                1.0
            };
            let f_params = FlareParams {
                amount: adjustments.global.flare_amount,
                is_raw: adjustments.global.is_raw_image,
                exposure: adjustments.global.exposure,
                brightness: adjustments.global.brightness,
                contrast: adjustments.global.contrast,
                whites: adjustments.global.whites,
                aspect_ratio,
                _pad: 0.0,
            };
            queue.write_buffer(&self.flare_params_buffer, 0, bytemuck::bytes_of(&f_params));

            let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Flare BG0"),
                layout: &self.flare_bgl_0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.flare_threshold_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.flare_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.flare_sampler),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&Default::default());
                cpass.set_pipeline(&self.flare_threshold_pipeline);
                cpass.set_bind_group(0, &bg0, &[]);
                cpass.dispatch_workgroups(FLARE_MAP_SIZE / 16, FLARE_MAP_SIZE / 16, 1);
            }

            let bg0_ghosts = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Flare BG0 Ghosts"),
                layout: &self.flare_bgl_0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.flare_final_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.flare_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.flare_sampler),
                    },
                ],
            });

            let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Flare BG1"),
                layout: &self.flare_bgl_1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.flare_threshold_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.flare_ghosts_view),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&Default::default());
                cpass.set_pipeline(&self.flare_ghosts_pipeline);
                cpass.set_bind_group(0, &bg0_ghosts, &[]);
                cpass.set_bind_group(1, &bg1, &[]);
                cpass.dispatch_workgroups(FLARE_MAP_SIZE / 16, FLARE_MAP_SIZE / 16, 1);
            }

            queue.submit(Some(encoder.finish()));

            let mut blur_encoder = device.create_command_encoder(&Default::default());

            let b_params = BlurParams {
                radius: 12,
                tile_offset_x: 0,
                tile_offset_y: 0,
                input_width: FLARE_MAP_SIZE,
                input_height: FLARE_MAP_SIZE,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            };
            queue.write_buffer(&self.blur_params_buffer, 0, bytemuck::bytes_of(&b_params));

            let h_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Flare Blur H"),
                layout: &self.blur_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.flare_ghosts_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.flare_threshold_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.blur_params_buffer.as_entire_binding(),
                    },
                ],
            });

            let v_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Flare Blur V"),
                layout: &self.blur_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.flare_threshold_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.flare_final_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.blur_params_buffer.as_entire_binding(),
                    },
                ],
            });

            {
                let mut cpass = blur_encoder.begin_compute_pass(&Default::default());
                cpass.set_pipeline(&self.h_blur_pipeline);
                cpass.set_bind_group(0, &h_bg, &[]);
                cpass.dispatch_workgroups(FLARE_MAP_SIZE / 256 + 1, FLARE_MAP_SIZE, 1);
            }

            {
                let mut cpass = blur_encoder.begin_compute_pass(&Default::default());
                cpass.set_pipeline(&self.v_blur_pipeline);
                cpass.set_bind_group(0, &v_bg, &[]);
                cpass.dispatch_workgroups(FLARE_MAP_SIZE, FLARE_MAP_SIZE / 256 + 1, 1);
            }

            queue.submit(Some(blur_encoder.finish()));
        }

        const TILE_SIZE: u32 = 2048;
        const TILE_OVERLAP: u32 = 128;

        let mut final_pixels = vec![0u8; (width * height * 4) as usize];
        let tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
        let tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;

        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                let x_start = tile_x * TILE_SIZE;
                let y_start = tile_y * TILE_SIZE;
                let tile_width = (width - x_start).min(TILE_SIZE);
                let tile_height = (height - y_start).min(TILE_SIZE);

                let input_x_start = (x_start as i32 - TILE_OVERLAP as i32).max(0) as u32;
                let input_y_start = (y_start as i32 - TILE_OVERLAP as i32).max(0) as u32;
                let input_x_end = (x_start + tile_width + TILE_OVERLAP).min(width);
                let input_y_end = (y_start + tile_height + TILE_OVERLAP).min(height);
                let input_width = input_x_end - input_x_start;
                let input_height = input_y_end - input_y_start;

                let input_texture_size = wgpu::Extent3d {
                    width: input_width,
                    height: input_height,
                    depth_or_array_layers: 1,
                };

                let run_blur = |base_radius: f32, output_view: &wgpu::TextureView| -> bool {
                    let radius = (base_radius * scale).ceil().max(1.0) as u32;
                    if radius == 0 {
                        return false;
                    }

                    let params = BlurParams {
                        radius,
                        tile_offset_x: input_x_start,
                        tile_offset_y: input_y_start,
                        input_width: input_width,
                        input_height: input_height,
                        _pad1: 0,
                        _pad2: 0,
                        _pad3: 0,
                    };
                    queue.write_buffer(&self.blur_params_buffer, 0, bytemuck::bytes_of(&params));

                    let mut blur_encoder = device.create_command_encoder(&Default::default());

                    let h_blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("H-Blur BG"),
                        layout: &self.blur_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(input_texture_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&self.ping_pong_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.blur_params_buffer.as_entire_binding(),
                            },
                        ],
                    });

                    {
                        let mut cpass = blur_encoder.begin_compute_pass(&Default::default());
                        cpass.set_pipeline(&self.h_blur_pipeline);
                        cpass.set_bind_group(0, &h_blur_bg, &[]);
                        cpass.dispatch_workgroups((input_width + 255) / 256, input_height, 1);
                    }

                    let v_blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("V-Blur BG"),
                        layout: &self.blur_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&self.ping_pong_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(output_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.blur_params_buffer.as_entire_binding(),
                            },
                        ],
                    });

                    {
                        let mut cpass = blur_encoder.begin_compute_pass(&Default::default());
                        cpass.set_pipeline(&self.v_blur_pipeline);
                        cpass.set_bind_group(0, &v_blur_bg, &[]);
                        cpass.dispatch_workgroups(input_width, (input_height + 255) / 256, 1);
                    }

                    queue.submit(Some(blur_encoder.finish()));
                    true
                };

                let did_create_sharpness_blur = run_blur(2.0, &self.sharpness_blur_view);
                let did_create_clarity_blur = run_blur(8.0, &self.clarity_blur_view);
                let did_create_structure_blur = run_blur(40.0, &self.structure_blur_view);

                let mut main_encoder = device.create_command_encoder(&Default::default());

                let mut tile_adjustments = adjustments;
                tile_adjustments.tile_offset_x = input_x_start;
                tile_adjustments.tile_offset_y = input_y_start;
                queue.write_buffer(
                    &self.adjustments_buffer,
                    0,
                    bytemuck::bytes_of(&tile_adjustments),
                );

                let mut bind_group_entries = vec![
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.output_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.adjustments_buffer.as_entire_binding(),
                    },
                ];
                for i in 0..MAX_MASKS as usize {
                    let view = mask_views.get(i).unwrap_or(&self.dummy_mask_view);
                    bind_group_entries.push(wgpu::BindGroupEntry {
                        binding: 3 + i as u32,
                        resource: wgpu::BindingResource::TextureView(view),
                    });
                }
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 3 + MAX_MASKS,
                    resource: wgpu::BindingResource::TextureView(&lut_texture_view),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 4 + MAX_MASKS,
                    resource: wgpu::BindingResource::Sampler(&lut_sampler),
                });

                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 5 + MAX_MASKS,
                    resource: wgpu::BindingResource::TextureView(if did_create_sharpness_blur {
                        &self.sharpness_blur_view
                    } else {
                        &self.dummy_blur_view
                    }),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 6 + MAX_MASKS,
                    resource: wgpu::BindingResource::TextureView(if did_create_clarity_blur {
                        &self.clarity_blur_view
                    } else {
                        &self.dummy_blur_view
                    }),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 7 + MAX_MASKS,
                    resource: wgpu::BindingResource::TextureView(if did_create_structure_blur {
                        &self.structure_blur_view
                    } else {
                        &self.dummy_blur_view
                    }),
                });

                let use_flare = adjustments.global.flare_amount > 0.0;
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 8 + MAX_MASKS,
                    resource: wgpu::BindingResource::TextureView(if use_flare {
                        &self.flare_final_view
                    } else {
                        &self.dummy_blur_view
                    }),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 9 + MAX_MASKS,
                    resource: wgpu::BindingResource::Sampler(&self.flare_sampler),
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Tile Bind Group"),
                    layout: &self.main_bgl,
                    entries: &bind_group_entries,
                });

                {
                    let mut compute_pass = main_encoder.begin_compute_pass(&Default::default());
                    compute_pass.set_pipeline(&self.main_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    compute_pass.dispatch_workgroups(
                        (input_width + 7) / 8,
                        (input_height + 7) / 8,
                        1,
                    );
                }
                queue.submit(Some(main_encoder.finish()));

                let processed_tile_data =
                    read_texture_data(device, queue, &self.output_texture, input_texture_size)?;

                let crop_x_start = x_start - input_x_start;
                let crop_y_start = y_start - input_y_start;

                for row in 0..tile_height {
                    let final_y = y_start + row;
                    let final_row_offset = (final_y * width + x_start) as usize * 4;
                    let source_y = crop_y_start + row;
                    let source_row_offset = (source_y * input_width + crop_x_start) as usize * 4;
                    let copy_bytes = (tile_width * 4) as usize;

                    final_pixels[final_row_offset..final_row_offset + copy_bytes].copy_from_slice(
                        &processed_tile_data[source_row_offset..source_row_offset + copy_bytes],
                    );
                }
            }
        }

        Ok(final_pixels)
    }
}

pub fn process_and_get_dynamic_image(
    context: &GpuContext,
    state: &tauri::State<AppState>,
    base_image: &DynamicImage,
    transform_hash: u64,
    all_adjustments: AllAdjustments,
    mask_bitmaps: &[ImageBuffer<Luma<u8>, Vec<u8>>],
    lut: Option<Arc<Lut>>,
    caller_id: &str,
) -> Result<DynamicImage, String> {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum CubeclMode {
        Off,
        Benchmark,
        PreferCubecl,
    }

    fn cubecl_mode_from_env() -> CubeclMode {
        match std::env::var("RAPIDRAW_CUBECL_MODE")
            .unwrap_or_else(|_| "off".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "benchmark" | "compare" => CubeclMode::Benchmark,
            "cubecl" | "prefer_cubecl" => CubeclMode::PreferCubecl,
            _ => CubeclMode::Off,
        }
    }

    fn cubecl_tolerance_from_env() -> u8 {
        std::env::var("RAPIDRAW_CUBECL_MATCH_TOLERANCE")
            .ok()
            .and_then(|value| value.parse::<u8>().ok())
            .unwrap_or(2)
    }

    let (width, height) = base_image.dimensions();
    log::info!(
        "[Caller: {}] GPU processing called for {}x{} image.",
        caller_id,
        width,
        height
    );
    let device = &context.device;
    let queue = &context.queue;

    let max_dim = context.limits.max_texture_dimension_2d;
    if width > max_dim || height > max_dim {
        log::warn!(
            "Image dimensions ({}x{}) exceed GPU limits ({}). Bypassing GPU processing and returning unprocessed image to prevent a crash. Try upgrading your GPU :)",
            width,
            height,
            max_dim
        );
        return Ok(base_image.clone());
    }

    let mut processor_lock = state.gpu_processor.lock().unwrap();
    if processor_lock.is_none()
        || processor_lock.as_ref().unwrap().width < width
        || processor_lock.as_ref().unwrap().height < height
    {
        let new_width = (width + 255) & !255;
        let new_height = (height + 255) & !255;
        log::info!(
            "Creating new GPU Processor for dimensions up to {}x{}",
            new_width,
            new_height
        );
        let processor = GpuProcessor::new(context.clone(), new_width, new_height)?;
        *processor_lock = Some(crate::GpuProcessorState {
            processor,
            width: new_width,
            height: new_height,
        });
    }
    let processor_state = processor_lock.as_ref().unwrap();
    let processor = &processor_state.processor;

    let mut cache_lock = state.gpu_image_cache.lock().unwrap();
    if let Some(cache) = &*cache_lock {
        if cache.transform_hash != transform_hash || cache.width != width || cache.height != height
        {
            *cache_lock = None;
        }
    }

    if cache_lock.is_none() {
        let img_rgba_f16 = to_rgba_f16(base_image);
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("Input Texture"),
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            TextureDataOrder::MipMajor,
            bytemuck::cast_slice(&img_rgba_f16),
        );
        let texture_view = texture.create_view(&Default::default());

        *cache_lock = Some(GpuImageCache {
            texture,
            texture_view,
            width,
            height,
            transform_hash,
        });
    }

    let cache = cache_lock.as_ref().unwrap();
    let cubecl_mode = cubecl_mode_from_env();
    let tolerance = cubecl_tolerance_from_env();
    let lut_for_cubecl = lut.clone();

    let wgsl_start_time = Instant::now();

    let wgsl_pixels = processor.run(
        &cache.texture_view,
        cache.width,
        cache.height,
        all_adjustments,
        mask_bitmaps,
        lut,
    )?;

    let wgsl_duration = wgsl_start_time.elapsed();
    let mut final_pixels = wgsl_pixels.clone();

    if cubecl_mode != CubeclMode::Off {
        match cubecl_processing::process_with_cubecl(
            base_image,
            all_adjustments,
            mask_bitmaps,
            lut_for_cubecl.as_deref(),
            Some(&wgsl_pixels),
        ) {
            Ok(cubecl_result) => {
                let diff_stats = cubecl_processing::compare_images(
                    &wgsl_pixels,
                    &cubecl_result.pixels,
                    tolerance,
                );
                log::info!(
                    "[GPU Compare][Caller: {}] {}x{} WGSL {:?} | CubeCL total {:?} (thr {:?}, blur {:?}, main {:?}, mask {:?}) | masks={} active_px={} mean_inf={:.4} max_inf={:.4} | fallback={} | mismatch {}/{} (tol {}) | max diff {} | mean diff {:.4} | {}",
                    caller_id,
                    width,
                    height,
                    wgsl_duration,
                    cubecl_result.timings.total,
                    cubecl_result.timings.flare_threshold,
                    cubecl_result.timings.flare_blur,
                    cubecl_result.timings.main,
                    cubecl_result.timings.mask_composite,
                    cubecl_result.mask_stats.mask_count,
                    cubecl_result.mask_stats.active_pixels,
                    cubecl_result.mask_stats.mean_influence,
                    cubecl_result.mask_stats.max_influence,
                    cubecl_result.used_wgsl_fallback,
                    diff_stats.mismatched_values,
                    diff_stats.compared_values,
                    tolerance,
                    diff_stats.max_abs_diff,
                    diff_stats.mean_abs_diff,
                    cubecl_result.parity_dashboard
                );

                if let Some(reason) = cubecl_result.fallback_reason.as_deref() {
                    log::info!(
                        "[GPU Compare][Caller: {}] CubeCL fallback reason: {}",
                        caller_id,
                        reason
                    );
                }

                if cubecl_mode == CubeclMode::PreferCubecl {
                    final_pixels = cubecl_result.pixels;
                }
            }
            Err(error) => {
                log::warn!(
                    "[GPU Compare][Caller: {}] CubeCL run failed for {}x{}: {}",
                    caller_id,
                    width,
                    height,
                    error
                );
                log::info!(
                    "GPU adjustments for {}x{} image took {:?} (WGSL only)",
                    width,
                    height,
                    wgsl_duration
                );
            }
        }
    } else {
        log::info!(
            "GPU adjustments for {}x{} image took {:?} (WGSL)",
            width,
            height,
            wgsl_duration
        );
    }

    let img_buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, final_pixels)
        .ok_or("Failed to create image buffer from GPU data")?;
    Ok(DynamicImage::ImageRgba8(img_buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubecl_processing;
    use crate::image_processing::{ColorCalibrationSettings, ColorGradeSettings, HslColor, Point};
    use crate::lut_processing::Lut;

    fn make_test_image(width: u32, height: u32) -> DynamicImage {
        let mut img = image::RgbaImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let r = ((x as f32 / width as f32) * 255.0).round() as u8;
                let g = ((y as f32 / height as f32) * 255.0).round() as u8;
                let b = (((x + y) as f32 / (width + height) as f32) * 255.0).round() as u8;
                img.put_pixel(x, y, image::Rgba([r, g, b, 255]));
            }
        }
        DynamicImage::ImageRgba8(img)
    }

    fn make_hdr_test_image(width: u32, height: u32) -> DynamicImage {
        let mut img = image::Rgba32FImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let fx = x as f32 / (width.saturating_sub(1).max(1)) as f32;
                let fy = y as f32 / (height.saturating_sub(1).max(1)) as f32;
                let r = 0.5 + fx * 7.5;
                let g = 0.3 + fy * 5.5;
                let b = 0.2 + ((fx + fy) * 0.5) * 6.0;
                img.put_pixel(x, y, image::Rgba([r, g, b, 1.0]));
            }
        }
        DynamicImage::ImageRgba32F(img)
    }

    fn test_gpu_context() -> Option<GpuContext> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;

        let mut required_features = wgpu::Features::empty();
        if adapter
            .features()
            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        {
            required_features |= wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        }

        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("WGSL vs CubeCL test device"),
            required_features,
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok()?;

        Some(GpuContext {
            device: Arc::new(device),
            queue: Arc::new(queue),
            limits,
        })
    }

    fn make_test_texture(
        context: &GpuContext,
        image: &DynamicImage,
        width: u32,
        height: u32,
    ) -> wgpu::TextureView {
        let image_f16 = to_rgba_f16(image);
        let texture = context.device.create_texture_with_data(
            &context.queue,
            &wgpu::TextureDescriptor {
                label: Some("WGSL test input texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            TextureDataOrder::MipMajor,
            bytemuck::cast_slice(&image_f16),
        );
        texture.create_view(&Default::default())
    }

    fn point(x: f32, y: f32) -> Point {
        bytemuck::cast([x, y, 0.0, 0.0])
    }

    fn hsl(h: f32, s: f32, l: f32) -> HslColor {
        bytemuck::cast([h, s, l, 0.0])
    }

    fn color_grade(h: f32, s: f32, l: f32) -> ColorGradeSettings {
        bytemuck::cast([h, s, l, 0.0])
    }

    fn color_calibration(
        shadows_tint: f32,
        red_hue: f32,
        red_saturation: f32,
        green_hue: f32,
        green_saturation: f32,
        blue_hue: f32,
        blue_saturation: f32,
    ) -> ColorCalibrationSettings {
        bytemuck::cast([
            shadows_tint,
            red_hue,
            red_saturation,
            green_hue,
            green_saturation,
            blue_hue,
            blue_saturation,
            0.0,
        ])
    }

    fn make_mask_bitmap<F: Fn(u32, u32) -> u8>(
        width: u32,
        height: u32,
        f: F,
    ) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let mut mask = ImageBuffer::from_pixel(width, height, Luma([0]));
        for y in 0..height {
            for x in 0..width {
                mask.put_pixel(x, y, Luma([f(x, y)]));
            }
        }
        mask
    }

    fn make_known_lut_cube() -> Arc<Lut> {
        // Same deterministic 2x2x2 cube used by CubeCL LUT golden tests.
        let data = vec![
            0.1, 0.2, 0.3, // (0,0,0)
            0.4, 0.5, 0.6, // (1,0,0)
            0.3, 0.9, 0.7, // (0,1,0)
            0.7, 0.1, 0.2, // (1,1,0)
            0.2, 0.7, 0.4, // (0,0,1)
            0.6, 0.3, 0.9, // (1,0,1)
            0.8, 0.2, 0.5, // (0,1,1)
            0.9, 0.8, 0.1, // (1,1,1)
        ];
        Arc::new(Lut { size: 2, data })
    }

    fn assert_mask_diff(
        diff: cubecl_processing::ImageDiffStats,
        width: u32,
        height: u32,
        max_ratio: f32,
        max_abs: u8,
    ) {
        let max_mismatch = ((width * height * 4) as f32 * max_ratio).ceil() as usize;
        assert!(
            diff.mismatched_values <= max_mismatch,
            "mismatch too high: {} > {} (max diff {}, mean {:.4})",
            diff.mismatched_values,
            max_mismatch,
            diff.max_abs_diff,
            diff.mean_abs_diff
        );
        assert!(
            diff.max_abs_diff <= max_abs,
            "max abs diff too high: {} > {}",
            diff.max_abs_diff,
            max_abs
        );
    }

    #[test]
    fn cubecl_matches_wgsl_identity_with_tolerance() {
        let Some(context) = test_gpu_context() else {
            return;
        };

        let width = 32;
        let height = 32;
        let image = make_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");

        let image_f16 = to_rgba_f16(&image);
        let texture = context.device.create_texture_with_data(
            &context.queue,
            &wgpu::TextureDescriptor {
                label: Some("WGSL test input texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            TextureDataOrder::MipMajor,
            bytemuck::cast_slice(&image_f16),
        );
        let texture_view = texture.create_view(&Default::default());

        let adjustments = AllAdjustments::default();
        let wgsl_pixels = processor
            .run(&texture_view, width, height, adjustments, &[], None)
            .expect("WGSL run failed");

        let cubecl_pixels =
            cubecl_processing::process_with_cubecl(&image, adjustments, &[], None, None)
                .expect("CubeCL run failed")
                .pixels;

        let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_pixels, 2);
        assert_eq!(diff.compared_values, (width * height * 4) as usize);
        assert_eq!(diff.mismatched_values, 0);
    }

    #[test]
    fn cubecl_uses_wgsl_fallback_for_unsupported_adjustments() {
        let width = 8;
        let height = 8;
        let image = make_test_image(width, height);
        let mut adjustments = AllAdjustments::default();
        adjustments.mask_count = 1;
        adjustments.mask_adjustments[0].luma_noise_reduction = 150.0;

        let fallback_pixels = vec![13u8; (width * height * 4) as usize];
        let result = cubecl_processing::process_with_cubecl(
            &image,
            adjustments,
            &[],
            None,
            Some(&fallback_pixels),
        )
        .expect("CubeCL run failed");

        assert!(result.used_wgsl_fallback);
        assert_eq!(result.pixels, fallback_pixels);
        assert!(result.fallback_reason.is_some());
    }

    #[test]
    fn cubecl_matches_wgsl_agx_identity_adjustments() {
        let Some(context) = test_gpu_context() else {
            return;
        };
        let width = 36;
        let height = 28;
        let image = make_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        let mut adjustments = AllAdjustments::default();
        adjustments.global.tonemapper_mode = 1;

        let wgsl_pixels = processor
            .run(&texture_view, width, height, adjustments, &[], None)
            .expect("WGSL run failed");
        let cubecl_pixels = cubecl_processing::process_with_cubecl(
            &image,
            adjustments,
            &[],
            None,
            Some(&wgsl_pixels),
        )
        .expect("CubeCL run failed")
        .pixels;

        let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_pixels, 3);
        assert_mask_diff(diff, width, height, 0.01, 5);
    }

    #[test]
    fn cubecl_matches_wgsl_agx_highlight_rolloff() {
        let Some(context) = test_gpu_context() else {
            return;
        };
        let width = 40;
        let height = 32;
        let image = make_hdr_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        let mut adjustments = AllAdjustments::default();
        adjustments.global.tonemapper_mode = 1;

        let wgsl_pixels = processor
            .run(&texture_view, width, height, adjustments, &[], None)
            .expect("WGSL run failed");
        let cubecl_pixels = cubecl_processing::process_with_cubecl(
            &image,
            adjustments,
            &[],
            None,
            Some(&wgsl_pixels),
        )
        .expect("CubeCL run failed")
        .pixels;

        let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_pixels, 3);
        assert_mask_diff(diff, width, height, 0.015, 6);
    }

    #[test]
    fn cubecl_matches_wgsl_single_mask_exposure() {
        let Some(context) = test_gpu_context() else {
            return;
        };
        let width = 48;
        let height = 40;
        let image = make_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        let mask0 = make_mask_bitmap(width, height, |x, _| if x < width / 2 { 255 } else { 0 });
        let mut adjustments = AllAdjustments::default();
        adjustments.mask_count = 1;
        adjustments.mask_adjustments[0].exposure = 0.45;

        let wgsl_pixels = processor
            .run(
                &texture_view,
                width,
                height,
                adjustments,
                &[mask0.clone()],
                None,
            )
            .expect("WGSL run failed");
        let cubecl_pixels = cubecl_processing::process_with_cubecl(
            &image,
            adjustments,
            &[mask0],
            None,
            Some(&wgsl_pixels),
        )
        .expect("CubeCL run failed")
        .pixels;
        let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_pixels, 3);
        assert_mask_diff(diff, width, height, 0.01, 8);
    }

    #[test]
    fn cubecl_matches_wgsl_overlapping_masks() {
        let Some(context) = test_gpu_context() else {
            return;
        };
        let width = 64;
        let height = 48;
        let image = make_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        let mask0 = make_mask_bitmap(
            width,
            height,
            |x, _| if x < width * 2 / 3 { 220 } else { 0 },
        );
        let mask1 = make_mask_bitmap(width, height, |_, y| if y > height / 3 { 180 } else { 0 });

        let mut adjustments = AllAdjustments::default();
        adjustments.mask_count = 2;
        adjustments.mask_adjustments[0].exposure = 0.4;
        adjustments.mask_adjustments[1].exposure = -0.35;

        let wgsl_pixels = processor
            .run(
                &texture_view,
                width,
                height,
                adjustments,
                &[mask0.clone(), mask1.clone()],
                None,
            )
            .expect("WGSL run failed");
        let cubecl_pixels = cubecl_processing::process_with_cubecl(
            &image,
            adjustments,
            &[mask0, mask1],
            None,
            Some(&wgsl_pixels),
        )
        .expect("CubeCL run failed")
        .pixels;
        let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_pixels, 3);
        assert_mask_diff(diff, width, height, 0.02, 10);
    }

    #[test]
    fn cubecl_matches_wgsl_mask_atlas_like_and_curves() {
        let Some(context) = test_gpu_context() else {
            return;
        };
        let width = 72;
        let height = 56;
        let image = make_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        let cx = (width / 2) as i32;
        let cy = (height / 2) as i32;
        let mask0 = make_mask_bitmap(width, height, |x, y| {
            let dx = x as i32 - cx;
            let dy = y as i32 - cy;
            let d2 = dx * dx + dy * dy;
            let outer = d2 < (width as i32 / 3).pow(2);
            let inner = d2 < (width as i32 / 6).pow(2);
            if outer && !inner { 255 } else { 0 }
        });
        let mask1 = make_mask_bitmap(width, height, |x, y| {
            let additive =
                (x > width / 4 && x < width * 3 / 4 && y > height / 4 && y < height * 3 / 4) as u8;
            let subtractive =
                (x > width / 3 && x < width * 2 / 3 && y > height / 3 && y < height * 2 / 3) as u8;
            if additive == 1 && subtractive == 0 {
                200
            } else {
                0
            }
        });

        let mut adjustments = AllAdjustments::default();
        adjustments.mask_count = 2;
        adjustments.mask_adjustments[0].exposure = 0.25;
        adjustments.mask_adjustments[1].brightness = 0.2;
        adjustments.mask_adjustments[0].luma_curve_count = 2;
        adjustments.mask_adjustments[0].luma_curve = [point(0.0, 0.0); 16];
        adjustments.mask_adjustments[0].luma_curve[1] = point(255.0, 230.0);

        let wgsl_pixels = processor
            .run(
                &texture_view,
                width,
                height,
                adjustments,
                &[mask0.clone(), mask1.clone()],
                None,
            )
            .expect("WGSL run failed");
        let cubecl_pixels = cubecl_processing::process_with_cubecl(
            &image,
            adjustments,
            &[mask0, mask1],
            None,
            Some(&wgsl_pixels),
        )
        .expect("CubeCL run failed")
        .pixels;
        let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_pixels, 4);
        assert_mask_diff(diff, width, height, 0.03, 12);
    }

    #[test]
    fn cubecl_matches_wgsl_mask_local_contrast_controls() {
        let Some(context) = test_gpu_context() else {
            return;
        };
        let width = 64;
        let height = 64;
        let image = make_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        let mask0 = make_mask_bitmap(width, height, |x, y| {
            if x > width / 4 && x < width * 3 / 4 && y > height / 4 && y < height * 3 / 4 {
                220
            } else {
                0
            }
        });

        let mut adjustments = AllAdjustments::default();
        adjustments.mask_count = 1;
        adjustments.mask_adjustments[0].sharpness = 0.2;
        adjustments.mask_adjustments[0].clarity = 0.25;
        adjustments.mask_adjustments[0].structure = 0.15;

        let wgsl_pixels = processor
            .run(
                &texture_view,
                width,
                height,
                adjustments,
                &[mask0.clone()],
                None,
            )
            .expect("WGSL run failed");
        let cubecl_pixels = cubecl_processing::process_with_cubecl(
            &image,
            adjustments,
            &[mask0],
            None,
            Some(&wgsl_pixels),
        )
        .expect("CubeCL run failed")
        .pixels;
        let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_pixels, 6);
        assert_mask_diff(diff, width, height, 0.16, 20);
    }

    #[test]
    fn cubecl_matches_wgsl_mask_glow_halation_flare_controls() {
        let Some(context) = test_gpu_context() else {
            return;
        };
        let width = 72;
        let height = 56;
        let image = make_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        let mask0 = make_mask_bitmap(width, height, |x, y| if (x + y) % 3 == 0 { 180 } else { 0 });

        let mut adjustments = AllAdjustments::default();
        adjustments.mask_count = 1;
        adjustments.mask_adjustments[0].glow_amount = 0.2;
        adjustments.mask_adjustments[0].halation_amount = 0.18;
        adjustments.mask_adjustments[0].flare_amount = 0.15;

        let wgsl_pixels = processor
            .run(
                &texture_view,
                width,
                height,
                adjustments,
                &[mask0.clone()],
                None,
            )
            .expect("WGSL run failed");
        let cubecl_pixels = cubecl_processing::process_with_cubecl(
            &image,
            adjustments,
            &[mask0],
            None,
            Some(&wgsl_pixels),
        )
        .expect("CubeCL run failed")
        .pixels;
        let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_pixels, 8);
        assert_mask_diff(diff, width, height, 0.22, 70);
    }

    #[test]
    fn cubecl_matches_wgsl_with_3d_lut_tetrahedral() {
        let Some(context) = test_gpu_context() else {
            return;
        };

        let width = 40;
        let height = 40;
        let image = make_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        let lut = make_known_lut_cube();
        let mut adjustments = AllAdjustments::default();
        adjustments.global.has_lut = 1;
        adjustments.global.lut_intensity = 0.75;

        let wgsl_pixels = processor
            .run(
                &texture_view,
                width,
                height,
                adjustments,
                &[],
                Some(lut.clone()),
            )
            .expect("WGSL run failed");
        let cubecl_pixels = cubecl_processing::process_with_cubecl(
            &image,
            adjustments,
            &[],
            Some(lut.as_ref()),
            Some(&wgsl_pixels),
        )
        .expect("CubeCL run failed")
        .pixels;

        let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_pixels, 2);
        assert_mask_diff(diff, width, height, 0.01, 3);
    }

    #[test]
    fn cubecl_matches_wgsl_global_identity_curve() {
        let Some(context) = test_gpu_context() else {
            return;
        };
        let width = 48;
        let height = 36;
        let image = make_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        let mut adjustments = AllAdjustments::default();
        adjustments.global.luma_curve_count = 2;
        adjustments.global.luma_curve = [point(0.0, 0.0); 16];
        adjustments.global.luma_curve[1] = point(255.0, 255.0);

        let wgsl_pixels = processor
            .run(&texture_view, width, height, adjustments, &[], None)
            .expect("WGSL run failed");
        let cubecl_result = cubecl_processing::process_with_cubecl(
            &image,
            adjustments,
            &[],
            None,
            Some(&wgsl_pixels),
        )
        .expect("CubeCL run failed");
        assert!(
            !cubecl_result.used_wgsl_fallback,
            "CubeCL unexpectedly fell back: {:?}",
            cubecl_result.fallback_reason
        );
        let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_result.pixels, 2);
        assert_mask_diff(diff, width, height, 0.01, 4);
    }

    #[test]
    fn cubecl_matches_wgsl_global_s_curve_contrast() {
        let Some(context) = test_gpu_context() else {
            return;
        };
        let width = 56;
        let height = 40;
        let image = make_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        let mut adjustments = AllAdjustments::default();
        adjustments.global.luma_curve_count = 4;
        adjustments.global.luma_curve = [point(0.0, 0.0); 16];
        adjustments.global.luma_curve[1] = point(64.0, 48.0);
        adjustments.global.luma_curve[2] = point(192.0, 208.0);
        adjustments.global.luma_curve[3] = point(255.0, 255.0);

        let wgsl_pixels = processor
            .run(&texture_view, width, height, adjustments, &[], None)
            .expect("WGSL run failed");
        let cubecl_result = cubecl_processing::process_with_cubecl(
            &image,
            adjustments,
            &[],
            None,
            Some(&wgsl_pixels),
        )
        .expect("CubeCL run failed");
        assert!(
            !cubecl_result.used_wgsl_fallback,
            "CubeCL unexpectedly fell back: {:?}",
            cubecl_result.fallback_reason
        );
        let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_result.pixels, 3);
        assert_mask_diff(diff, width, height, 0.02, 8);
    }

    #[test]
    fn cubecl_matches_wgsl_global_rgb_split_tone_curves() {
        let Some(context) = test_gpu_context() else {
            return;
        };
        let width = 56;
        let height = 44;
        let image = make_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        let mut adjustments = AllAdjustments::default();
        adjustments.global.red_curve_count = 3;
        adjustments.global.red_curve = [point(0.0, 0.0); 16];
        adjustments.global.red_curve[1] = point(128.0, 150.0);
        adjustments.global.red_curve[2] = point(255.0, 255.0);
        adjustments.global.green_curve_count = 3;
        adjustments.global.green_curve = [point(0.0, 0.0); 16];
        adjustments.global.green_curve[1] = point(128.0, 118.0);
        adjustments.global.green_curve[2] = point(255.0, 245.0);
        adjustments.global.blue_curve_count = 3;
        adjustments.global.blue_curve = [point(0.0, 8.0); 16];
        adjustments.global.blue_curve[1] = point(128.0, 120.0);
        adjustments.global.blue_curve[2] = point(255.0, 240.0);

        let wgsl_pixels = processor
            .run(&texture_view, width, height, adjustments, &[], None)
            .expect("WGSL run failed");
        let cubecl_result = cubecl_processing::process_with_cubecl(
            &image,
            adjustments,
            &[],
            None,
            Some(&wgsl_pixels),
        )
        .expect("CubeCL run failed");
        assert!(
            !cubecl_result.used_wgsl_fallback,
            "CubeCL unexpectedly fell back: {:?}",
            cubecl_result.fallback_reason
        );
        let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_result.pixels, 3);
        assert_mask_diff(diff, width, height, 0.02, 10);
    }

    #[test]
    fn cubecl_matches_wgsl_global_hsl_all_hue_ranges() {
        let Some(context) = test_gpu_context() else {
            return;
        };
        let width = 60;
        let height = 48;
        let image = make_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        for range_idx in 0..8usize {
            let mut adjustments = AllAdjustments::default();
            adjustments.global.hsl[range_idx] = hsl(0.12, 0.28, 0.15);

            let wgsl_pixels = processor
                .run(&texture_view, width, height, adjustments, &[], None)
                .expect("WGSL run failed");
            let cubecl_result = cubecl_processing::process_with_cubecl(
                &image,
                adjustments,
                &[],
                None,
                Some(&wgsl_pixels),
            )
            .expect("CubeCL run failed");
            assert!(
                !cubecl_result.used_wgsl_fallback,
                "CubeCL unexpectedly fell back for hue range {}: {:?}",
                range_idx, cubecl_result.fallback_reason
            );
            let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_result.pixels, 3);
            assert_mask_diff(diff, width, height, 0.03, 10);
        }
    }

    #[test]
    fn cubecl_matches_wgsl_global_color_grading_and_balance_extremes() {
        let Some(context) = test_gpu_context() else {
            return;
        };
        let width = 64;
        let height = 48;
        let image = make_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        let cases: [(&str, AllAdjustments); 5] = [
            {
                let mut adj = AllAdjustments::default();
                adj.global.color_grading_shadows = color_grade(220.0, 0.45, 0.20);
                adj.global.color_grading_blending = 1.0;
                adj.global.color_grading_balance = -0.6;
                ("shadows_only", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.color_grading_midtones = color_grade(35.0, 0.40, 0.18);
                adj.global.color_grading_blending = 1.0;
                ("midtones_only", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.color_grading_highlights = color_grade(55.0, 0.35, 0.25);
                adj.global.color_grading_blending = 1.0;
                adj.global.color_grading_balance = 0.7;
                ("highlights_only", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.color_grading_midtones = color_grade(20.0, 0.35, 0.10);
                adj.global.color_grading_blending = 1.0;
                adj.global.color_grading_balance = -1.0;
                ("balance_extreme_negative", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.color_grading_midtones = color_grade(20.0, 0.35, 0.10);
                adj.global.color_grading_blending = 1.0;
                adj.global.color_grading_balance = 1.0;
                ("balance_extreme_positive", adj)
            },
        ];

        for (label, adjustments) in cases {
            let wgsl_pixels = processor
                .run(&texture_view, width, height, adjustments, &[], None)
                .expect("WGSL run failed");
            let cubecl_result = cubecl_processing::process_with_cubecl(
                &image,
                adjustments,
                &[],
                None,
                Some(&wgsl_pixels),
            )
            .expect("CubeCL run failed");
            assert!(
                !cubecl_result.used_wgsl_fallback,
                "CubeCL unexpectedly fell back for {}: {:?}",
                label, cubecl_result.fallback_reason
            );
            let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_result.pixels, 3);
            assert_mask_diff(diff, width, height, 0.03, 10);
        }
    }

    #[test]
    fn cubecl_matches_wgsl_global_color_calibration_controls() {
        let Some(context) = test_gpu_context() else {
            return;
        };
        let width = 64;
        let height = 48;
        let image = make_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        let cases: [(&str, ColorCalibrationSettings); 7] = [
            (
                "red_hue",
                color_calibration(0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0),
            ),
            (
                "red_saturation",
                color_calibration(0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0),
            ),
            (
                "green_hue",
                color_calibration(0.0, 0.0, 0.0, -0.22, 0.0, 0.0, 0.0),
            ),
            (
                "green_saturation",
                color_calibration(0.0, 0.0, 0.0, 0.0, -0.30, 0.0, 0.0),
            ),
            (
                "blue_hue",
                color_calibration(0.0, 0.0, 0.0, 0.0, 0.0, 0.18, 0.0),
            ),
            (
                "blue_saturation",
                color_calibration(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32),
            ),
            (
                "shadows_tint",
                color_calibration(0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ),
        ];

        for (label, calibration) in cases {
            let mut adjustments = AllAdjustments::default();
            adjustments.global.color_calibration = calibration;

            let wgsl_pixels = processor
                .run(&texture_view, width, height, adjustments, &[], None)
                .expect("WGSL run failed");
            let cubecl_result = cubecl_processing::process_with_cubecl(
                &image,
                adjustments,
                &[],
                None,
                Some(&wgsl_pixels),
            )
            .expect("CubeCL run failed");
            assert!(
                !cubecl_result.used_wgsl_fallback,
                "CubeCL unexpectedly fell back for {}: {:?}",
                label, cubecl_result.fallback_reason
            );
            let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_result.pixels, 3);
            assert_mask_diff(diff, width, height, 0.03, 10);
        }
    }

    #[test]
    fn cubecl_matches_wgsl_global_local_contrast_and_centre_controls() {
        let Some(context) = test_gpu_context() else {
            return;
        };
        let width = 72;
        let height = 56;
        let image = make_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        let cases: [(&str, AllAdjustments); 5] = [
            {
                let mut adj = AllAdjustments::default();
                adj.global.sharpness = 0.28;
                ("sharpness_only", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.clarity = 0.30;
                ("clarity_only", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.structure = 0.22;
                ("structure_only", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.centré = 0.26;
                ("centre_only", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.sharpness = 0.22;
                adj.global.clarity = 0.18;
                adj.global.structure = 0.16;
                adj.global.centré = 0.24;
                ("combined", adj)
            },
        ];

        for (label, adjustments) in cases {
            let wgsl_pixels = processor
                .run(&texture_view, width, height, adjustments, &[], None)
                .expect("WGSL run failed");
            let cubecl_result = cubecl_processing::process_with_cubecl(
                &image,
                adjustments,
                &[],
                None,
                Some(&wgsl_pixels),
            )
            .expect("CubeCL run failed");
            assert!(
                !cubecl_result.used_wgsl_fallback,
                "CubeCL unexpectedly fell back for {}: {:?}",
                label, cubecl_result.fallback_reason
            );
            let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_result.pixels, 4);
            assert_mask_diff(diff, width, height, 0.08, 22);
        }
    }

    #[test]
    fn cubecl_matches_wgsl_global_advanced_effects_bundle_controls() {
        let Some(context) = test_gpu_context() else {
            return;
        };
        let width = 64;
        let height = 48;
        let image = make_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        let cases: [(&str, AllAdjustments); 8] = [
            {
                let mut adj = AllAdjustments::default();
                adj.global.dehaze = 0.35;
                ("dehaze_positive", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.dehaze = -0.35;
                ("dehaze_negative", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.glow_amount = 0.28;
                ("glow", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.halation_amount = 0.24;
                ("halation", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.vignette_amount = -0.45;
                adj.global.vignette_midpoint = 0.45;
                adj.global.vignette_roundness = 0.35;
                adj.global.vignette_feather = 0.75;
                ("vignette_darken", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.vignette_amount = 0.32;
                adj.global.vignette_midpoint = 0.42;
                adj.global.vignette_roundness = 0.55;
                adj.global.vignette_feather = 0.70;
                ("vignette_brighten", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.grain_amount = 0.35;
                adj.global.grain_size = 0.45;
                adj.global.grain_roughness = 0.65;
                ("grain", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.chromatic_aberration_red_cyan = 0.0008;
                adj.global.chromatic_aberration_blue_yellow = -0.0008;
                ("chromatic_aberration", adj)
            },
        ];

        for (label, adjustments) in cases {
            let wgsl_pixels = processor
                .run(&texture_view, width, height, adjustments, &[], None)
                .expect("WGSL run failed");
            let cubecl_result = cubecl_processing::process_with_cubecl(
                &image,
                adjustments,
                &[],
                None,
                Some(&wgsl_pixels),
            )
            .expect("CubeCL run failed");
            assert!(
                !cubecl_result.used_wgsl_fallback,
                "CubeCL unexpectedly fell back for {}: {:?}",
                label, cubecl_result.fallback_reason
            );
            let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_result.pixels, 8);
            assert_mask_diff(diff, width, height, 0.20, 45);
        }
    }

    #[test]
    fn cubecl_matches_wgsl_global_main_tonal_color_parameter_sweep() {
        let Some(context) = test_gpu_context() else {
            return;
        };
        let width = 56;
        let height = 44;
        let image = make_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        let cases: [(&str, AllAdjustments); 12] = [
            {
                let mut adj = AllAdjustments::default();
                adj.global.brightness = 0.35;
                ("brightness_pos", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.contrast = -0.30;
                ("contrast_neg", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.highlights = -0.45;
                ("highlights_neg", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.highlights = 0.30;
                ("highlights_pos", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.shadows = 0.35;
                ("shadows_pos", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.whites = 0.28;
                ("whites_pos", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.blacks = -0.32;
                ("blacks_neg", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.temperature = 0.30;
                ("temperature_pos", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.tint = -0.30;
                ("tint_neg", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.saturation = 0.35;
                ("saturation_pos", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.vibrance = 0.35;
                ("vibrance_pos", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.brightness = -0.25;
                adj.global.contrast = 0.22;
                adj.global.highlights = -0.20;
                adj.global.shadows = 0.25;
                adj.global.whites = 0.12;
                adj.global.blacks = -0.15;
                adj.global.temperature = 0.18;
                adj.global.tint = -0.12;
                adj.global.saturation = 0.18;
                adj.global.vibrance = 0.16;
                ("combined", adj)
            },
        ];

        for (label, adjustments) in cases {
            let wgsl_pixels = processor
                .run(&texture_view, width, height, adjustments, &[], None)
                .expect("WGSL run failed");
            let cubecl_result = cubecl_processing::process_with_cubecl(
                &image,
                adjustments,
                &[],
                None,
                Some(&wgsl_pixels),
            )
            .expect("CubeCL run failed");
            assert!(
                !cubecl_result.used_wgsl_fallback,
                "CubeCL unexpectedly fell back for {}: {:?}",
                label, cubecl_result.fallback_reason
            );
            let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_result.pixels, 8);
            assert_mask_diff(diff, width, height, 0.16, 40);
        }
    }

    #[test]
    fn cubecl_matches_wgsl_clipping_overlay_near_thresholds() {
        let Some(context) = test_gpu_context() else {
            return;
        };
        let width = 52;
        let height = 40;
        let image = make_hdr_test_image(width, height);
        let processor = GpuProcessor::new(
            context.clone(),
            width.next_multiple_of(256),
            height.next_multiple_of(256),
        )
        .expect("Failed to create GPU processor");
        let texture_view = make_test_texture(&context, &image, width, height);

        let cases: [(&str, AllAdjustments); 2] = [
            {
                let mut adj = AllAdjustments::default();
                adj.global.show_clipping = 1;
                adj.global.exposure = 1.8;
                ("highlight_clip", adj)
            },
            {
                let mut adj = AllAdjustments::default();
                adj.global.show_clipping = 1;
                adj.global.exposure = -2.2;
                ("shadow_clip", adj)
            },
        ];

        for (label, adjustments) in cases {
            let wgsl_pixels = processor
                .run(&texture_view, width, height, adjustments, &[], None)
                .expect("WGSL run failed");
            let cubecl_result = cubecl_processing::process_with_cubecl(
                &image,
                adjustments,
                &[],
                None,
                Some(&wgsl_pixels),
            )
            .expect("CubeCL run failed");
            assert!(
                !cubecl_result.used_wgsl_fallback,
                "CubeCL unexpectedly fell back for {}: {:?}",
                label, cubecl_result.fallback_reason
            );
            let diff = cubecl_processing::compare_images(&wgsl_pixels, &cubecl_result.pixels, 8);
            assert_mask_diff(diff, width, height, 0.12, 30);
        }
    }
}
