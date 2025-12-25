use std::sync::Arc;
use std::time::Instant;

use bytemuck;
use half::f16;
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Rgba};
use wgpu::util::{DeviceExt, TextureDataOrder};

use crate::image_processing::{AllAdjustments, GpuContext};
use crate::lut_processing::Lut;
use crate::{AppState, GpuImageCache};

/// Initialize GPU context without requiring AppState. Used by both Tauri and CLI modes.
pub fn init_gpu_context() -> Result<GpuContext, String> {
    let mut instance_desc = wgpu::InstanceDescriptor::from_env_or_default();
    #[cfg(debug_assertions)]
    {
        instance_desc.flags |= wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DEBUG;
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
    #[cfg(debug_assertions)]
    {
        let adapter_features = adapter.features();
        if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES) {
            required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        }
    }

    let limits = adapter.limits();
    let trace = if cfg!(debug_assertions) {
        #[cfg(feature = "trace")]
        {
            match std::env::var("RAPIDRAW_WGPU_TRACE_DIR") {
                Ok(dir) if !dir.is_empty() => {
                    if let Err(err) = std::fs::create_dir_all(&dir) {
                        log::warn!("Failed to create wgpu trace dir {}: {}", dir, err);
                        wgpu::Trace::Off
                    } else {
                        wgpu::Trace::Directory(dir.into())
                    }
                }
                _ => wgpu::Trace::Off,
            }
        }
        #[cfg(not(feature = "trace"))]
        {
            if std::env::var("RAPIDRAW_WGPU_TRACE_DIR").is_ok() {
                log::warn!("RAPIDRAW_WGPU_TRACE_DIR set but wgpu \"trace\" feature is disabled");
            }
            wgpu::Trace::Off
        }
    } else {
        wgpu::Trace::Off
    };

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Processing Device"),
            required_features,
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace,
        },
    ))
    .map_err(|e| e.to_string())?;

    Ok(GpuContext {
        device: Arc::new(device),
        queue: Arc::new(queue),
        limits,
    })
}

pub fn get_or_init_gpu_context(state: &tauri::State<AppState>) -> Result<GpuContext, String> {
    let mut context_lock = state.gpu_context.lock().unwrap();
    if let Some(context) = &*context_lock {
        return Ok(context.clone());
    }
    let new_context = init_gpu_context()?;
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
    rgba_f32
        .into_raw()
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurParams {
    radius: u32,
    tile_offset_x: u32,
    tile_offset_y: u32,
    tile_width: u32,
    tile_height: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[cfg(debug_assertions)]
struct GpuTimingRecord {
    label: String,
    start: u32,
    end: u32,
}

#[cfg(debug_assertions)]
struct GpuProfiler {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    timestamp_period: f32,
    next_query: u32,
    records: Vec<GpuTimingRecord>,
}

#[cfg(debug_assertions)]
impl GpuProfiler {
    fn new(device: &wgpu::Device, timestamp_period: f32, query_count: u32) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GPU Timing Query Set"),
            ty: wgpu::QueryType::Timestamp,
            count: query_count,
        });
        let buffer_size = (query_count as u64) * std::mem::size_of::<u64>() as u64;
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Timing Resolve Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Timing Readback Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            timestamp_period,
            next_query: 0,
            records: Vec::new(),
        }
    }

    fn write_timestamp(&mut self, pass: &mut wgpu::ComputePass) -> u32 {
        let index = self.next_query;
        pass.write_timestamp(&self.query_set, index);
        self.next_query += 1;
        index
    }

    fn record(&mut self, label: String, start: u32, end: u32) {
        self.records.push(GpuTimingRecord { label, start, end });
    }

    fn resolve_and_log(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.next_query == 0 {
            return;
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPU Timing Resolve Encoder"),
        });
        encoder.resolve_query_set(
            &self.query_set,
            0..self.next_query,
            &self.resolve_buffer,
            0,
        );
        let byte_count = (self.next_query as u64) * std::mem::size_of::<u64>() as u64;
        encoder.copy_buffer_to_buffer(&self.resolve_buffer, 0, &self.readback_buffer, 0, byte_count);
        queue.submit(Some(encoder.finish()));

        let buffer_slice = self.readback_buffer.slice(..byte_count);
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
        if let Err(err) = rx.recv().unwrap() {
            log::warn!("GPU timing readback failed: {}", err);
            self.readback_buffer.unmap();
            return;
        }

        let data = buffer_slice.get_mapped_range();
        let mut timestamps = Vec::with_capacity(self.next_query as usize);
        for chunk in data.chunks_exact(8) {
            timestamps.push(u64::from_le_bytes(chunk.try_into().unwrap()));
        }
        drop(data);
        self.readback_buffer.unmap();

        let period_ns = self.timestamp_period as f64;
        for record in &self.records {
            let start = timestamps.get(record.start as usize).copied().unwrap_or(0);
            let end = timestamps.get(record.end as usize).copied().unwrap_or(0);
            if end > start {
                let duration_ns = (end - start) as f64 * period_ns;
                log::debug!("GPU timing {}: {:.3} us", record.label, duration_ns / 1000.0);
            }
        }
    }
}

#[cfg(not(debug_assertions))]
struct GpuProfiler;

#[cfg(not(debug_assertions))]
impl GpuProfiler {
    fn new(_device: &wgpu::Device, _timestamp_period: f32, _query_count: u32) -> Self {
        Self
    }

    fn write_timestamp(&mut self, _pass: &mut wgpu::ComputePass) -> u32 {
        0
    }

    fn record(&mut self, _label: String, _start: u32, _end: u32) {}

    fn resolve_and_log(&self, _device: &wgpu::Device, _queue: &wgpu::Queue) {}
}

struct GpuProcessor<'a> {
    context: &'a GpuContext,
    blur_bgl: wgpu::BindGroupLayout,
    h_blur_pipeline: wgpu::ComputePipeline,
    v_blur_pipeline: wgpu::ComputePipeline,
    blur_params_buffer: wgpu::Buffer,
    main_bgl: wgpu::BindGroupLayout,
    main_pipeline: wgpu::ComputePipeline,
    adjustments_buffer: wgpu::Buffer,
    dummy_blur_view: wgpu::TextureView,
    dummy_mask_view: wgpu::TextureView,
    lut_texture_view: wgpu::TextureView,
    lut_sampler: wgpu::Sampler,
    mask_views: Vec<wgpu::TextureView>,
}

impl<'a> GpuProcessor<'a> {
    fn new(
        context: &'a GpuContext,
        width: u32,
        height: u32,
        mask_bitmaps: &[ImageBuffer<Luma<u8>, Vec<u8>>],
        lut: Option<Arc<Lut>>,
    ) -> Result<Self, String> {
        let device = &context.device;
        let queue = &context.queue;
        const MAX_MASKS: u32 = 11;

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
            (dummy_lut_view.clone(), dummy_lut_sampler)
        };

        Ok(Self {
            context,
            blur_bgl,
            h_blur_pipeline,
            v_blur_pipeline,
            blur_params_buffer,
            main_bgl,
            main_pipeline,
            adjustments_buffer,
            dummy_blur_view,
            dummy_mask_view,
            lut_texture_view,
            lut_sampler,
            mask_views,
        })
    }

    fn run(
        &self,
        input_texture_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        adjustments: AllAdjustments,
    ) -> Result<Vec<u8>, String> {
        let device = &self.context.device;
        let queue = &self.context.queue;
        let scale = (width.min(height) as f32) / 1080.0;
        const MAX_MASKS: u32 = 11;
        let profiling_enabled = cfg!(debug_assertions)
            && device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
            && device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);

        const TILE_SIZE: u32 = 2048;
        const TILE_OVERLAP: u32 = 128;
        let max_tile_input_dim = TILE_SIZE + 2 * TILE_OVERLAP;

        let max_tile_size = wgpu::Extent3d {
            width: max_tile_input_dim,
            height: max_tile_input_dim,
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

        let mut final_pixels = vec![0u8; (width * height * 4) as usize];
        let tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
        let tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;

        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                let mut tile_profiler = if profiling_enabled {
                    Some(GpuProfiler::new(
                        device,
                        queue.get_timestamp_period(),
                        16,
                    ))
                } else {
                    None
                };
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

                let create_blur = |label: &str,
                                   base_radius: f32,
                                   output_view: &wgpu::TextureView,
                                   profiler: &mut Option<GpuProfiler>|
                 -> bool {
                        #[cfg(not(debug_assertions))]
                        let _ = label;
                        let radius = (base_radius * scale).ceil().max(1.0) as u32;
                        if radius == 0 {
                            return false;
                        }

                        let params = BlurParams {
                            radius,
                            tile_offset_x: input_x_start,
                            tile_offset_y: input_y_start,
                            tile_width: input_width,
                            tile_height: input_height,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        };
                        queue.write_buffer(&self.blur_params_buffer, 0, bytemuck::bytes_of(&params));

                        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Blur Encoder"),
                        });

                        let h_blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("H-Blur BG"),
                            layout: &self.blur_bgl,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        input_texture_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(&ping_pong_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: self.blur_params_buffer.as_entire_binding(),
                                },
                            ],
                        });

                        #[cfg(debug_assertions)]
                        encoder.push_debug_group("horizontal_blur");
                        {
                            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("Horizontal Blur Pass"),
                                ..Default::default()
                            });
                            cpass.set_pipeline(&self.h_blur_pipeline);
                            cpass.set_bind_group(0, &h_blur_bg, &[]);
                            #[cfg(debug_assertions)]
                            let start = profiler.as_mut().map(|p| p.write_timestamp(&mut cpass));
                            cpass.dispatch_workgroups((input_width + 255) / 256, input_height, 1);
                            #[cfg(debug_assertions)]
                            if let Some((p, start)) = profiler.as_mut().zip(start) {
                                let end = p.write_timestamp(&mut cpass);
                                p.record(
                                    format!("tile({},{})/{}/horizontal_blur", tile_x, tile_y, label),
                                    start,
                                    end,
                                );
                            }
                        }
                        #[cfg(debug_assertions)]
                        encoder.pop_debug_group();

                        let v_blur_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("V-Blur BG"),
                            layout: &self.blur_bgl,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(&ping_pong_view),
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

                        #[cfg(debug_assertions)]
                        encoder.push_debug_group("vertical_blur");
                        {
                            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("Vertical Blur Pass"),
                                ..Default::default()
                            });
                            cpass.set_pipeline(&self.v_blur_pipeline);
                            cpass.set_bind_group(0, &v_blur_bg, &[]);
                            #[cfg(debug_assertions)]
                            let start = profiler.as_mut().map(|p| p.write_timestamp(&mut cpass));
                            cpass.dispatch_workgroups(input_width, (input_height + 255) / 256, 1);
                            #[cfg(debug_assertions)]
                            if let Some((p, start)) = profiler.as_mut().zip(start) {
                                let end = p.write_timestamp(&mut cpass);
                                p.record(
                                    format!("tile({},{})/{}/vertical_blur", tile_x, tile_y, label),
                                    start,
                                    end,
                                );
                            }
                        }
                        #[cfg(debug_assertions)]
                        encoder.pop_debug_group();

                        queue.submit(Some(encoder.finish()));
                        true
                    };

                let did_create_sharpness_blur =
                    create_blur("sharpness", 2.0, &sharpness_blur_view, &mut tile_profiler);
                let did_create_clarity_blur =
                    create_blur("clarity", 8.0, &clarity_blur_view, &mut tile_profiler);
                let did_create_structure_blur =
                    create_blur("structure", 40.0, &structure_blur_view, &mut tile_profiler);

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
                        resource: wgpu::BindingResource::TextureView(&output_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.adjustments_buffer.as_entire_binding(),
                    },
                ];
                for i in 0..MAX_MASKS as usize {
                    let view = self.mask_views.get(i).unwrap_or(&self.dummy_mask_view);
                    bind_group_entries.push(wgpu::BindGroupEntry {
                        binding: 3 + i as u32,
                        resource: wgpu::BindingResource::TextureView(view),
                    });
                }
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 3 + MAX_MASKS,
                    resource: wgpu::BindingResource::TextureView(&self.lut_texture_view),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 4 + MAX_MASKS,
                    resource: wgpu::BindingResource::Sampler(&self.lut_sampler),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 5 + MAX_MASKS,
                    resource: wgpu::BindingResource::TextureView(if did_create_sharpness_blur {
                        &sharpness_blur_view
                    } else {
                        &self.dummy_blur_view
                    }),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 6 + MAX_MASKS,
                    resource: wgpu::BindingResource::TextureView(if did_create_clarity_blur {
                        &clarity_blur_view
                    } else {
                        &self.dummy_blur_view
                    }),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 7 + MAX_MASKS,
                    resource: wgpu::BindingResource::TextureView(if did_create_structure_blur {
                        &structure_blur_view
                    } else {
                        &self.dummy_blur_view
                    }),
                });

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Tile Bind Group"),
                    layout: &self.main_bgl,
                    entries: &bind_group_entries,
                });

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Main Compute Encoder"),
                });
                #[cfg(debug_assertions)]
                encoder.push_debug_group("main_compute");
                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Main Compute Pass"),
                            ..Default::default()
                        });
                    compute_pass.set_pipeline(&self.main_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    #[cfg(debug_assertions)]
                    let start = tile_profiler
                        .as_mut()
                        .map(|p| p.write_timestamp(&mut compute_pass));
                    compute_pass.dispatch_workgroups(
                        (input_width + 7) / 8,
                        (input_height + 7) / 8,
                        1,
                    );
                    #[cfg(debug_assertions)]
                    if let Some((p, start)) = tile_profiler.as_mut().zip(start) {
                        let end = p.write_timestamp(&mut compute_pass);
                        p.record(
                            format!("tile({},{})/main_compute", tile_x, tile_y),
                            start,
                            end,
                        );
                    }
                }
                #[cfg(debug_assertions)]
                encoder.pop_debug_group();
                queue.submit(Some(encoder.finish()));

                if let Some(profiler) = tile_profiler.as_ref() {
                    profiler.resolve_and_log(device, queue);
                }

                let processed_tile_data =
                    read_texture_data(device, queue, &output_texture, input_texture_size)?;

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

pub fn run_gpu_processing(
    context: &GpuContext,
    input_texture_view: &wgpu::TextureView,
    width: u32,
    height: u32,
    adjustments: AllAdjustments,
    mask_bitmaps: &[ImageBuffer<Luma<u8>, Vec<u8>>],
    lut: Option<Arc<Lut>>,
) -> Result<Vec<u8>, String> {
    let start_time = Instant::now();
    let max_dim = context.limits.max_texture_dimension_2d;

    if width > max_dim || height > max_dim {
        return Err(format!(
            "Image dimensions ({}x{}) exceed GPU limits ({}).",
            width, height, max_dim
        ));
    }

    let processor = GpuProcessor::new(context, width, height, mask_bitmaps, lut)?;
    let final_pixels = processor.run(input_texture_view, width, height, adjustments)?;

    let duration = start_time.elapsed();
    log::info!(
        "GPU adjustments for {}x{} image took {:?}",
        width,
        height,
        duration
    );
    Ok(final_pixels)
}

/// CLI-friendly GPU processing that doesn't require Tauri state
pub fn process_image_gpu_cli(
    context: &GpuContext,
    base_image: &DynamicImage,
    all_adjustments: AllAdjustments,
    mask_bitmaps: &[ImageBuffer<Luma<u8>, Vec<u8>>],
    lut: Option<Arc<Lut>>,
) -> Result<DynamicImage, String> {
    let (width, height) = base_image.dimensions();
    log::info!(
        "[CLI] GPU processing called for {}x{} image.",
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

    // Convert image to GPU-compatible format
    let img_rgba_f16 = to_rgba_f16(base_image);
    let texture_size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("CLI Input Texture"),
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

    // Run GPU processing
    let processed_pixels = run_gpu_processing(
        context,
        &texture_view,
        width,
        height,
        all_adjustments,
        mask_bitmaps,
        lut,
    )?;

    // Convert result back to DynamicImage
    let img_buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, processed_pixels)
        .ok_or("Failed to create image buffer from GPU data")?;
    Ok(DynamicImage::ImageRgba8(img_buf))
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

    let processed_pixels = run_gpu_processing(
        context,
        &cache.texture_view,
        cache.width,
        cache.height,
        all_adjustments,
        mask_bitmaps,
        lut,
    )?;

    let img_buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, processed_pixels)
        .ok_or("Failed to create image buffer from GPU data")?;
    Ok(DynamicImage::ImageRgba8(img_buf))
}
