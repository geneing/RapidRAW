use std::time::{Duration, Instant};

use cubecl::prelude::*;
use image::{DynamicImage, GenericImageView};
use once_cell::sync::Lazy;

use crate::image_processing::{AllAdjustments, ColorCalibrationSettings, ColorGradeSettings};

static CUBECL_DEVICE: Lazy<cubecl::wgpu::WgpuDevice> =
    Lazy::new(cubecl::wgpu::WgpuDevice::default);

const FLARE_MAP_SIZE: u32 = 512;
const LN_2: f32 = 0.693_147_2;

#[derive(Debug, Clone, Copy)]
pub struct CubeclTimings {
    pub total: Duration,
    pub flare_threshold: Duration,
    pub flare_blur: Duration,
    pub main: Duration,
}

#[derive(Debug, Clone)]
pub struct CubeclRunResult {
    pub pixels: Vec<u8>,
    pub timings: CubeclTimings,
    pub used_wgsl_fallback: bool,
    pub fallback_reason: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct ImageDiffStats {
    pub compared_values: usize,
    pub mismatched_values: usize,
    pub max_abs_diff: u8,
    pub mean_abs_diff: f32,
}

#[cube(launch_unchecked)]
fn cubecl_copy_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
    }
}

#[cube(launch_unchecked)]
fn cubecl_blur_horizontal_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    radius: u32,
) {
    let pixel_index = ABSOLUTE_POS;
    let pixel_count = width * height;
    if pixel_index < pixel_count {
        let x = pixel_index % width;
        let y = pixel_index / width;
        let out_base = pixel_index * 4;

        if radius == 0 {
            output[out_base] = input[out_base];
            output[out_base + 1] = input[out_base + 1];
            output[out_base + 2] = input[out_base + 2];
            output[out_base + 3] = input[out_base + 3];
        } else {
            let sigma = f32::cast_from(radius) / 2.0;
            let two_sigma_sq = 2.0 * sigma * sigma;
            let span = radius * 2 + 1;
            let max_x = width as i32 - 1;
            let xi = x as i32;

            let mut total_weight = 0.0;
            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;
            let mut a = 0.0;

            for i in 0..span {
                let offset = i as i32 - radius as i32;
                let mut sx_i = xi + offset;
                if sx_i < 0 {
                    sx_i = 0;
                }
                if sx_i > max_x {
                    sx_i = max_x;
                }
                let sx = sx_i as u32;
                let sample_base = (y * width + sx) * 4;
                let off_f = f32::cast_from(offset);
                let weight = Exp::exp(-(off_f * off_f) / two_sigma_sq);

                r += input[sample_base] * weight;
                g += input[sample_base + 1] * weight;
                b += input[sample_base + 2] * weight;
                a += input[sample_base + 3] * weight;
                total_weight += weight;
            }

            output[out_base] = r / total_weight;
            output[out_base + 1] = g / total_weight;
            output[out_base + 2] = b / total_weight;
            output[out_base + 3] = a / total_weight;
        }
    }
}

#[cube(launch_unchecked)]
fn cubecl_blur_vertical_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    radius: u32,
) {
    let pixel_index = ABSOLUTE_POS;
    let pixel_count = width * height;
    if pixel_index < pixel_count {
        let x = pixel_index % width;
        let y = pixel_index / width;
        let out_base = pixel_index * 4;

        if radius == 0 {
            output[out_base] = input[out_base];
            output[out_base + 1] = input[out_base + 1];
            output[out_base + 2] = input[out_base + 2];
            output[out_base + 3] = input[out_base + 3];
        } else {
            let sigma = f32::cast_from(radius) / 2.0;
            let two_sigma_sq = 2.0 * sigma * sigma;
            let span = radius * 2 + 1;
            let max_y = height as i32 - 1;
            let yi = y as i32;

            let mut total_weight = 0.0;
            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;
            let mut a = 0.0;

            for i in 0..span {
                let offset = i as i32 - radius as i32;
                let mut sy_i = yi + offset;
                if sy_i < 0 {
                    sy_i = 0;
                }
                if sy_i > max_y {
                    sy_i = max_y;
                }
                let sy = sy_i as u32;
                let sample_base = (sy * width + x) * 4;
                let off_f = f32::cast_from(offset);
                let weight = Exp::exp(-(off_f * off_f) / two_sigma_sq);

                r += input[sample_base] * weight;
                g += input[sample_base + 1] * weight;
                b += input[sample_base + 2] * weight;
                a += input[sample_base + 3] * weight;
                total_weight += weight;
            }

            output[out_base] = r / total_weight;
            output[out_base + 1] = g / total_weight;
            output[out_base + 2] = b / total_weight;
            output[out_base + 3] = a / total_weight;
        }
    }
}

#[cube(launch_unchecked)]
fn cubecl_flare_threshold_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    input_width: u32,
    input_height: u32,
    out_size: u32,
    amount: f32,
    exposure: f32,
) {
    let pixel_index = ABSOLUTE_POS;
    let pixel_count = out_size * out_size;
    if pixel_index < pixel_count {
        let x = pixel_index % out_size;
        let y = pixel_index / out_size;
        let sx = (x * input_width) / out_size;
        let sy = (y * input_height) / out_size;
        let sample_base = (sy * input_width + sx) * 4;
        let out_base = pixel_index * 4;

        let exposure_factor = Exp::exp(exposure * LN_2);
        let r = input[sample_base] * exposure_factor;
        let g = input[sample_base + 1] * exposure_factor;
        let b = input[sample_base + 2] * exposure_factor;
        let luma = r * 0.2126 + g * 0.7152 + b * 0.0722;

        let mut amount_clamped = amount;
        if amount_clamped < 0.0 {
            amount_clamped = 0.0;
        }
        if amount_clamped > 1.0 {
            amount_clamped = 1.0;
        }
        let threshold_val = 0.88 + (0.50 - 0.88) * amount_clamped;
        let knee = 0.15;
        let x_thresh = luma - threshold_val + knee;

        let bright_contrib = if x_thresh <= 0.0 {
            f32::new(0.0)
        } else if x_thresh < knee * 2.0 {
            (x_thresh * x_thresh) / (knee * 4.0)
        } else {
            x_thresh - knee
        };
        let denom = if luma < 0.001 { f32::new(0.001) } else { luma };
        let scale = bright_contrib / denom;

        output[out_base] = r * scale;
        output[out_base + 1] = g * scale;
        output[out_base + 2] = b * scale;
        output[out_base + 3] = 1.0;
    }
}

#[cube(launch_unchecked)]
fn cubecl_main_flare_composite_kernel(
    input: &Array<f32>,
    flare: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    flare_size: u32,
    flare_amount: f32,
) {
    let pixel_index = ABSOLUTE_POS;
    let pixel_count = width * height;
    if pixel_index < pixel_count {
        let x = pixel_index % width;
        let y = pixel_index / width;
        let base = pixel_index * 4;

        let mut r = input[base];
        let mut g = input[base + 1];
        let mut b = input[base + 2];
        let a = input[base + 3];

        if flare_amount > 0.0 {
            let fx = (x * flare_size) / width;
            let fy = (y * flare_size) / height;
            let flare_base = (fy * flare_size + fx) * 4;

            let mut fr = flare[flare_base] * 1.4;
            let mut fg = flare[flare_base + 1] * 1.4;
            let mut fb = flare[flare_base + 2] * 1.4;

            fr = fr * fr;
            fg = fg * fg;
            fb = fb * fb;

            let safe_r = if r < 0.0 { f32::new(0.0) } else { r };
            let safe_g = if g < 0.0 { f32::new(0.0) } else { g };
            let safe_b = if b < 0.0 { f32::new(0.0) } else { b };
            let luma = safe_r * 0.2126 + safe_g * 0.7152 + safe_b * 0.0722;

            let mut t = (luma - 0.7) / (1.8 - 0.7);
            if t < 0.0 {
                t = 0.0;
            }
            if t > 1.0 {
                t = 1.0;
            }
            let smooth = t * t * (3.0 - 2.0 * t);
            let protection = 1.0 - smooth;

            r += fr * flare_amount * protection;
            g += fg * flare_amount * protection;
            b += fb * flare_amount * protection;
        }

        if r < 0.0 {
            r = 0.0;
        }
        if g < 0.0 {
            g = 0.0;
        }
        if b < 0.0 {
            b = 0.0;
        }
        if r > 1.0 {
            r = 1.0;
        }
        if g > 1.0 {
            g = 1.0;
        }
        if b > 1.0 {
            b = 1.0;
        }

        output[base] = r;
        output[base + 1] = g;
        output[base + 2] = b;
        output[base + 3] = a;
    }
}
fn dispatch_count_1d(len: usize, group_size: u32) -> u32 {
    let groups = (len as u32 + group_size - 1) / group_size;
    if groups == 0 { 1 } else { groups }
}

fn run_copy_kernel(client: &ComputeClient<cubecl::wgpu::WgpuServer>, input: &[f32]) -> Vec<f32> {
    let values = input.len();
    let input_handle = client.create(f32::as_bytes(input));
    let output_handle = client.empty(values * std::mem::size_of::<f32>());
    let cube_dim_x = 256u32;

    unsafe {
        cubecl_copy_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            client,
            CubeCount::Static(dispatch_count_1d(values, cube_dim_x), 1, 1),
            CubeDim::new_1d(cube_dim_x),
            ArrayArg::from_raw_parts::<f32>(&input_handle, values, 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, values, 1),
        );
    }

    let bytes = client.read_one(output_handle);
    let out: &[f32] = f32::from_bytes(&bytes);
    out.to_vec()
}

fn run_gaussian_blur_kernel(
    client: &ComputeClient<cubecl::wgpu::WgpuServer>,
    input: &[f32],
    width: u32,
    height: u32,
    radius: u32,
) -> Vec<f32> {
    let pixel_count = (width * height) as usize;
    let values = pixel_count * 4;
    let cube_dim_x = 256u32;

    let input_handle = client.create(f32::as_bytes(input));
    let temp_handle = client.empty(values * std::mem::size_of::<f32>());
    let output_handle = client.empty(values * std::mem::size_of::<f32>());

    unsafe {
        cubecl_blur_horizontal_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            client,
            CubeCount::Static(dispatch_count_1d(pixel_count, cube_dim_x), 1, 1),
            CubeDim::new_1d(cube_dim_x),
            ArrayArg::from_raw_parts::<f32>(&input_handle, values, 1),
            ArrayArg::from_raw_parts::<f32>(&temp_handle, values, 1),
            ScalarArg::new(width),
            ScalarArg::new(height),
            ScalarArg::new(radius),
        );
    }

    unsafe {
        cubecl_blur_vertical_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            client,
            CubeCount::Static(dispatch_count_1d(pixel_count, cube_dim_x), 1, 1),
            CubeDim::new_1d(cube_dim_x),
            ArrayArg::from_raw_parts::<f32>(&temp_handle, values, 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, values, 1),
            ScalarArg::new(width),
            ScalarArg::new(height),
            ScalarArg::new(radius),
        );
    }

    let bytes = client.read_one(output_handle);
    let out: &[f32] = f32::from_bytes(&bytes);
    out.to_vec()
}

fn run_flare_threshold_kernel(
    client: &ComputeClient<cubecl::wgpu::WgpuServer>,
    input: &[f32],
    input_width: u32,
    input_height: u32,
    all_adjustments: AllAdjustments,
) -> Vec<f32> {
    let out_size = FLARE_MAP_SIZE;
    let out_pixels = (out_size * out_size) as usize;
    let out_values = out_pixels * 4;
    let cube_dim_x = 256u32;

    let input_handle = client.create(f32::as_bytes(input));
    let output_handle = client.empty(out_values * std::mem::size_of::<f32>());

    unsafe {
        cubecl_flare_threshold_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            client,
            CubeCount::Static(dispatch_count_1d(out_pixels, cube_dim_x), 1, 1),
            CubeDim::new_1d(cube_dim_x),
            ArrayArg::from_raw_parts::<f32>(&input_handle, input.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, out_values, 1),
            ScalarArg::new(input_width),
            ScalarArg::new(input_height),
            ScalarArg::new(out_size),
            ScalarArg::new(all_adjustments.global.flare_amount),
            ScalarArg::new(all_adjustments.global.exposure),
        );
    }

    let bytes = client.read_one(output_handle);
    let out: &[f32] = f32::from_bytes(&bytes);
    out.to_vec()
}

fn run_main_flare_composite_kernel(
    client: &ComputeClient<cubecl::wgpu::WgpuServer>,
    input: &[f32],
    flare: Option<&[f32]>,
    width: u32,
    height: u32,
    flare_amount: f32,
) -> Vec<f32> {
    let pixel_count = (width * height) as usize;
    let values = pixel_count * 4;
    let cube_dim_x = 256u32;

    let input_handle = client.create(f32::as_bytes(input));
    let flare_data = if let Some(flare_data) = flare {
        flare_data
    } else {
        input
    };
    let flare_handle = client.create(f32::as_bytes(flare_data));
    let output_handle = client.empty(values * std::mem::size_of::<f32>());

    unsafe {
        cubecl_main_flare_composite_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            client,
            CubeCount::Static(dispatch_count_1d(pixel_count, cube_dim_x), 1, 1),
            CubeDim::new_1d(cube_dim_x),
            ArrayArg::from_raw_parts::<f32>(&input_handle, values, 1),
            ArrayArg::from_raw_parts::<f32>(&flare_handle, flare_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, values, 1),
            ScalarArg::new(width),
            ScalarArg::new(height),
            ScalarArg::new(FLARE_MAP_SIZE),
            ScalarArg::new(flare_amount),
        );
    }

    let bytes = client.read_one(output_handle);
    let out: &[f32] = f32::from_bytes(&bytes);
    out.to_vec()
}

fn near_zero(v: f32) -> bool {
    v.abs() < 1.0e-6
}

fn color_grade_is_zero(cg: ColorGradeSettings) -> bool {
    near_zero(cg.hue) && near_zero(cg.saturation) && near_zero(cg.luminance)
}

fn color_calibration_is_zero(c: ColorCalibrationSettings) -> bool {
    near_zero(c.shadows_tint)
        && near_zero(c.red_hue)
        && near_zero(c.red_saturation)
        && near_zero(c.green_hue)
        && near_zero(c.green_saturation)
        && near_zero(c.blue_hue)
        && near_zero(c.blue_saturation)
}

fn unsupported_reason(adjustments: &AllAdjustments) -> Option<&'static str> {
    let g = adjustments.global;

    if adjustments.mask_count != 0 {
        return Some("mask adjustments are not yet implemented in CubeCL path");
    }
    if g.has_lut != 0 {
        return Some("LUT path is not yet implemented in CubeCL");
    }
    if g.tonemapper_mode != 0 {
        return Some("AgX tonemapper is not yet implemented in CubeCL");
    }
    if g.show_clipping != 0 {
        return Some("clipping overlay is not yet implemented in CubeCL");
    }
    if !near_zero(g.sharpness) || !near_zero(g.clarity) || !near_zero(g.structure) {
        return Some("local contrast blur path is not yet implemented in CubeCL");
    }
    if !near_zero(g.dehaze)
        || !near_zero(g.glow_amount)
        || !near_zero(g.halation_amount)
        || !near_zero(g.vignette_amount)
        || !near_zero(g.vignette_midpoint)
        || !near_zero(g.vignette_roundness)
        || !near_zero(g.vignette_feather)
        || !near_zero(g.grain_amount)
        || !near_zero(g.grain_size)
        || !near_zero(g.grain_roughness)
        || !near_zero(g.chromatic_aberration_red_cyan)
        || !near_zero(g.chromatic_aberration_blue_yellow)
    {
        return Some("advanced effects are not yet implemented in CubeCL");
    }
    if !near_zero(g.brightness)
        || !near_zero(g.contrast)
        || !near_zero(g.highlights)
        || !near_zero(g.shadows)
        || !near_zero(g.whites)
        || !near_zero(g.blacks)
        || !near_zero(g.temperature)
        || !near_zero(g.tint)
        || !near_zero(g.saturation)
        || !near_zero(g.vibrance)
    {
        return Some("main shader tonal/color controls are not fully implemented in CubeCL");
    }
    if !color_grade_is_zero(g.color_grading_shadows)
        || !color_grade_is_zero(g.color_grading_midtones)
        || !color_grade_is_zero(g.color_grading_highlights)
        || !near_zero(g.color_grading_blending)
        || !near_zero(g.color_grading_balance)
    {
        return Some("color grading is not yet implemented in CubeCL");
    }
    if !color_calibration_is_zero(g.color_calibration) {
        return Some("color calibration is not yet implemented in CubeCL");
    }
    if g.luma_curve_count != 0 || g.red_curve_count != 0 || g.green_curve_count != 0 || g.blue_curve_count != 0 {
        return Some("curves are not yet implemented in CubeCL");
    }

    None
}

fn f32_rgba_to_u8(pixels: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixels.len());
    for &v in pixels {
        out.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
    }
    out
}

pub fn process_with_cubecl(
    base_image: &DynamicImage,
    all_adjustments: AllAdjustments,
    wgsl_fallback_pixels: Option<&[u8]>,
) -> Result<CubeclRunResult, String> {
    let total_start = Instant::now();
    let (width, height) = base_image.dimensions();

    if let Some(reason) = unsupported_reason(&all_adjustments) {
        if let Some(fallback) = wgsl_fallback_pixels {
            return Ok(CubeclRunResult {
                pixels: fallback.to_vec(),
                timings: CubeclTimings {
                    total: total_start.elapsed(),
                    flare_threshold: Duration::ZERO,
                    flare_blur: Duration::ZERO,
                    main: Duration::ZERO,
                },
                used_wgsl_fallback: true,
                fallback_reason: Some(reason.to_string()),
            });
        }
        return Err(format!(
            "CubeCL path cannot execute this adjustment set without WGSL fallback: {}",
            reason
        ));
    }

    let rgba_f32 = base_image.to_rgba32f();
    let input = rgba_f32.as_raw();
    let client = cubecl::wgpu::WgpuRuntime::client(&*CUBECL_DEVICE);

    let mut flare_threshold_time = Duration::ZERO;
    let mut flare_blur_time = Duration::ZERO;

    let flare_map = if all_adjustments.global.flare_amount > 0.0 {
        let flare_threshold_start = Instant::now();
        let threshold = run_flare_threshold_kernel(&client, input, width, height, all_adjustments);
        flare_threshold_time = flare_threshold_start.elapsed();

        let flare_blur_start = Instant::now();
        let flare_blurred =
            run_gaussian_blur_kernel(&client, &threshold, FLARE_MAP_SIZE, FLARE_MAP_SIZE, 12);
        flare_blur_time = flare_blur_start.elapsed();
        Some(flare_blurred)
    } else {
        None
    };

    let main_start = Instant::now();
    let output = if let Some(flare_map) = flare_map.as_ref() {
        run_main_flare_composite_kernel(
            &client,
            input,
            Some(flare_map),
            width,
            height,
            all_adjustments.global.flare_amount,
        )
    } else {
        run_copy_kernel(&client, input)
    };
    let main_time = main_start.elapsed();

    Ok(CubeclRunResult {
        pixels: f32_rgba_to_u8(&output),
        timings: CubeclTimings {
            total: total_start.elapsed(),
            flare_threshold: flare_threshold_time,
            flare_blur: flare_blur_time,
            main: main_time,
        },
        used_wgsl_fallback: false,
        fallback_reason: None,
    })
}

pub fn compare_images(wgsl_pixels: &[u8], cubecl_pixels: &[u8], tolerance: u8) -> ImageDiffStats {
    let compared_values = wgsl_pixels.len().min(cubecl_pixels.len());
    if compared_values == 0 {
        return ImageDiffStats {
            compared_values: 0,
            mismatched_values: 0,
            max_abs_diff: 0,
            mean_abs_diff: 0.0,
        };
    }

    let mut mismatched_values = 0usize;
    let mut max_abs_diff = 0u8;
    let mut diff_sum = 0f64;

    for i in 0..compared_values {
        let abs_diff = wgsl_pixels[i].abs_diff(cubecl_pixels[i]);
        if abs_diff > tolerance {
            mismatched_values += 1;
        }
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }
        diff_sum += abs_diff as f64;
    }

    ImageDiffStats {
        compared_values,
        mismatched_values,
        max_abs_diff,
        mean_abs_diff: (diff_sum / compared_values as f64) as f32,
    }
}
