use std::time::{Duration, Instant};

use cubecl::prelude::*;
use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};
use once_cell::sync::Lazy;

use crate::image_processing::{
    AllAdjustments, ColorCalibrationSettings, ColorGradeSettings, HslColor, MaskAdjustments, Point,
};
use crate::lut_processing::Lut;

static CUBECL_DEVICE: Lazy<cubecl::wgpu::WgpuDevice> = Lazy::new(cubecl::wgpu::WgpuDevice::default);

const FLARE_MAP_SIZE: u32 = 512;
const CUBECL_TILE_SIZE: u32 = 1024;
const CUBECL_TILE_OVERLAP: u32 = 64;
const LN_2: f32 = 0.693_147_2;
const AGX_EPSILON: f32 = 1.0e-6;
const AGX_MIN_EV: f32 = -15.2;
const AGX_MAX_EV: f32 = 5.0;
const AGX_RANGE_EV: f32 = AGX_MAX_EV - AGX_MIN_EV;
const AGX_GAMMA: f32 = 2.4;
const AGX_SLOPE: f32 = 2.3843;
const AGX_TOE_POWER: f32 = 1.5;
const AGX_SHOULDER_POWER: f32 = 1.5;
const AGX_TOE_TRANSITION_X: f32 = 0.606_060_6;
const AGX_TOE_TRANSITION_Y: f32 = 0.43446;
const AGX_SHOULDER_TRANSITION_X: f32 = 0.606_060_6;
const AGX_SHOULDER_TRANSITION_Y: f32 = 0.43446;
const AGX_INTERCEPT: f32 = -1.0112;
const AGX_TOE_SCALE: f32 = -1.0359;
const AGX_SHOULDER_SCALE: f32 = 1.3475;
const AGX_TARGET_BLACK_PRE_GAMMA: f32 = 0.0;
const AGX_TARGET_WHITE_PRE_GAMMA: f32 = 1.0;

#[derive(Debug, Clone, Copy)]
pub struct CubeclTimings {
    pub total: Duration,
    pub flare_threshold: Duration,
    pub flare_blur: Duration,
    pub main: Duration,
    pub mask_composite: Duration,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MaskCompositeStats {
    pub mask_count: u32,
    pub active_pixels: usize,
    pub max_influence: f32,
    pub mean_influence: f32,
}

#[derive(Debug, Clone)]
pub struct CubeclRunResult {
    pub pixels: Vec<u8>,
    pub timings: CubeclTimings,
    pub mask_stats: MaskCompositeStats,
    pub used_wgsl_fallback: bool,
    pub fallback_reason: Option<String>,
    pub parity_dashboard: String,
}

#[derive(Debug, Clone, Copy)]
struct CubeclParityFlags {
    dehaze: bool,
    glow: bool,
    halation: bool,
    vignette: bool,
    grain: bool,
    chromatic_aberration: bool,
    brightness: bool,
    contrast: bool,
    highlights: bool,
    shadows: bool,
    whites: bool,
    blacks: bool,
    temperature: bool,
    tint: bool,
    saturation: bool,
    vibrance: bool,
    clipping_overlay: bool,
}

fn cubecl_flag(name: &str, default_enabled: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "0" | "false" | "off" | "no" | "disabled" => false,
            "1" | "true" | "on" | "yes" | "enabled" => true,
            _ => default_enabled,
        },
        Err(_) => default_enabled,
    }
}

fn cubecl_parity_flags() -> CubeclParityFlags {
    CubeclParityFlags {
        dehaze: cubecl_flag("RAPIDRAW_CUBECL_ENABLE_DEHAZE", true),
        glow: cubecl_flag("RAPIDRAW_CUBECL_ENABLE_GLOW", true),
        halation: cubecl_flag("RAPIDRAW_CUBECL_ENABLE_HALATION", true),
        vignette: cubecl_flag("RAPIDRAW_CUBECL_ENABLE_VIGNETTE", true),
        grain: cubecl_flag("RAPIDRAW_CUBECL_ENABLE_GRAIN", true),
        chromatic_aberration: cubecl_flag("RAPIDRAW_CUBECL_ENABLE_CA", true),
        brightness: cubecl_flag("RAPIDRAW_CUBECL_ENABLE_BRIGHTNESS", true),
        contrast: cubecl_flag("RAPIDRAW_CUBECL_ENABLE_CONTRAST", true),
        highlights: cubecl_flag("RAPIDRAW_CUBECL_ENABLE_HIGHLIGHTS", true),
        shadows: cubecl_flag("RAPIDRAW_CUBECL_ENABLE_SHADOWS", true),
        whites: cubecl_flag("RAPIDRAW_CUBECL_ENABLE_WHITES", true),
        blacks: cubecl_flag("RAPIDRAW_CUBECL_ENABLE_BLACKS", true),
        temperature: cubecl_flag("RAPIDRAW_CUBECL_ENABLE_TEMPERATURE", true),
        tint: cubecl_flag("RAPIDRAW_CUBECL_ENABLE_TINT", true),
        saturation: cubecl_flag("RAPIDRAW_CUBECL_ENABLE_SATURATION", true),
        vibrance: cubecl_flag("RAPIDRAW_CUBECL_ENABLE_VIBRANCE", true),
        clipping_overlay: cubecl_flag("RAPIDRAW_CUBECL_ENABLE_CLIPPING", true),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImageDiffStats {
    pub compared_values: usize,
    pub mismatched_values: usize,
    pub max_abs_diff: u8,
    pub mean_abs_diff: f32,
}

#[derive(Debug, Clone)]
struct CubeclLutBuffer {
    size: u32,
    rgba: Vec<f32>,
}

impl CubeclLutBuffer {
    fn from_lut(lut: &Lut) -> Result<Self, String> {
        let expected = (lut.size as usize)
            .saturating_mul(lut.size as usize)
            .saturating_mul(lut.size as usize)
            .saturating_mul(3);
        if lut.data.len() != expected {
            return Err(format!(
                "invalid LUT buffer length: expected {} RGB values for size {}, got {}",
                expected,
                lut.size,
                lut.data.len()
            ));
        }

        // Matches WGSL texture layout: 3D texture texel order with X-major RGBA texels.
        let mut rgba = Vec::with_capacity((lut.data.len() / 3) * 4);
        for chunk in lut.data.chunks_exact(3) {
            rgba.push(chunk[0]);
            rgba.push(chunk[1]);
            rgba.push(chunk[2]);
            rgba.push(1.0);
        }

        Ok(Self {
            size: if lut.size == 0 { 1 } else { lut.size },
            rgba,
        })
    }

    #[inline]
    fn load_rgb(&self, x: u32, y: u32, z: u32) -> [f32; 3] {
        let size = self.size;
        let max_coord = size - 1;
        let cx = if x > max_coord { max_coord } else { x };
        let cy = if y > max_coord { max_coord } else { y };
        let cz = if z > max_coord { max_coord } else { z };
        let idx = (((cz * size + cy) * size + cx) * 4) as usize;
        [
            self.rgba.get(idx).copied().unwrap_or(0.0),
            self.rgba.get(idx + 1).copied().unwrap_or(0.0),
            self.rgba.get(idx + 2).copied().unwrap_or(0.0),
        ]
    }
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
    full_width: u32,
    full_height: u32,
    tile_offset_x: u32,
    tile_offset_y: u32,
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
            let abs_x = x + tile_offset_x;
            let abs_y = y + tile_offset_y;
            let fx = (abs_x * flare_size) / full_width;
            let fy = (abs_y * flare_size) / full_height;
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

#[cube(launch_unchecked)]
fn cubecl_mask_composite_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    mask_values: &Array<f32>,
    mask_params: &Array<f32>,
    mask_curve_lut: &Array<f32>,
    blur_data: &Array<f32>,
    flare_map: &Array<f32>,
    width: u32,
    pixel_count: u32,
    mask_count: u32,
    is_raw: u32,
) {
    let pixel_index = ABSOLUTE_POS;
    if pixel_index < pixel_count {
        let base = pixel_index * 4;
        let mut comp_r = input[base];
        let mut comp_g = input[base + 1];
        let mut comp_b = input[base + 2];
        let alpha = input[base + 3];

        if is_raw == 0u32 {
            if comp_r <= 0.04045f32 {
                comp_r = comp_r / 12.92f32;
            } else {
                comp_r = ((comp_r + 0.055f32) / 1.055f32).powf(2.4f32);
            }
            if comp_g <= 0.04045f32 {
                comp_g = comp_g / 12.92f32;
            } else {
                comp_g = ((comp_g + 0.055f32) / 1.055f32).powf(2.4f32);
            }
            if comp_b <= 0.04045f32 {
                comp_b = comp_b / 12.92f32;
            } else {
                comp_b = ((comp_b + 0.055f32) / 1.055f32).powf(2.4f32);
            }
        }

        for i in 0u32..9u32 {
            if i < mask_count {
                let influence = mask_values[i * pixel_count + pixel_index];
                if influence > 0.001 {
                    let param_base = i * 8;
                    let exp_adj = mask_params[param_base];
                    let bright_adj = mask_params[param_base + 1];
                    let sharpness = mask_params[param_base + 2];
                    let clarity = mask_params[param_base + 3];
                    let structure = mask_params[param_base + 4];
                    let glow_amount = mask_params[param_base + 5];
                    let halation_amount = mask_params[param_base + 6];
                    let flare_amount = mask_params[param_base + 7];

                    let blur_base = pixel_index * 9;
                    let sharp_r = blur_data[blur_base];
                    let sharp_g = blur_data[blur_base + 1];
                    let sharp_b = blur_data[blur_base + 2];
                    let clar_r = blur_data[blur_base + 3];
                    let clar_g = blur_data[blur_base + 4];
                    let clar_b = blur_data[blur_base + 5];
                    let stru_r = blur_data[blur_base + 6];
                    let stru_g = blur_data[blur_base + 7];
                    let stru_b = blur_data[blur_base + 8];

                    let mut mask_r = comp_r;
                    let mut mask_g = comp_g;
                    let mut mask_b = comp_b;

                    // Local contrast (simplified), mode 0/1 behavior approximated.
                    if sharpness != 0.0f32 {
                        let center_luma =
                            mask_r * 0.2126f32 + mask_g * 0.7152f32 + mask_b * 0.0722f32;
                        let blur_luma =
                            sharp_r * 0.2126f32 + sharp_g * 0.7152f32 + sharp_b * 0.0722f32;
                        let detail = center_luma - blur_luma;
                        let amount = sharpness * 0.8f32;
                        mask_r = mask_r + mask_r * detail * amount;
                        mask_g = mask_g + mask_g * detail * amount;
                        mask_b = mask_b + mask_b * detail * amount;
                    }
                    if clarity != 0.0f32 {
                        let center_luma =
                            mask_r * 0.2126f32 + mask_g * 0.7152f32 + mask_b * 0.0722f32;
                        let blur_luma =
                            clar_r * 0.2126f32 + clar_g * 0.7152f32 + clar_b * 0.0722f32;
                        let detail = center_luma - blur_luma;
                        mask_r = mask_r + mask_r * detail * clarity;
                        mask_g = mask_g + mask_g * detail * clarity;
                        mask_b = mask_b + mask_b * detail * clarity;
                    }
                    if structure != 0.0f32 {
                        let center_luma =
                            mask_r * 0.2126f32 + mask_g * 0.7152f32 + mask_b * 0.0722f32;
                        let blur_luma =
                            stru_r * 0.2126f32 + stru_g * 0.7152f32 + stru_b * 0.0722f32;
                        let detail = center_luma - blur_luma;
                        mask_r = mask_r + mask_r * detail * structure;
                        mask_g = mask_g + mask_g * detail * structure;
                        mask_b = mask_b + mask_b * detail * structure;
                    }

                    // Glow (simplified, structure blur driven).
                    if glow_amount > 0.0 {
                        let exp_scale_glow = Exp::exp(exp_adj * LN_2);
                        let bright_scale_glow = Exp::exp(bright_adj * 0.5 * LN_2);
                        let blr = stru_r * exp_scale_glow * bright_scale_glow;
                        let blg = stru_g * exp_scale_glow * bright_scale_glow;
                        let blb = stru_b * exp_scale_glow * bright_scale_glow;
                        let blur_luma = blr * 0.2126f32 + blg * 0.7152f32 + blb * 0.0722f32;
                        let mut glow_mask = (blur_luma - 0.2f32) / 1.5f32;
                        if glow_mask < 0.0f32 {
                            glow_mask = 0.0f32;
                        }
                        if glow_mask > 1.0f32 {
                            glow_mask = 1.0f32;
                        }
                        mask_r += blr * glow_amount * glow_mask * 2.0f32;
                        mask_g += blg * glow_amount * glow_mask * 2.0f32;
                        mask_b += blb * glow_amount * glow_mask * 2.0f32;
                    }

                    // Halation (simplified, clarity blur driven).
                    if halation_amount > 0.0f32 {
                        let exp_scale_hal = Exp::exp(exp_adj * LN_2);
                        let bright_scale_hal = Exp::exp(bright_adj * 0.5 * LN_2);
                        let blr = clar_r * exp_scale_hal * bright_scale_hal;
                        let blg = clar_g * exp_scale_hal * bright_scale_hal;
                        let blb = clar_b * exp_scale_hal * bright_scale_hal;
                        let blur_luma = blr * 0.2126f32 + blg * 0.7152f32 + blb * 0.0722f32;
                        let mut halo_mask = (blur_luma - 0.4f32) / 1.2f32;
                        if halo_mask < 0.0f32 {
                            halo_mask = 0.0f32;
                        }
                        if halo_mask > 1.0f32 {
                            halo_mask = 1.0f32;
                        }
                        mask_r += blr * halation_amount * halo_mask * 2.0f32;
                        mask_g += blg * halation_amount * halo_mask * 0.8f32;
                        mask_b += blb * halation_amount * halo_mask * 0.5f32;
                    }

                    // Basic mask exposure/brightness stack
                    let exp_scale = Exp::exp(exp_adj * LN_2);
                    let mut adj_r = mask_r * exp_scale;
                    let mut adj_g = mask_g * exp_scale;
                    let mut adj_b = mask_b * exp_scale;
                    if bright_adj != 0.0 {
                        let bright_scale = Exp::exp(bright_adj * 0.5 * LN_2);
                        adj_r *= bright_scale;
                        adj_g *= bright_scale;
                        adj_b *= bright_scale;
                    }

                    comp_r = comp_r + (adj_r - comp_r) * influence;
                    comp_g = comp_g + (adj_g - comp_g) * influence;
                    comp_b = comp_b + (adj_b - comp_b) * influence;

                    // Mask flare (post-adjustment)
                    if flare_amount > 0.0f32 {
                        let x = pixel_index % width;
                        let y = pixel_index / width;
                        let fx = (x * FLARE_MAP_SIZE) / width;
                        let height = pixel_count / width;
                        let fy = (y * FLARE_MAP_SIZE) / height;
                        let flare_base = (fy * FLARE_MAP_SIZE + fx) * 4;
                        let mut fr = flare_map[flare_base] * 1.4f32;
                        let mut fg = flare_map[flare_base + 1] * 1.4f32;
                        let mut fb = flare_map[flare_base + 2] * 1.4f32;
                        fr = fr * fr;
                        fg = fg * fg;
                        fb = fb * fb;
                        let mut safe_r = comp_r;
                        let mut safe_g = comp_g;
                        let mut safe_b = comp_b;
                        if safe_r < 0.0f32 {
                            safe_r = 0.0f32;
                        }
                        if safe_g < 0.0f32 {
                            safe_g = 0.0f32;
                        }
                        if safe_b < 0.0f32 {
                            safe_b = 0.0f32;
                        }
                        let luma = safe_r * 0.2126f32 + safe_g * 0.7152f32 + safe_b * 0.0722f32;
                        let mut t = (luma - 0.7f32) / (1.8f32 - 0.7f32);
                        if t < 0.0 {
                            t = 0.0;
                        }
                        if t > 1.0 {
                            t = 1.0;
                        }
                        let smooth = t * t * (3.0 - 2.0 * t);
                        let protection = 1.0 - smooth;
                        comp_r += fr * flare_amount * protection * influence;
                        comp_g += fg * flare_amount * protection * influence;
                        comp_b += fb * flare_amount * protection * influence;
                    }
                }
            }
        }

        let mut srgb_r = comp_r;
        let mut srgb_g = comp_g;
        let mut srgb_b = comp_b;

        let mut lin_r = comp_r;
        let mut lin_g = comp_g;
        let mut lin_b = comp_b;
        if lin_r < 0.0f32 {
            lin_r = 0.0f32;
        }
        if lin_g < 0.0f32 {
            lin_g = 0.0f32;
        }
        if lin_b < 0.0f32 {
            lin_b = 0.0f32;
        }
        if lin_r > 1.0f32 {
            lin_r = 1.0f32;
        }
        if lin_g > 1.0f32 {
            lin_g = 1.0f32;
        }
        if lin_b > 1.0f32 {
            lin_b = 1.0f32;
        }
        srgb_r = if lin_r <= 0.0031308f32 {
            lin_r * 12.92f32
        } else {
            1.055f32 * lin_r.powf(1.0f32 / 2.4f32) - 0.055f32
        };
        srgb_g = if lin_g <= 0.0031308f32 {
            lin_g * 12.92f32
        } else {
            1.055f32 * lin_g.powf(1.0f32 / 2.4f32) - 0.055f32
        };
        srgb_b = if lin_b <= 0.0031308f32 {
            lin_b * 12.92f32
        } else {
            1.055f32 * lin_b.powf(1.0f32 / 2.4f32) - 0.055f32
        };

        for i in 0u32..9u32 {
            if i < mask_count {
                let influence = mask_values[i * pixel_count + pixel_index];
                if influence > 0.001 {
                    let lut_base = i * 256;
                    let mut cr = srgb_r;
                    let mut cg = srgb_g;
                    let mut cb = srgb_b;
                    if cr < 0.0 {
                        cr = 0.0;
                    }
                    if cg < 0.0 {
                        cg = 0.0;
                    }
                    if cb < 0.0 {
                        cb = 0.0;
                    }
                    if cr > 1.0 {
                        cr = 1.0;
                    }
                    if cg > 1.0 {
                        cg = 1.0;
                    }
                    if cb > 1.0 {
                        cb = 1.0;
                    }
                    let ir = (cr * 255.0) as u32;
                    let ig = (cg * 255.0) as u32;
                    let ib = (cb * 255.0) as u32;
                    let curved_r = mask_curve_lut[lut_base + ir];
                    let curved_g = mask_curve_lut[lut_base + ig];
                    let curved_b = mask_curve_lut[lut_base + ib];
                    srgb_r = srgb_r + (curved_r - srgb_r) * influence;
                    srgb_g = srgb_g + (curved_g - srgb_g) * influence;
                    srgb_b = srgb_b + (curved_b - srgb_b) * influence;
                }
            }
        }

        if srgb_r < 0.0 {
            srgb_r = 0.0;
        }
        if srgb_g < 0.0 {
            srgb_g = 0.0;
        }
        if srgb_b < 0.0 {
            srgb_b = 0.0;
        }
        if srgb_r > 1.0 {
            srgb_r = 1.0;
        }
        if srgb_g > 1.0 {
            srgb_g = 1.0;
        }
        if srgb_b > 1.0 {
            srgb_b = 1.0;
        }
        output[base] = srgb_r;
        output[base + 1] = srgb_g;
        output[base + 2] = srgb_b;
        output[base + 3] = alpha;
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

fn extract_rgba_tile(
    src: &[f32],
    width: u32,
    height: u32,
    x_start: u32,
    y_start: u32,
    tile_width: u32,
    tile_height: u32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; (tile_width * tile_height * 4) as usize];
    for row in 0..tile_height {
        let src_y = y_start + row;
        if src_y >= height {
            break;
        }
        let src_base = ((src_y * width + x_start) * 4) as usize;
        let dst_base = (row * tile_width * 4) as usize;
        let copy_values = (tile_width * 4) as usize;
        out[dst_base..dst_base + copy_values].copy_from_slice(&src[src_base..src_base + copy_values]);
    }
    out
}

fn run_copy_kernel_tiled(
    client: &ComputeClient<cubecl::wgpu::WgpuServer>,
    input: &[f32],
    width: u32,
    height: u32,
) -> Vec<f32> {
    let mut final_output = vec![0.0f32; (width * height * 4) as usize];
    let tiles_x = (width + CUBECL_TILE_SIZE - 1) / CUBECL_TILE_SIZE;
    let tiles_y = (height + CUBECL_TILE_SIZE - 1) / CUBECL_TILE_SIZE;

    for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            let x_start = tile_x * CUBECL_TILE_SIZE;
            let y_start = tile_y * CUBECL_TILE_SIZE;
            let tile_width = std::cmp::min(width - x_start, CUBECL_TILE_SIZE);
            let tile_height = std::cmp::min(height - y_start, CUBECL_TILE_SIZE);

            let input_x_start = std::cmp::max(x_start as i32 - CUBECL_TILE_OVERLAP as i32, 0) as u32;
            let input_y_start = std::cmp::max(y_start as i32 - CUBECL_TILE_OVERLAP as i32, 0) as u32;
            let input_x_end = std::cmp::min(x_start + tile_width + CUBECL_TILE_OVERLAP, width);
            let input_y_end = std::cmp::min(y_start + tile_height + CUBECL_TILE_OVERLAP, height);
            let input_width = input_x_end - input_x_start;
            let input_height = input_y_end - input_y_start;

            let tile_input = extract_rgba_tile(
                input,
                width,
                height,
                input_x_start,
                input_y_start,
                input_width,
                input_height,
            );
            let tile_output = run_copy_kernel(client, &tile_input);

            let crop_x_start = x_start - input_x_start;
            let crop_y_start = y_start - input_y_start;

            for row in 0..tile_height {
                let final_y = y_start + row;
                let final_row_offset = (final_y * width + x_start) as usize * 4;
                let source_y = crop_y_start + row;
                let source_row_offset = (source_y * input_width + crop_x_start) as usize * 4;
                let copy_values = (tile_width * 4) as usize;
                final_output[final_row_offset..final_row_offset + copy_values]
                    .copy_from_slice(&tile_output[source_row_offset..source_row_offset + copy_values]);
            }
        }
    }

    final_output
}

fn run_main_flare_composite_kernel_tiled(
    client: &ComputeClient<cubecl::wgpu::WgpuServer>,
    input: &[f32],
    flare: Option<&[f32]>,
    width: u32,
    height: u32,
    flare_amount: f32,
) -> Vec<f32> {
    let mut final_output = vec![0.0f32; (width * height * 4) as usize];
    let tiles_x = (width + CUBECL_TILE_SIZE - 1) / CUBECL_TILE_SIZE;
    let tiles_y = (height + CUBECL_TILE_SIZE - 1) / CUBECL_TILE_SIZE;

    for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            let x_start = tile_x * CUBECL_TILE_SIZE;
            let y_start = tile_y * CUBECL_TILE_SIZE;
            let tile_width = std::cmp::min(width - x_start, CUBECL_TILE_SIZE);
            let tile_height = std::cmp::min(height - y_start, CUBECL_TILE_SIZE);

            let input_x_start = std::cmp::max(x_start as i32 - CUBECL_TILE_OVERLAP as i32, 0) as u32;
            let input_y_start = std::cmp::max(y_start as i32 - CUBECL_TILE_OVERLAP as i32, 0) as u32;
            let input_x_end = std::cmp::min(x_start + tile_width + CUBECL_TILE_OVERLAP, width);
            let input_y_end = std::cmp::min(y_start + tile_height + CUBECL_TILE_OVERLAP, height);
            let input_width = input_x_end - input_x_start;
            let input_height = input_y_end - input_y_start;

            let tile_input = extract_rgba_tile(
                input,
                width,
                height,
                input_x_start,
                input_y_start,
                input_width,
                input_height,
            );

            let tile_output = run_main_flare_composite_kernel(
                client,
                &tile_input,
                flare,
                input_width,
                input_height,
                width,
                height,
                input_x_start,
                input_y_start,
                flare_amount,
            );

            let crop_x_start = x_start - input_x_start;
            let crop_y_start = y_start - input_y_start;

            for row in 0..tile_height {
                let final_y = y_start + row;
                let final_row_offset = (final_y * width + x_start) as usize * 4;
                let source_y = crop_y_start + row;
                let source_row_offset = (source_y * input_width + crop_x_start) as usize * 4;
                let copy_values = (tile_width * 4) as usize;
                final_output[final_row_offset..final_row_offset + copy_values]
                    .copy_from_slice(&tile_output[source_row_offset..source_row_offset + copy_values]);
            }
        }
    }

    final_output
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
    full_width: u32,
    full_height: u32,
    tile_offset_x: u32,
    tile_offset_y: u32,
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
            ScalarArg::new(full_width),
            ScalarArg::new(full_height),
            ScalarArg::new(tile_offset_x),
            ScalarArg::new(tile_offset_y),
            ScalarArg::new(FLARE_MAP_SIZE),
            ScalarArg::new(flare_amount),
        );
    }

    let bytes = client.read_one(output_handle);
    let out: &[f32] = f32::from_bytes(&bytes);
    out.to_vec()
}

fn run_mask_composite_kernel(
    client: &ComputeClient<cubecl::wgpu::WgpuServer>,
    composite_input: &[f32],
    original_input: &[f32],
    width: u32,
    height: u32,
    all_adjustments: &AllAdjustments,
    mask_bitmaps: &[ImageBuffer<Luma<u8>, Vec<u8>>],
    flare_map: Option<&[f32]>,
) -> (Vec<f32>, MaskCompositeStats) {
    let pixel_count = (width * height) as usize;
    let values = pixel_count * 4;
    let cube_dim_x = 256u32;

    let mut masks = vec![0.0f32; pixel_count * 9];
    for i in 0..9usize {
        if let Some(bitmap) = mask_bitmaps.get(i) {
            if bitmap.dimensions() == (width, height) {
                for (idx, p) in bitmap.pixels().enumerate() {
                    masks[i * pixel_count + idx] = p.0[0] as f32 / 255.0;
                }
            }
        }
    }

    let mut mask_params = vec![0.0f32; 9 * 8];
    let mut mask_curve_lut = vec![0.0f32; 9 * 256];
    let mut influence_sum = 0.0f64;
    let mut influence_count = 0usize;
    let mut max_influence = 0.0f32;
    let mut active_pixels = 0usize;

    for i in 0..std::cmp::min(all_adjustments.mask_count, 9) as usize {
        let m = all_adjustments.mask_adjustments[i];
        let p = i * 8;
        mask_params[p] = m.exposure;
        mask_params[p + 1] = m.brightness;
        mask_params[p + 2] = m.sharpness;
        mask_params[p + 3] = m.clarity;
        mask_params[p + 4] = m.structure;
        mask_params[p + 5] = m.glow_amount;
        mask_params[p + 6] = m.halation_amount;
        mask_params[p + 7] = m.flare_amount;

        let luma_curve = m.luma_curve;
        for x in 0..256usize {
            let in_v = x as f32 / 255.0;
            mask_curve_lut[i * 256 + x] = apply_curve(in_v, &luma_curve, m.luma_curve_count);
        }
    }

    for px in 0..pixel_count {
        let mut any_active = false;
        for i in 0..std::cmp::min(all_adjustments.mask_count, 9) as usize {
            let v = masks[i * pixel_count + px];
            influence_sum += v as f64;
            influence_count += 1;
            max_influence = max_influence.max(v);
            if v > 0.001 {
                any_active = true;
            }
        }
        if any_active {
            active_pixels += 1;
        }
    }

    let mut blur_data = vec![0.0f32; pixel_count * 9];
    let need_blurs = (0..std::cmp::min(all_adjustments.mask_count, 9) as usize).any(|i| {
        let m = all_adjustments.mask_adjustments[i];
        !near_zero(m.sharpness)
            || !near_zero(m.clarity)
            || !near_zero(m.structure)
            || !near_zero(m.glow_amount)
            || !near_zero(m.halation_amount)
    });
    if need_blurs {
        let scale = f32::max(std::cmp::min(width, height) as f32 / 1080.0_f32, 0.1_f32);
        let sharpness_radius = f32::max((2.0_f32 * scale).ceil(), 1.0_f32) as u32;
        let clarity_radius = f32::max((8.0_f32 * scale).ceil(), 1.0_f32) as u32;
        let structure_radius = f32::max((40.0_f32 * scale).ceil(), 1.0_f32) as u32;
        let sharp =
            run_gaussian_blur_kernel(client, original_input, width, height, sharpness_radius);
        let clar = run_gaussian_blur_kernel(client, original_input, width, height, clarity_radius);
        let stru =
            run_gaussian_blur_kernel(client, original_input, width, height, structure_radius);
        for px in 0..pixel_count {
            let src = px * 4;
            let dst = px * 9;
            blur_data[dst] = sharp[src];
            blur_data[dst + 1] = sharp[src + 1];
            blur_data[dst + 2] = sharp[src + 2];
            blur_data[dst + 3] = clar[src];
            blur_data[dst + 4] = clar[src + 1];
            blur_data[dst + 5] = clar[src + 2];
            blur_data[dst + 6] = stru[src];
            blur_data[dst + 7] = stru[src + 1];
            blur_data[dst + 8] = stru[src + 2];
        }
    }

    let flare_data: Vec<f32> = if let Some(f) = flare_map {
        f.to_vec()
    } else {
        vec![0.0f32; (FLARE_MAP_SIZE * FLARE_MAP_SIZE * 4) as usize]
    };

    let input_handle = client.create(f32::as_bytes(composite_input));
    let output_handle = client.empty(values * std::mem::size_of::<f32>());
    let mask_values_handle = client.create(f32::as_bytes(&masks));
    let params_handle = client.create(f32::as_bytes(&mask_params));
    let curve_lut_handle = client.create(f32::as_bytes(&mask_curve_lut));
    let blur_handle = client.create(f32::as_bytes(&blur_data));
    let flare_handle = client.create(f32::as_bytes(&flare_data));

    unsafe {
        cubecl_mask_composite_kernel::launch_unchecked::<cubecl::wgpu::WgpuRuntime>(
            client,
            CubeCount::Static(dispatch_count_1d(pixel_count, cube_dim_x), 1, 1),
            CubeDim::new_1d(cube_dim_x),
            ArrayArg::from_raw_parts::<f32>(&input_handle, values, 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, values, 1),
            ArrayArg::from_raw_parts::<f32>(&mask_values_handle, masks.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&params_handle, mask_params.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&curve_lut_handle, mask_curve_lut.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&blur_handle, blur_data.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&flare_handle, flare_data.len(), 1),
            ScalarArg::new(width),
            ScalarArg::new(width * height),
            ScalarArg::new(std::cmp::min(all_adjustments.mask_count, 9)),
            ScalarArg::new(all_adjustments.global.is_raw_image),
        );
    }

    let bytes = client.read_one(output_handle);
    let out: &[f32] = f32::from_bytes(&bytes);
    let mean_influence = if influence_count == 0 {
        0.0
    } else {
        (influence_sum / influence_count as f64) as f32
    };
    (
        out.to_vec(),
        MaskCompositeStats {
            mask_count: std::cmp::min(all_adjustments.mask_count, 9),
            active_pixels,
            max_influence,
            mean_influence,
        },
    )
}

#[inline]
fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge0 == edge1 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn fract(v: f32) -> f32 {
    v - v.floor()
}

fn hash2(p: [f32; 2]) -> f32 {
    let mut p3 = [
        fract(p[0] * 0.1031),
        fract(p[1] * 0.1031),
        fract(p[0] * 0.1031),
    ];
    let dot_term = p3[0] * (p3[1] + 33.33) + p3[1] * (p3[2] + 33.33) + p3[2] * (p3[0] + 33.33);
    p3[0] += dot_term;
    p3[1] += dot_term;
    p3[2] += dot_term;
    fract((p3[0] + p3[1]) * p3[2])
}

fn gradient_noise(p: [f32; 2]) -> f32 {
    let i = [p[0].floor(), p[1].floor()];
    let f = [fract(p[0]), fract(p[1])];
    let u = [
        f[0] * f[0] * f[0] * (f[0] * (f[0] * 6.0 - 15.0) + 10.0),
        f[1] * f[1] * f[1] * (f[1] * (f[1] * 6.0 - 15.0) + 10.0),
    ];

    let ga = [
        hash2([i[0], i[1]]) * 2.0 - 1.0,
        hash2([i[0] + 11.0, i[1] + 37.0]) * 2.0 - 1.0,
    ];
    let gb = [
        hash2([i[0] + 1.0, i[1]]) * 2.0 - 1.0,
        hash2([i[0] + 12.0, i[1] + 37.0]) * 2.0 - 1.0,
    ];
    let gc = [
        hash2([i[0], i[1] + 1.0]) * 2.0 - 1.0,
        hash2([i[0] + 11.0, i[1] + 38.0]) * 2.0 - 1.0,
    ];
    let gd = [
        hash2([i[0] + 1.0, i[1] + 1.0]) * 2.0 - 1.0,
        hash2([i[0] + 12.0, i[1] + 38.0]) * 2.0 - 1.0,
    ];

    let dot_00 = ga[0] * f[0] + ga[1] * f[1];
    let dot_10 = gb[0] * (f[0] - 1.0) + gb[1] * f[1];
    let dot_01 = gc[0] * f[0] + gc[1] * (f[1] - 1.0);
    let dot_11 = gd[0] * (f[0] - 1.0) + gd[1] * (f[1] - 1.0);

    let bottom = mix(dot_00, dot_10, u[0]);
    let top = mix(dot_01, dot_11, u[0]);
    mix(bottom, top, u[1])
}

fn dither_noise(x: u32, y: u32) -> f32 {
    let p = [x as f32, y as f32];
    fract(((p[0] * 12.9898 + p[1] * 78.233).sin()) * 43758.5453) - 0.5
}

fn apply_ca_correction(
    input_rgba_srgb: &[f32],
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    ca_rc: f32,
    ca_by: f32,
) -> [f32; 3] {
    if near_zero(ca_rc) && near_zero(ca_by) {
        let base = ((y * width + x) * 4) as usize;
        return [
            input_rgba_srgb[base],
            input_rgba_srgb[base + 1],
            input_rgba_srgb[base + 2],
        ];
    }

    let dims = [width as f32, height as f32];
    let center = [dims[0] * 0.5, dims[1] * 0.5];
    let current = [x as f32, y as f32];
    let to_center = [current[0] - center[0], current[1] - center[1]];
    let dist = (to_center[0] * to_center[0] + to_center[1] * to_center[1]).sqrt();
    if dist == 0.0 {
        let base = ((y * width + x) * 4) as usize;
        return [
            input_rgba_srgb[base],
            input_rgba_srgb[base + 1],
            input_rgba_srgb[base + 2],
        ];
    }

    let dir = [to_center[0] / dist, to_center[1] / dist];
    let red_shift = [dir[0] * dist * ca_rc, dir[1] * dist * ca_rc];
    let blue_shift = [dir[0] * dist * ca_by, dir[1] * dist * ca_by];

    let red_coords = [
        (current[0] - red_shift[0]).round(),
        (current[1] - red_shift[1]).round(),
    ];
    let blue_coords = [
        (current[0] - blue_shift[0]).round(),
        (current[1] - blue_shift[1]).round(),
    ];
    let max_x = width.saturating_sub(1) as f32;
    let max_y = height.saturating_sub(1) as f32;
    let rx = red_coords[0].clamp(0.0, max_x) as u32;
    let ry = red_coords[1].clamp(0.0, max_y) as u32;
    let gx = x;
    let gy = y;
    let bx = blue_coords[0].clamp(0.0, max_x) as u32;
    let by = blue_coords[1].clamp(0.0, max_y) as u32;

    let r_base = ((ry * width + rx) * 4) as usize;
    let g_base = ((gy * width + gx) * 4) as usize;
    let b_base = ((by * width + bx) * 4) as usize;
    [
        input_rgba_srgb[r_base],
        input_rgba_srgb[g_base + 1],
        input_rgba_srgb[b_base + 2],
    ]
}

#[inline]
fn get_luma(c: [f32; 3]) -> f32 {
    c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722
}

#[inline]
fn srgb_to_linear_channel(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb_channel(v: f32) -> f32 {
    let clamped = v.clamp(0.0, 1.0);
    if clamped <= 0.003_130_8 {
        clamped * 12.92
    } else {
        1.055 * clamped.powf(1.0 / 2.4) - 0.055
    }
}

#[inline]
fn srgb_to_linear(c: [f32; 3]) -> [f32; 3] {
    [
        srgb_to_linear_channel(c[0]),
        srgb_to_linear_channel(c[1]),
        srgb_to_linear_channel(c[2]),
    ]
}

#[inline]
fn linear_to_srgb(c: [f32; 3]) -> [f32; 3] {
    [
        linear_to_srgb_channel(c[0]),
        linear_to_srgb_channel(c[1]),
        linear_to_srgb_channel(c[2]),
    ]
}

#[inline]
fn agx_sigmoid(x: f32, power: f32) -> f32 {
    x / (1.0 + x.powf(power)).powf(1.0 / power)
}

#[inline]
fn agx_scaled_sigmoid(
    x: f32,
    scale: f32,
    slope: f32,
    power: f32,
    transition_x: f32,
    transition_y: f32,
) -> f32 {
    scale * agx_sigmoid(slope * (x - transition_x) / scale, power) + transition_y
}

#[inline]
fn agx_apply_curve_channel(x: f32) -> f32 {
    let result = if x < AGX_TOE_TRANSITION_X {
        agx_scaled_sigmoid(
            x,
            AGX_TOE_SCALE,
            AGX_SLOPE,
            AGX_TOE_POWER,
            AGX_TOE_TRANSITION_X,
            AGX_TOE_TRANSITION_Y,
        )
    } else if x <= AGX_SHOULDER_TRANSITION_X {
        AGX_SLOPE * x + AGX_INTERCEPT
    } else {
        agx_scaled_sigmoid(
            x,
            AGX_SHOULDER_SCALE,
            AGX_SLOPE,
            AGX_SHOULDER_POWER,
            AGX_SHOULDER_TRANSITION_X,
            AGX_SHOULDER_TRANSITION_Y,
        )
    };
    result.clamp(AGX_TARGET_BLACK_PRE_GAMMA, AGX_TARGET_WHITE_PRE_GAMMA)
}

#[inline]
fn agx_compress_gamut(c: [f32; 3]) -> [f32; 3] {
    let min_c = c[0].min(c[1].min(c[2]));
    if min_c < 0.0 {
        [c[0] - min_c, c[1] - min_c, c[2] - min_c]
    } else {
        c
    }
}

#[inline]
fn mat3_mul_vec3(columns: &[f32; 12], v: [f32; 3]) -> [f32; 3] {
    [
        columns[0] * v[0] + columns[4] * v[1] + columns[8] * v[2],
        columns[1] * v[0] + columns[5] * v[1] + columns[9] * v[2],
        columns[2] * v[0] + columns[6] * v[1] + columns[10] * v[2],
    ]
}

#[inline]
fn agx_tonemap(c: [f32; 3]) -> [f32; 3] {
    let x_relative = [
        (c[0] / 0.18).max(AGX_EPSILON),
        (c[1] / 0.18).max(AGX_EPSILON),
        (c[2] / 0.18).max(AGX_EPSILON),
    ];
    let mapped = [
        ((x_relative[0].log2() - AGX_MIN_EV) / AGX_RANGE_EV).clamp(0.0, 1.0),
        ((x_relative[1].log2() - AGX_MIN_EV) / AGX_RANGE_EV).clamp(0.0, 1.0),
        ((x_relative[2].log2() - AGX_MIN_EV) / AGX_RANGE_EV).clamp(0.0, 1.0),
    ];
    let curved = [
        agx_apply_curve_channel(mapped[0]),
        agx_apply_curve_channel(mapped[1]),
        agx_apply_curve_channel(mapped[2]),
    ];
    [
        curved[0].max(0.0).powf(AGX_GAMMA),
        curved[1].max(0.0).powf(AGX_GAMMA),
        curved[2].max(0.0).powf(AGX_GAMMA),
    ]
}

#[inline]
fn agx_full_transform(color_in: [f32; 3], all_adjustments: &AllAdjustments) -> [f32; 3] {
    let compressed = agx_compress_gamut(color_in);
    let pipe_to_rendering: &[f32; 12] =
        bytemuck::cast_ref(&all_adjustments.global.agx_pipe_to_rendering_matrix);
    let rendering_to_pipe: &[f32; 12] =
        bytemuck::cast_ref(&all_adjustments.global.agx_rendering_to_pipe_matrix);
    let in_agx_space = mat3_mul_vec3(pipe_to_rendering, compressed);
    let tonemapped_agx = agx_tonemap(in_agx_space);
    mat3_mul_vec3(rendering_to_pipe, tonemapped_agx)
}

#[inline]
fn point_xy(p: &Point) -> (f32, f32) {
    let raw: &[f32; 4] = bytemuck::cast_ref(p);
    (raw[0], raw[1])
}

#[inline]
fn hsl_fields(h: &HslColor) -> (f32, f32, f32) {
    let raw: &[f32; 4] = bytemuck::cast_ref(h);
    (raw[0], raw[1], raw[2])
}

#[inline]
fn rgb_to_hsv(c: [f32; 3]) -> [f32; 3] {
    let c_max = c[0].max(c[1].max(c[2]));
    let c_min = c[0].min(c[1].min(c[2]));
    let delta = c_max - c_min;
    let mut h = 0.0;
    if delta > 0.0 {
        if c_max == c[0] {
            h = 60.0 * (((c[1] - c[2]) / delta) % 6.0);
        } else if c_max == c[1] {
            h = 60.0 * (((c[2] - c[0]) / delta) + 2.0);
        } else {
            h = 60.0 * (((c[0] - c[1]) / delta) + 4.0);
        }
    }
    if h < 0.0 {
        h += 360.0;
    }
    let s = if c_max > 0.0 { delta / c_max } else { 0.0 };
    [h, s, c_max]
}

#[inline]
fn hsv_to_rgb(c: [f32; 3]) -> [f32; 3] {
    let h = c[0];
    let s = c[1];
    let v = c[2];
    let chroma = v * s;
    let x = chroma * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - chroma;
    let rgb_prime = if h < 60.0 {
        [chroma, x, 0.0]
    } else if h < 120.0 {
        [x, chroma, 0.0]
    } else if h < 180.0 {
        [0.0, chroma, x]
    } else if h < 240.0 {
        [0.0, x, chroma]
    } else if h < 300.0 {
        [x, 0.0, chroma]
    } else {
        [chroma, 0.0, x]
    };
    [rgb_prime[0] + m, rgb_prime[1] + m, rgb_prime[2] + m]
}

fn interpolate_cubic_hermite(x: f32, p1: Point, p2: Point, m1: f32, m2: f32) -> f32 {
    let (p1x, p1y) = point_xy(&p1);
    let (p2x, p2y) = point_xy(&p2);
    let dx = p2x - p1x;
    if dx <= 0.0 {
        return p1y;
    }
    let t = (x - p1x) / dx;
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * p1y + h10 * m1 * dx + h01 * p2y + h11 * m2 * dx
}

fn apply_curve(val: f32, points: &[Point; 16], count: u32) -> f32 {
    if count < 2 {
        return val;
    }
    let x = val * 255.0;
    let (p0x, p0y) = point_xy(&points[0]);
    if x <= p0x {
        return p0y / 255.0;
    }
    let (plastx, plasty) = point_xy(&points[count as usize - 1]);
    if x >= plastx {
        return plasty / 255.0;
    }
    for i in 0..15usize {
        if i >= count as usize - 1 {
            break;
        }
        let p1 = points[i];
        let p2 = points[i + 1];
        let (p2x, p2y) = point_xy(&p2);
        if x <= p2x {
            let p0 = points[i.saturating_sub(1)];
            let p3 = points[(i + 2).min(count as usize - 1)];
            let (p0x, p0y) = point_xy(&p0);
            let (p1x, p1y) = point_xy(&p1);
            let (p3x, p3y) = point_xy(&p3);
            let delta_before = (p1y - p0y) / (p1x - p0x).max(0.001);
            let delta_current = (p2y - p1y) / (p2x - p1x).max(0.001);
            let delta_after = (p3y - p2y) / (p3x - p2x).max(0.001);
            let mut tangent_at_p1 = if i == 0 {
                delta_current
            } else if delta_before * delta_current <= 0.0 {
                0.0
            } else {
                (delta_before + delta_current) / 2.0
            };
            let mut tangent_at_p2 = if i + 1 == count as usize - 1 {
                delta_current
            } else if delta_current * delta_after <= 0.0 {
                0.0
            } else {
                (delta_current + delta_after) / 2.0
            };
            if delta_current != 0.0 {
                let alpha = tangent_at_p1 / delta_current;
                let beta = tangent_at_p2 / delta_current;
                if alpha * alpha + beta * beta > 9.0 {
                    let tau = 3.0 / (alpha * alpha + beta * beta).sqrt();
                    tangent_at_p1 *= tau;
                    tangent_at_p2 *= tau;
                }
            }
            let result_y = interpolate_cubic_hermite(x, p1, p2, tangent_at_p1, tangent_at_p2);
            return (result_y / 255.0).clamp(0.0, 1.0);
        }
    }
    plasty / 255.0
}

fn is_default_curve(points: &[Point; 16], count: u32) -> bool {
    if count != 2 {
        return false;
    }
    let (p0x, p0y) = point_xy(&points[0]);
    let (p1x, p1y) = point_xy(&points[1]);
    (p0x - 0.0).abs() < 0.1
        && (p0y - 0.0).abs() < 0.1
        && (p1x - 255.0).abs() < 0.1
        && (p1y - 255.0).abs() < 0.1
}

fn apply_all_curves(
    color: [f32; 3],
    luma_curve: &[Point; 16],
    luma_curve_count: u32,
    red_curve: &[Point; 16],
    red_curve_count: u32,
    green_curve: &[Point; 16],
    green_curve_count: u32,
    blue_curve: &[Point; 16],
    blue_curve_count: u32,
) -> [f32; 3] {
    let rgb_curves_active = !is_default_curve(red_curve, red_curve_count)
        || !is_default_curve(green_curve, green_curve_count)
        || !is_default_curve(blue_curve, blue_curve_count);
    if rgb_curves_active {
        let color_graded = [
            apply_curve(color[0], red_curve, red_curve_count),
            apply_curve(color[1], green_curve, green_curve_count),
            apply_curve(color[2], blue_curve, blue_curve_count),
        ];
        let luma_initial = get_luma(color);
        let luma_target = apply_curve(luma_initial, luma_curve, luma_curve_count);
        let luma_graded = get_luma(color_graded);
        let mut final_color = if luma_graded > 0.001 {
            let s = luma_target / luma_graded;
            [
                color_graded[0] * s,
                color_graded[1] * s,
                color_graded[2] * s,
            ]
        } else {
            [luma_target; 3]
        };
        let max_comp = final_color[0].max(final_color[1].max(final_color[2]));
        if max_comp > 1.0 {
            final_color = [
                final_color[0] / max_comp,
                final_color[1] / max_comp,
                final_color[2] / max_comp,
            ];
        }
        final_color
    } else {
        [
            apply_curve(color[0], luma_curve, luma_curve_count),
            apply_curve(color[1], luma_curve, luma_curve_count),
            apply_curve(color[2], luma_curve, luma_curve_count),
        ]
    }
}

fn get_raw_hsl_influence(hue: f32, center: f32, width: f32) -> f32 {
    let dist = (hue - center).abs().min(360.0 - (hue - center).abs());
    let sharpness = 1.5;
    let falloff = dist / (width * 0.5);
    (-sharpness * falloff * falloff).exp()
}

fn apply_hsl_panel(color: [f32; 3], hsl_adjustments: &[HslColor; 8]) -> [f32; 3] {
    if (color[0] - color[1]).abs() < 0.001 && (color[1] - color[2]).abs() < 0.001 {
        return color;
    }
    let ranges = [
        (358.0, 35.0),
        (25.0, 45.0),
        (60.0, 40.0),
        (115.0, 90.0),
        (180.0, 60.0),
        (225.0, 60.0),
        (280.0, 55.0),
        (330.0, 50.0),
    ];
    let original_hsv = rgb_to_hsv(color);
    let original_luma = get_luma(color);
    let saturation_mask = smoothstep(0.05, 0.20, original_hsv[1]);
    let luminance_weight = smoothstep(0.0, 1.0, original_hsv[1]);
    if saturation_mask < 0.001 && luminance_weight < 0.001 {
        return color;
    }
    let mut raw = [0.0f32; 8];
    let mut total = 0.0;
    for i in 0..8usize {
        let infl = get_raw_hsl_influence(original_hsv[0], ranges[i].0, ranges[i].1);
        raw[i] = infl;
        total += infl;
    }
    if total <= 0.0 {
        return color;
    }
    let mut total_hue_shift = 0.0;
    let mut total_sat_multiplier = 0.0;
    let mut total_lum_adjust = 0.0;
    for i in 0..8usize {
        let norm = raw[i] / total;
        let hue_sat_influence = norm * saturation_mask;
        let luma_influence = norm * luminance_weight;
        let (h, s, l) = hsl_fields(&hsl_adjustments[i]);
        total_hue_shift += h * 2.0 * hue_sat_influence;
        total_sat_multiplier += s * hue_sat_influence;
        total_lum_adjust += l * luma_influence;
    }
    if original_hsv[1] * (1.0 + total_sat_multiplier) < 0.0001 {
        let final_luma = original_luma * (1.0 + total_lum_adjust);
        return [final_luma; 3];
    }
    let mut hsv = original_hsv;
    hsv[0] = (hsv[0] + total_hue_shift + 360.0) % 360.0;
    hsv[1] = (hsv[1] * (1.0 + total_sat_multiplier)).clamp(0.0, 1.0);
    let hs_shifted = hsv_to_rgb([hsv[0], hsv[1], original_hsv[2]]);
    let new_luma = get_luma(hs_shifted);
    let target_luma = original_luma * (1.0 + total_lum_adjust);
    if new_luma < 0.0001 {
        return [target_luma.max(0.0); 3];
    }
    let s = target_luma / new_luma;
    [hs_shifted[0] * s, hs_shifted[1] * s, hs_shifted[2] * s]
}

fn apply_linear_exposure(color: [f32; 3], exposure_adj: f32) -> [f32; 3] {
    if exposure_adj == 0.0 {
        return color;
    }
    let s = 2.0f32.powf(exposure_adj);
    [color[0] * s, color[1] * s, color[2] * s]
}

fn apply_filmic_exposure(color: [f32; 3], brightness_adj: f32) -> [f32; 3] {
    if brightness_adj == 0.0 {
        return color;
    }
    const RATIONAL_CURVE_MIX: f32 = 0.95;
    const MIDTONE_STRENGTH: f32 = 1.2;
    let original_luma = get_luma(color);
    if original_luma.abs() < 0.00001 {
        return color;
    }
    let direct_adj = brightness_adj * (1.0 - RATIONAL_CURVE_MIX);
    let rational_adj = brightness_adj * RATIONAL_CURVE_MIX;
    let scale = 2.0f32.powf(direct_adj);
    let k = 2.0f32.powf(-rational_adj * MIDTONE_STRENGTH);
    let luma_abs = original_luma.abs();
    let luma_floor = luma_abs.floor();
    let luma_fract = luma_abs - luma_floor;
    let shaped_fract = luma_fract / (luma_fract + (1.0 - luma_fract) * k);
    let shaped_luma_abs = luma_floor + shaped_fract;
    let new_luma = original_luma.signum() * shaped_luma_abs * scale;
    let chroma = [
        color[0] - original_luma,
        color[1] - original_luma,
        color[2] - original_luma,
    ];
    let total_luma_scale = new_luma / original_luma;
    let chroma_scale = total_luma_scale.powf(0.8);
    [
        new_luma + chroma[0] * chroma_scale,
        new_luma + chroma[1] * chroma_scale,
        new_luma + chroma[2] * chroma_scale,
    ]
}

fn apply_tonal_adjustments(color: [f32; 3], con: f32, sh: f32, wh: f32, bl: f32) -> [f32; 3] {
    let mut rgb = color;
    if wh != 0.0 {
        let white_level = 1.0 - wh * 0.25;
        rgb = [
            rgb[0] / white_level.max(0.01),
            rgb[1] / white_level.max(0.01),
            rgb[2] / white_level.max(0.01),
        ];
    }
    if bl != 0.0 {
        let luma_for_blacks = get_luma([rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)]);
        let mask = 1.0 - smoothstep(0.0, 0.25, luma_for_blacks);
        if mask > 0.001 {
            let factor = 2.0f32.powf(bl * 0.75);
            let adjusted = [rgb[0] * factor, rgb[1] * factor, rgb[2] * factor];
            rgb = [
                mix(rgb[0], adjusted[0], mask),
                mix(rgb[1], adjusted[1], mask),
                mix(rgb[2], adjusted[2], mask),
            ];
        }
    }
    let luma = get_luma([rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)]);
    if sh != 0.0 {
        let mask = (1.0 - smoothstep(0.0, 0.4, luma)).powi(3);
        if mask > 0.001 {
            let factor = 2.0f32.powf(sh * 1.5);
            let adjusted = [rgb[0] * factor, rgb[1] * factor, rgb[2] * factor];
            rgb = [
                mix(rgb[0], adjusted[0], mask),
                mix(rgb[1], adjusted[1], mask),
                mix(rgb[2], adjusted[2], mask),
            ];
        }
    }
    if con != 0.0 {
        let safe_rgb = [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)];
        let g = 2.2;
        let perceptual = [
            safe_rgb[0].powf(1.0 / g),
            safe_rgb[1].powf(1.0 / g),
            safe_rgb[2].powf(1.0 / g),
        ];
        let clamped = [
            perceptual[0].clamp(0.0, 1.0),
            perceptual[1].clamp(0.0, 1.0),
            perceptual[2].clamp(0.0, 1.0),
        ];
        let strength = 2.0f32.powf(con * 1.25);
        let mut curved = [0.0; 3];
        for i in 0..3 {
            curved[i] = if clamped[i] < 0.5 {
                0.5 * (2.0 * clamped[i]).powf(strength)
            } else {
                1.0 - 0.5 * (2.0 * (1.0 - clamped[i])).powf(strength)
            };
        }
        let contrast_adjusted = [curved[0].powf(g), curved[1].powf(g), curved[2].powf(g)];
        let mix_factor = [
            smoothstep(1.0, 1.01, safe_rgb[0]),
            smoothstep(1.0, 1.01, safe_rgb[1]),
            smoothstep(1.0, 1.01, safe_rgb[2]),
        ];
        rgb = [
            mix(contrast_adjusted[0], rgb[0], mix_factor[0]),
            mix(contrast_adjusted[1], rgb[1], mix_factor[1]),
            mix(contrast_adjusted[2], rgb[2], mix_factor[2]),
        ];
    }
    rgb
}

fn apply_highlights_adjustment(color: [f32; 3], highlights_adj: f32) -> [f32; 3] {
    if highlights_adj == 0.0 {
        return color;
    }
    let luma = get_luma([color[0].max(0.0), color[1].max(0.0), color[2].max(0.0)]);
    let highlight_mask = smoothstep(0.3, 0.95, (luma * 1.5).tanh());
    if highlight_mask < 0.001 {
        return color;
    }
    let final_adjusted = if highlights_adj < 0.0 {
        let new_luma = if luma <= 1.0 {
            luma.powf(1.0 - highlights_adj * 1.75)
        } else {
            let luma_excess = luma - 1.0;
            let compressed = luma_excess / (1.0 + luma_excess * (-highlights_adj * 6.0));
            1.0 + compressed
        };
        let tonally_adjusted = [
            color[0] * (new_luma / luma.max(0.0001)),
            color[1] * (new_luma / luma.max(0.0001)),
            color[2] * (new_luma / luma.max(0.0001)),
        ];
        let desat_amount = smoothstep(1.0, 10.0, luma);
        let white_point = [new_luma; 3];
        [
            mix(tonally_adjusted[0], white_point[0], desat_amount),
            mix(tonally_adjusted[1], white_point[1], desat_amount),
            mix(tonally_adjusted[2], white_point[2], desat_amount),
        ]
    } else {
        let factor = 2.0f32.powf(highlights_adj * 1.75);
        [color[0] * factor, color[1] * factor, color[2] * factor]
    };
    [
        mix(color[0], final_adjusted[0], highlight_mask),
        mix(color[1], final_adjusted[1], highlight_mask),
        mix(color[2], final_adjusted[2], highlight_mask),
    ]
}

fn apply_white_balance(color: [f32; 3], temp: f32, tnt: f32) -> [f32; 3] {
    let temp_mult = [1.0 + temp * 0.2, 1.0 + temp * 0.05, 1.0 - temp * 0.2];
    let tint_mult = [1.0 + tnt * 0.25, 1.0 - tnt * 0.25, 1.0 + tnt * 0.25];
    [
        color[0] * temp_mult[0] * tint_mult[0],
        color[1] * temp_mult[1] * tint_mult[1],
        color[2] * temp_mult[2] * tint_mult[2],
    ]
}

fn apply_creative_color(color: [f32; 3], sat: f32, vib: f32) -> [f32; 3] {
    let mut processed = color;
    let luma = get_luma(processed);
    if sat != 0.0 {
        processed = [
            mix(luma, processed[0], 1.0 + sat),
            mix(luma, processed[1], 1.0 + sat),
            mix(luma, processed[2], 1.0 + sat),
        ];
    }
    if vib == 0.0 {
        return processed;
    }
    let c_max = processed[0].max(processed[1].max(processed[2]));
    let c_min = processed[0].min(processed[1].min(processed[2]));
    let delta = c_max - c_min;
    if delta < 0.02 {
        return processed;
    }
    let current_sat = delta / c_max.max(0.001);
    if vib > 0.0 {
        let sat_mask = 1.0 - smoothstep(0.4, 0.9, current_sat);
        let hsv = rgb_to_hsv(processed);
        let hue_dist = (hsv[0] - 25.0).abs().min(360.0 - (hsv[0] - 25.0).abs());
        let is_skin = smoothstep(35.0, 10.0, hue_dist);
        let skin_dampener = mix(1.0, 0.6, is_skin);
        let amount = vib * sat_mask * skin_dampener * 3.0;
        processed = [
            mix(luma, processed[0], 1.0 + amount),
            mix(luma, processed[1], 1.0 + amount),
            mix(luma, processed[2], 1.0 + amount),
        ];
    } else {
        let desat_mask = 1.0 - smoothstep(0.2, 0.8, current_sat);
        let amount = vib * desat_mask;
        processed = [
            mix(luma, processed[0], 1.0 + amount),
            mix(luma, processed[1], 1.0 + amount),
            mix(luma, processed[2], 1.0 + amount),
        ];
    }
    processed
}

fn apply_dehaze(color: [f32; 3], amount: f32) -> [f32; 3] {
    if amount == 0.0 {
        return color;
    }
    let atmospheric_light = [0.95, 0.97, 1.0];
    if amount > 0.0 {
        let dark_channel = color[0].min(color[1].min(color[2]));
        let transmission_estimate = 1.0 - dark_channel;
        let t = 1.0 - amount * transmission_estimate;
        let recovered = [
            (color[0] - atmospheric_light[0]) / t.max(0.1) + atmospheric_light[0],
            (color[1] - atmospheric_light[1]) / t.max(0.1) + atmospheric_light[1],
            (color[2] - atmospheric_light[2]) / t.max(0.1) + atmospheric_light[2],
        ];
        let mut result = [
            mix(color[0], recovered[0], amount),
            mix(color[1], recovered[1], amount),
            mix(color[2], recovered[2], amount),
        ];
        result = [
            0.5 + (result[0] - 0.5) * (1.0 + amount * 0.15),
            0.5 + (result[1] - 0.5) * (1.0 + amount * 0.15),
            0.5 + (result[2] - 0.5) * (1.0 + amount * 0.15),
        ];
        let luma = get_luma(result);
        [
            mix(luma, result[0], 1.0 + amount * 0.1),
            mix(luma, result[1], 1.0 + amount * 0.1),
            mix(luma, result[2], 1.0 + amount * 0.1),
        ]
    } else {
        [
            mix(color[0], atmospheric_light[0], amount.abs() * 0.7),
            mix(color[1], atmospheric_light[1], amount.abs() * 0.7),
            mix(color[2], atmospheric_light[2], amount.abs() * 0.7),
        ]
    }
}

fn apply_color_calibration(color: [f32; 3], cal: ColorCalibrationSettings) -> [f32; 3] {
    let h_r = cal.red_hue;
    let h_g = cal.green_hue;
    let h_b = cal.blue_hue;
    let r_prime = [1.0 - h_r.abs(), h_r.max(0.0), (-h_r).max(0.0)];
    let g_prime = [(-h_g).max(0.0), 1.0 - h_g.abs(), h_g.max(0.0)];
    let b_prime = [h_b.max(0.0), (-h_b).max(0.0), 1.0 - h_b.abs()];

    let mut c = [
        r_prime[0] * color[0] + g_prime[0] * color[1] + b_prime[0] * color[2],
        r_prime[1] * color[0] + g_prime[1] * color[1] + b_prime[1] * color[2],
        r_prime[2] * color[0] + g_prime[2] * color[1] + b_prime[2] * color[2],
    ];

    let luma = get_luma([c[0].max(0.0), c[1].max(0.0), c[2].max(0.0)]);
    let desaturated = [luma; 3];
    let sat_vector = [
        c[0] - desaturated[0],
        c[1] - desaturated[1],
        c[2] - desaturated[2],
    ];

    let color_sum = c[0] + c[1] + c[2];
    let masks = if color_sum > 0.001 {
        [c[0] / color_sum, c[1] / color_sum, c[2] / color_sum]
    } else {
        [0.0, 0.0, 0.0]
    };
    let total_sat_adjustment = masks[0] * cal.red_saturation
        + masks[1] * cal.green_saturation
        + masks[2] * cal.blue_saturation;
    c = [
        c[0] + sat_vector[0] * total_sat_adjustment,
        c[1] + sat_vector[1] * total_sat_adjustment,
        c[2] + sat_vector[2] * total_sat_adjustment,
    ];

    let st = cal.shadows_tint;
    if st.abs() > 0.001 {
        let shadow_luma = get_luma([c[0].max(0.0), c[1].max(0.0), c[2].max(0.0)]);
        let mask = 1.0 - smoothstep(0.0, 0.3, shadow_luma);
        let tint_mult = [1.0 + st * 0.25, 1.0 - st * 0.25, 1.0 + st * 0.25];
        c = [
            mix(c[0], c[0] * tint_mult[0], mask),
            mix(c[1], c[1] * tint_mult[1], mask),
            mix(c[2], c[2] * tint_mult[2], mask),
        ];
    }

    c
}

fn apply_color_grading(
    color: [f32; 3],
    shadows: ColorGradeSettings,
    midtones: ColorGradeSettings,
    highlights: ColorGradeSettings,
    blending: f32,
    balance: f32,
) -> [f32; 3] {
    let luma = get_luma([color[0].max(0.0), color[1].max(0.0), color[2].max(0.0)]);
    let shadow_crossover = 0.1 + 0.5 * (-balance).max(0.0);
    let highlight_crossover = 0.5 - 0.5 * balance.max(0.0);
    let feather = 0.2 * blending;
    let final_shadow_crossover = shadow_crossover.min(highlight_crossover - 0.01);
    let shadow_mask = 1.0
        - smoothstep(
            final_shadow_crossover - feather,
            final_shadow_crossover + feather,
            luma,
        );
    let highlight_mask = smoothstep(
        highlight_crossover - feather,
        highlight_crossover + feather,
        luma,
    );
    let midtone_mask = (1.0 - shadow_mask - highlight_mask).max(0.0);
    let mut graded = color;
    if shadows.saturation > 0.001 {
        let tint = hsv_to_rgb([shadows.hue, 1.0, 1.0]);
        graded[0] += (tint[0] - 0.5) * shadows.saturation * shadow_mask * 0.3;
        graded[1] += (tint[1] - 0.5) * shadows.saturation * shadow_mask * 0.3;
        graded[2] += (tint[2] - 0.5) * shadows.saturation * shadow_mask * 0.3;
    }
    graded[0] += shadows.luminance * shadow_mask * 0.5;
    graded[1] += shadows.luminance * shadow_mask * 0.5;
    graded[2] += shadows.luminance * shadow_mask * 0.5;
    if midtones.saturation > 0.001 {
        let tint = hsv_to_rgb([midtones.hue, 1.0, 1.0]);
        graded[0] += (tint[0] - 0.5) * midtones.saturation * midtone_mask * 0.6;
        graded[1] += (tint[1] - 0.5) * midtones.saturation * midtone_mask * 0.6;
        graded[2] += (tint[2] - 0.5) * midtones.saturation * midtone_mask * 0.6;
    }
    graded[0] += midtones.luminance * midtone_mask * 0.8;
    graded[1] += midtones.luminance * midtone_mask * 0.8;
    graded[2] += midtones.luminance * midtone_mask * 0.8;
    if highlights.saturation > 0.001 {
        let tint = hsv_to_rgb([highlights.hue, 1.0, 1.0]);
        graded[0] += (tint[0] - 0.5) * highlights.saturation * highlight_mask * 0.8;
        graded[1] += (tint[1] - 0.5) * highlights.saturation * highlight_mask * 0.8;
        graded[2] += (tint[2] - 0.5) * highlights.saturation * highlight_mask * 0.8;
    }
    graded[0] += highlights.luminance * highlight_mask;
    graded[1] += highlights.luminance * highlight_mask;
    graded[2] += highlights.luminance * highlight_mask;
    graded
}

fn apply_local_contrast(
    processed_color_linear: [f32; 3],
    blurred_color_input_space: [f32; 3],
    amount: f32,
    is_raw: u32,
    mode: u32,
) -> [f32; 3] {
    if amount == 0.0 {
        return processed_color_linear;
    }
    let center_luma = get_luma(processed_color_linear);
    let shadow_threshold = if is_raw == 1 { 0.1 } else { 0.03 };
    let shadow_protection = smoothstep(0.0, shadow_threshold, center_luma);
    let highlight_protection = 1.0 - smoothstep(0.9, 1.0, center_luma);
    let midtone_mask = shadow_protection * highlight_protection;
    if midtone_mask < 0.001 {
        return processed_color_linear;
    }
    let blurred_linear = if is_raw == 1 {
        blurred_color_input_space
    } else {
        srgb_to_linear(blurred_color_input_space)
    };
    let blurred_luma = get_luma(blurred_linear);
    let safe_center_luma = center_luma.max(0.0001);
    let safe_blurred_luma = blurred_luma.max(0.0001);
    let final_color = if amount < 0.0 {
        let blurred_projected = [
            processed_color_linear[0] * (safe_blurred_luma / safe_center_luma),
            processed_color_linear[1] * (safe_blurred_luma / safe_center_luma),
            processed_color_linear[2] * (safe_blurred_luma / safe_center_luma),
        ];
        let mut blur_amount = -amount;
        if mode == 0 {
            blur_amount *= 0.5;
        }
        [
            mix(processed_color_linear[0], blurred_projected[0], blur_amount),
            mix(processed_color_linear[1], blurred_projected[1], blur_amount),
            mix(processed_color_linear[2], blurred_projected[2], blur_amount),
        ]
    } else {
        let log_ratio = (safe_center_luma / safe_blurred_luma).log2();
        let mut effective_amount = amount;
        if mode == 0 {
            let edge_magnitude = log_ratio.abs();
            let normalized_edge = (edge_magnitude / 3.0).clamp(0.0, 1.0);
            let edge_dampener = 1.0 - normalized_edge.powf(0.5);
            effective_amount = amount * edge_dampener * 0.8;
        }
        let contrast_factor = 2.0f32.powf(log_ratio * effective_amount);
        [
            processed_color_linear[0] * contrast_factor,
            processed_color_linear[1] * contrast_factor,
            processed_color_linear[2] * contrast_factor,
        ]
    };
    [
        mix(processed_color_linear[0], final_color[0], midtone_mask),
        mix(processed_color_linear[1], final_color[1], midtone_mask),
        mix(processed_color_linear[2], final_color[2], midtone_mask),
    ]
}

fn apply_centre_local_contrast(
    color_in: [f32; 3],
    centre_amount: f32,
    x: u32,
    y: u32,
    blurred_color_input_space: [f32; 3],
    is_raw: u32,
    width: u32,
    height: u32,
) -> [f32; 3] {
    if centre_amount == 0.0 {
        return color_in;
    }

    let full_dims = [
        std::cmp::max(width, 1) as f32,
        std::cmp::max(height, 1) as f32,
    ];
    let coord_f = [x as f32, y as f32];
    let midpoint = 0.4;
    let feather = 0.375;
    let aspect = full_dims[1] / full_dims[0];
    let uv_centered = [
        (coord_f[0] / full_dims[0] - 0.5) * 2.0,
        (coord_f[1] / full_dims[1] - 0.5) * 2.0,
    ];
    let d = (uv_centered[0] * uv_centered[0] + (uv_centered[1] * aspect).powi(2)).sqrt() * 0.5;
    let vignette_mask = smoothstep(midpoint - feather, midpoint + feather, d);
    let centre_mask = 1.0 - vignette_mask;

    let clarity_strength = centre_amount * (2.0 * centre_mask - 1.0) * 0.9;
    if clarity_strength.abs() > 0.001 {
        apply_local_contrast(
            color_in,
            blurred_color_input_space,
            clarity_strength,
            is_raw,
            1,
        )
    } else {
        color_in
    }
}

fn apply_centre_tonal_and_color(
    color_in: [f32; 3],
    centre_amount: f32,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> [f32; 3] {
    if centre_amount == 0.0 {
        return color_in;
    }

    let full_dims = [
        std::cmp::max(width, 1) as f32,
        std::cmp::max(height, 1) as f32,
    ];
    let coord_f = [x as f32, y as f32];
    let midpoint = 0.4;
    let feather = 0.375;
    let aspect = full_dims[1] / full_dims[0];
    let uv_centered = [
        (coord_f[0] / full_dims[0] - 0.5) * 2.0,
        (coord_f[1] / full_dims[1] - 0.5) * 2.0,
    ];
    let d = (uv_centered[0] * uv_centered[0] + (uv_centered[1] * aspect).powi(2)).sqrt() * 0.5;
    let vignette_mask = smoothstep(midpoint - feather, midpoint + feather, d);
    let centre_mask = 1.0 - vignette_mask;

    let exposure_boost = centre_mask * centre_amount * 0.5;
    let mut processed = apply_filmic_exposure(color_in, exposure_boost);

    let vibrance_center_boost = centre_mask * centre_amount * 0.4;
    let saturation_center_boost = centre_mask * centre_amount * 0.3;
    let saturation_edge_effect = -(1.0 - centre_mask) * centre_amount * 0.8;
    let total_saturation_effect = saturation_center_boost + saturation_edge_effect;
    processed = apply_creative_color(processed, total_saturation_effect, vibrance_center_boost);
    processed
}

fn apply_glow_bloom(
    color: [f32; 3],
    blurred_color_input_space: [f32; 3],
    amount: f32,
    is_raw: u32,
    exp: f32,
    bright: f32,
    _con: f32,
    wh: f32,
) -> [f32; 3] {
    if amount <= 0.0 {
        return color;
    }
    let mut blurred_linear = if is_raw == 1 {
        blurred_color_input_space
    } else {
        srgb_to_linear(blurred_color_input_space)
    };
    blurred_linear = apply_linear_exposure(blurred_linear, exp);
    blurred_linear = apply_filmic_exposure(blurred_linear, bright);
    blurred_linear = apply_tonal_adjustments(blurred_linear, 0.0, 0.0, wh, 0.0);
    let linear_luma = get_luma([
        blurred_linear[0].max(0.0),
        blurred_linear[1].max(0.0),
        blurred_linear[2].max(0.0),
    ]);
    let perceptual_luma = if linear_luma <= 1.0 {
        linear_luma.max(0.0).powf(1.0 / 2.2)
    } else {
        1.0 + (linear_luma - 1.0).powf(1.0 / 2.2)
    };
    let luma_cutoff = mix(0.75, 0.08, amount.clamp(0.0, 1.0));
    let cutoff_fade = smoothstep(luma_cutoff, luma_cutoff + 0.15, perceptual_luma);
    let excess = (perceptual_luma - luma_cutoff).max(0.0);
    let bloom_intensity = smoothstep(0.0, 1.0, excess / 5.5).powf(0.45);
    let mut bloom_color = if linear_luma > 0.01 {
        let ratio = [
            blurred_linear[0] / linear_luma,
            blurred_linear[1] / linear_luma,
            blurred_linear[2] / linear_luma,
        ];
        [ratio[0] * 1.03, ratio[1], ratio[2] * 0.97]
    } else {
        [1.0, 0.99, 0.98]
    };
    let scalar = bloom_intensity
        * linear_luma.powf(0.6)
        * cutoff_fade
        * smoothstep(0.0, 0.5, linear_luma).powf(0.5);
    bloom_color = [
        bloom_color[0] * scalar,
        bloom_color[1] * scalar,
        bloom_color[2] * scalar,
    ];
    let current_luma = get_luma([color[0].max(0.0), color[1].max(0.0), color[2].max(0.0)]);
    let protection = 1.0 - smoothstep(1.0, 2.2, current_luma);
    [
        color[0] + bloom_color[0] * amount * 3.8 * protection,
        color[1] + bloom_color[1] * amount * 3.8 * protection,
        color[2] + bloom_color[2] * amount * 3.8 * protection,
    ]
}

fn apply_halation(
    color: [f32; 3],
    blurred_color_input_space: [f32; 3],
    amount: f32,
    is_raw: u32,
    exp: f32,
    bright: f32,
    _con: f32,
    wh: f32,
) -> [f32; 3] {
    if amount <= 0.0 {
        return color;
    }
    let mut blurred_linear = if is_raw == 1 {
        blurred_color_input_space
    } else {
        srgb_to_linear(blurred_color_input_space)
    };
    blurred_linear = apply_linear_exposure(blurred_linear, exp);
    blurred_linear = apply_filmic_exposure(blurred_linear, bright);
    blurred_linear = apply_tonal_adjustments(blurred_linear, 0.0, 0.0, wh, 0.0);
    let linear_luma = get_luma([
        blurred_linear[0].max(0.0),
        blurred_linear[1].max(0.0),
        blurred_linear[2].max(0.0),
    ]);
    let perceptual_luma = if linear_luma <= 1.0 {
        linear_luma.max(0.0).powf(1.0 / 2.2)
    } else {
        1.0 + (linear_luma - 1.0).powf(1.0 / 2.2)
    };
    let luma_cutoff = mix(0.85, 0.1, amount.clamp(0.0, 1.0));
    if perceptual_luma <= luma_cutoff {
        return color;
    }
    let halation_mask = smoothstep(
        0.0,
        (1.5 - luma_cutoff).max(0.1) * 0.6,
        perceptual_luma - luma_cutoff,
    );
    let intensity_blend = smoothstep(0.0, 0.7, halation_mask);
    let halation_tint = [
        mix(1.0, 1.0, intensity_blend),
        mix(0.32, 0.15, intensity_blend),
        mix(0.10, 0.03, intensity_blend),
    ];
    let halation_glow = [
        halation_tint[0] * halation_mask * linear_luma,
        halation_tint[1] * halation_mask * linear_luma,
        halation_tint[2] * halation_mask * linear_luma,
    ];
    let color_luma = get_luma([color[0].max(0.0), color[1].max(0.0), color[2].max(0.0)]);
    let desat_strength = halation_mask * 0.12;
    let affected = [
        mix(color[0], color_luma, desat_strength),
        mix(color[1], color_luma, desat_strength),
        mix(color[2], color_luma, desat_strength),
    ];
    let contrast_reduced = [
        mix(0.5, affected[0], 1.0 - halation_mask * 0.06),
        mix(0.5, affected[1], 1.0 - halation_mask * 0.06),
        mix(0.5, affected[2], 1.0 - halation_mask * 0.06),
    ];
    [
        contrast_reduced[0] + halation_glow[0] * amount * 2.5,
        contrast_reduced[1] + halation_glow[1] * amount * 2.5,
        contrast_reduced[2] + halation_glow[2] * amount * 2.5,
    ]
}

fn apply_all_mask_adjustments(initial_rgb: [f32; 3], adj: &MaskAdjustments) -> [f32; 3] {
    let mut processed = initial_rgb;
    processed = apply_dehaze(processed, adj.dehaze);
    processed = apply_linear_exposure(processed, adj.exposure);
    processed = apply_white_balance(processed, adj.temperature, adj.tint);
    processed = apply_filmic_exposure(processed, adj.brightness);
    processed = apply_highlights_adjustment(processed, adj.highlights);
    processed =
        apply_tonal_adjustments(processed, adj.contrast, adj.shadows, adj.whites, adj.blacks);
    processed = apply_hsl_panel(processed, &adj.hsl);
    processed = apply_color_grading(
        processed,
        adj.color_grading_shadows,
        adj.color_grading_midtones,
        adj.color_grading_highlights,
        adj.color_grading_blending,
        adj.color_grading_balance,
    );
    apply_creative_color(processed, adj.saturation, adj.vibrance)
}

fn get_mask_influence(mask_index: usize, pixel_index: usize, masks: &[Vec<f32>; 9]) -> f32 {
    masks[mask_index].get(pixel_index).copied().unwrap_or(0.0)
}

fn sample_flare_linear(
    flare: &[f32],
    flare_size: u32,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
) -> [f32; 3] {
    let fx = (x * flare_size) / std::cmp::max(width, 1);
    let fy = (y * flare_size) / std::cmp::max(height, 1);
    let base = ((fy * flare_size + fx) * 4) as usize;
    [
        flare.get(base).copied().unwrap_or(0.0),
        flare.get(base + 1).copied().unwrap_or(0.0),
        flare.get(base + 2).copied().unwrap_or(0.0),
    ]
}

fn sample_lut_tetrahedral(uv: [f32; 3], lut: &CubeclLutBuffer) -> [f32; 3] {
    let dims = [lut.size as f32, lut.size as f32, lut.size as f32];
    let size = [dims[0] - 1.0, dims[1] - 1.0, dims[2] - 1.0];
    let scaled = [
        uv[0].clamp(0.0, 1.0) * size[0],
        uv[1].clamp(0.0, 1.0) * size[1],
        uv[2].clamp(0.0, 1.0) * size[2],
    ];
    let i_base = [
        scaled[0].floor() as u32,
        scaled[1].floor() as u32,
        scaled[2].floor() as u32,
    ];
    let f = [
        scaled[0] - i_base[0] as f32,
        scaled[1] - i_base[1] as f32,
        scaled[2] - i_base[2] as f32,
    ];
    let coord0 = i_base;
    let coord1 = [
        if coord0[0] + 1 > lut.size - 1 {
            lut.size - 1
        } else {
            coord0[0] + 1
        },
        if coord0[1] + 1 > lut.size - 1 {
            lut.size - 1
        } else {
            coord0[1] + 1
        },
        if coord0[2] + 1 > lut.size - 1 {
            lut.size - 1
        } else {
            coord0[2] + 1
        },
    ];
    let c000 = lut.load_rgb(coord0[0], coord0[1], coord0[2]);
    let c111 = lut.load_rgb(coord1[0], coord1[1], coord1[2]);

    if f[0] > f[1] {
        if f[1] > f[2] {
            let c100 = lut.load_rgb(coord1[0], coord0[1], coord0[2]);
            let c110 = lut.load_rgb(coord1[0], coord1[1], coord0[2]);
            [
                c000[0] * (1.0 - f[0])
                    + c100[0] * (f[0] - f[1])
                    + c110[0] * (f[1] - f[2])
                    + c111[0] * f[2],
                c000[1] * (1.0 - f[0])
                    + c100[1] * (f[0] - f[1])
                    + c110[1] * (f[1] - f[2])
                    + c111[1] * f[2],
                c000[2] * (1.0 - f[0])
                    + c100[2] * (f[0] - f[1])
                    + c110[2] * (f[1] - f[2])
                    + c111[2] * f[2],
            ]
        } else if f[0] > f[2] {
            let c100 = lut.load_rgb(coord1[0], coord0[1], coord0[2]);
            let c101 = lut.load_rgb(coord1[0], coord0[1], coord1[2]);
            [
                c000[0] * (1.0 - f[0])
                    + c100[0] * (f[0] - f[2])
                    + c101[0] * (f[2] - f[1])
                    + c111[0] * f[1],
                c000[1] * (1.0 - f[0])
                    + c100[1] * (f[0] - f[2])
                    + c101[1] * (f[2] - f[1])
                    + c111[1] * f[1],
                c000[2] * (1.0 - f[0])
                    + c100[2] * (f[0] - f[2])
                    + c101[2] * (f[2] - f[1])
                    + c111[2] * f[1],
            ]
        } else {
            let c001 = lut.load_rgb(coord0[0], coord0[1], coord1[2]);
            let c101 = lut.load_rgb(coord1[0], coord0[1], coord1[2]);
            [
                c000[0] * (1.0 - f[2])
                    + c001[0] * (f[2] - f[0])
                    + c101[0] * (f[0] - f[1])
                    + c111[0] * f[1],
                c000[1] * (1.0 - f[2])
                    + c001[1] * (f[2] - f[0])
                    + c101[1] * (f[0] - f[1])
                    + c111[1] * f[1],
                c000[2] * (1.0 - f[2])
                    + c001[2] * (f[2] - f[0])
                    + c101[2] * (f[0] - f[1])
                    + c111[2] * f[1],
            ]
        }
    } else if f[2] > f[1] {
        let c001 = lut.load_rgb(coord0[0], coord0[1], coord1[2]);
        let c011 = lut.load_rgb(coord0[0], coord1[1], coord1[2]);
        [
            c000[0] * (1.0 - f[2])
                + c001[0] * (f[2] - f[1])
                + c011[0] * (f[1] - f[0])
                + c111[0] * f[0],
            c000[1] * (1.0 - f[2])
                + c001[1] * (f[2] - f[1])
                + c011[1] * (f[1] - f[0])
                + c111[1] * f[0],
            c000[2] * (1.0 - f[2])
                + c001[2] * (f[2] - f[1])
                + c011[2] * (f[1] - f[0])
                + c111[2] * f[0],
        ]
    } else if f[2] > f[0] {
        let c010 = lut.load_rgb(coord0[0], coord1[1], coord0[2]);
        let c011 = lut.load_rgb(coord0[0], coord1[1], coord1[2]);
        [
            c000[0] * (1.0 - f[1])
                + c010[0] * (f[1] - f[2])
                + c011[0] * (f[2] - f[0])
                + c111[0] * f[0],
            c000[1] * (1.0 - f[1])
                + c010[1] * (f[1] - f[2])
                + c011[1] * (f[2] - f[0])
                + c111[1] * f[0],
            c000[2] * (1.0 - f[1])
                + c010[2] * (f[1] - f[2])
                + c011[2] * (f[2] - f[0])
                + c111[2] * f[0],
        ]
    } else {
        let c010 = lut.load_rgb(coord0[0], coord1[1], coord0[2]);
        let c110 = lut.load_rgb(coord1[0], coord1[1], coord0[2]);
        [
            c000[0] * (1.0 - f[1])
                + c010[0] * (f[1] - f[0])
                + c110[0] * (f[0] - f[2])
                + c111[0] * f[2],
            c000[1] * (1.0 - f[1])
                + c010[1] * (f[1] - f[0])
                + c110[1] * (f[0] - f[2])
                + c111[1] * f[2],
            c000[2] * (1.0 - f[1])
                + c010[2] * (f[1] - f[0])
                + c110[2] * (f[0] - f[2])
                + c111[2] * f[2],
        ]
    }
}

fn apply_lut_in_place_srgb(output_rgba: &mut [f32], lut: &CubeclLutBuffer, lut_intensity: f32) {
    for px in output_rgba.chunks_exact_mut(4) {
        let src = [px[0], px[1], px[2]];
        let lut_color = sample_lut_tetrahedral(src, lut);
        px[0] = mix(src[0], lut_color[0], lut_intensity);
        px[1] = mix(src[1], lut_color[1], lut_intensity);
        px[2] = mix(src[2], lut_color[2], lut_intensity);
    }
}

fn blend_mask_adjustments(
    base_rgba_srgb: &[f32],
    all_adjustments: &AllAdjustments,
    mask_bitmaps: &[ImageBuffer<Luma<u8>, Vec<u8>>],
    flare_map: Option<&[f32]>,
    sharpness_blur: Option<&[f32]>,
    clarity_blur: Option<&[f32]>,
    structure_blur: Option<&[f32]>,
    width: u32,
    height: u32,
) -> (Vec<f32>, MaskCompositeStats) {
    let pixel_count = (width * height) as usize;
    let mut masks: [Vec<f32>; 9] = std::array::from_fn(|_| vec![0.0; pixel_count]);
    for i in 0..9usize {
        if let Some(bitmap) = mask_bitmaps.get(i) {
            if bitmap.dimensions() == (width, height) {
                for (idx, p) in bitmap.pixels().enumerate() {
                    masks[i][idx] = p.0[0] as f32 / 255.0;
                }
            }
        }
    }

    let mut out = base_rgba_srgb.to_vec();
    let mut active_pixels = 0usize;
    let mut influence_sum = 0.0f64;
    let mut influence_count = 0usize;
    let mut max_influence = 0.0f32;
    let min_dim = std::cmp::max(std::cmp::min(width, height), 1) as f32;
    let scale = (min_dim / 1080.0).max(0.1);
    let global = all_adjustments.global;

    for idx in 0..pixel_count {
        let x = (idx as u32) % width;
        let y = (idx as u32) / width;
        let base = idx * 4;
        let base_srgb = apply_ca_correction(
            base_rgba_srgb,
            width,
            height,
            x,
            y,
            global.chromatic_aberration_red_cyan,
            global.chromatic_aberration_blue_yellow,
        );
        let mut composite_linear = if global.is_raw_image == 1 {
            base_srgb
        } else {
            srgb_to_linear(base_srgb)
        };
        let sharp_rgb = sharpness_blur.map(|buf| [buf[base], buf[base + 1], buf[base + 2]]);
        let clarity_rgb = clarity_blur.map(|buf| [buf[base], buf[base + 1], buf[base + 2]]);
        let structure_rgb = structure_blur.map(|buf| [buf[base], buf[base + 1], buf[base + 2]]);

        if let Some(blurred) = sharp_rgb {
            composite_linear = apply_local_contrast(
                composite_linear,
                blurred,
                global.sharpness,
                global.is_raw_image,
                0,
            );
        }
        if let Some(blurred) = clarity_rgb {
            composite_linear = apply_local_contrast(
                composite_linear,
                blurred,
                global.clarity,
                global.is_raw_image,
                1,
            );
        }
        if let Some(blurred) = structure_rgb {
            composite_linear = apply_local_contrast(
                composite_linear,
                blurred,
                global.structure,
                global.is_raw_image,
                1,
            );
        }
        if let Some(blurred) = clarity_rgb {
            composite_linear = apply_centre_local_contrast(
                composite_linear,
                global.centré,
                x,
                y,
                blurred,
                global.is_raw_image,
                width,
                height,
            );
        }

        composite_linear = apply_linear_exposure(composite_linear, global.exposure);
        if global.is_raw_image == 1 && global.tonemapper_mode != 1 {
            let mut srgb_emulated = linear_to_srgb(composite_linear);
            srgb_emulated = [
                srgb_emulated[0].powf(1.0 / 1.1),
                srgb_emulated[1].powf(1.0 / 1.1),
                srgb_emulated[2].powf(1.0 / 1.1),
            ];
            let contrast_curve = [
                srgb_emulated[0] * srgb_emulated[0] * (3.0 - 2.0 * srgb_emulated[0]),
                srgb_emulated[1] * srgb_emulated[1] * (3.0 - 2.0 * srgb_emulated[1]),
                srgb_emulated[2] * srgb_emulated[2] * (3.0 - 2.0 * srgb_emulated[2]),
            ];
            srgb_emulated = [
                mix(srgb_emulated[0], contrast_curve[0], 0.75),
                mix(srgb_emulated[1], contrast_curve[1], 0.75),
                mix(srgb_emulated[2], contrast_curve[2], 0.75),
            ];
            composite_linear = srgb_to_linear(srgb_emulated);
        }

        if let Some(blurred) = structure_rgb {
            composite_linear = apply_glow_bloom(
                composite_linear,
                blurred,
                global.glow_amount,
                global.is_raw_image,
                global.exposure,
                global.brightness,
                global.contrast,
                global.whites,
            );
        }
        if let Some(blurred) = clarity_rgb {
            composite_linear = apply_halation(
                composite_linear,
                blurred,
                global.halation_amount,
                global.is_raw_image,
                global.exposure,
                global.brightness,
                global.contrast,
                global.whites,
            );
        }
        if global.flare_amount > 0.0 {
            let mut flare_color = flare_map
                .map(|m| sample_flare_linear(m, FLARE_MAP_SIZE, width, height, x, y))
                .unwrap_or([0.0, 0.0, 0.0]);
            flare_color = [
                flare_color[0] * 1.4,
                flare_color[1] * 1.4,
                flare_color[2] * 1.4,
            ];
            flare_color = [
                flare_color[0] * flare_color[0],
                flare_color[1] * flare_color[1],
                flare_color[2] * flare_color[2],
            ];
            let linear_luma = get_luma([
                composite_linear[0].max(0.0),
                composite_linear[1].max(0.0),
                composite_linear[2].max(0.0),
            ]);
            let perceptual_luma = if linear_luma <= 1.0 {
                linear_luma.max(0.0).powf(1.0 / 2.2)
            } else {
                1.0 + (linear_luma - 1.0).max(0.0).powf(1.0 / 2.2)
            };
            let protection = 1.0 - smoothstep(0.7, 1.8, perceptual_luma);
            composite_linear = [
                composite_linear[0] + flare_color[0] * global.flare_amount * protection,
                composite_linear[1] + flare_color[1] * global.flare_amount * protection,
                composite_linear[2] + flare_color[2] * global.flare_amount * protection,
            ];
        }

        composite_linear = apply_dehaze(composite_linear, global.dehaze);
        composite_linear =
            apply_centre_tonal_and_color(composite_linear, global.centré, x, y, width, height);
        composite_linear = apply_white_balance(composite_linear, global.temperature, global.tint);
        composite_linear = apply_filmic_exposure(composite_linear, global.brightness);
        composite_linear = apply_tonal_adjustments(
            composite_linear,
            global.contrast,
            global.shadows,
            global.whites,
            global.blacks,
        );
        composite_linear = apply_highlights_adjustment(composite_linear, global.highlights);
        composite_linear = apply_color_calibration(composite_linear, global.color_calibration);
        composite_linear = apply_hsl_panel(composite_linear, &global.hsl);
        composite_linear = apply_color_grading(
            composite_linear,
            global.color_grading_shadows,
            global.color_grading_midtones,
            global.color_grading_highlights,
            global.color_grading_blending,
            global.color_grading_balance,
        );
        composite_linear =
            apply_creative_color(composite_linear, global.saturation, global.vibrance);

        let mut any_active = false;
        for i in 0..std::cmp::min(all_adjustments.mask_count, 9) as usize {
            let influence = get_mask_influence(i, idx, &masks);
            influence_sum += influence as f64;
            influence_count += 1;
            max_influence = max_influence.max(influence);
            if influence <= 0.001 {
                continue;
            }
            any_active = true;
            let mask_adj = all_adjustments.mask_adjustments[i];
            let mut mask_base_linear = composite_linear;

            if let Some(blurred) = sharp_rgb {
                mask_base_linear = apply_local_contrast(
                    mask_base_linear,
                    blurred,
                    mask_adj.sharpness,
                    global.is_raw_image,
                    0,
                );
            }
            if let Some(blurred) = clarity_rgb {
                mask_base_linear = apply_local_contrast(
                    mask_base_linear,
                    blurred,
                    mask_adj.clarity,
                    global.is_raw_image,
                    1,
                );
            }
            if let Some(blurred) = structure_rgb {
                mask_base_linear = apply_local_contrast(
                    mask_base_linear,
                    blurred,
                    mask_adj.structure,
                    global.is_raw_image,
                    1,
                );
            }
            if let Some(blurred) = structure_rgb {
                mask_base_linear = apply_glow_bloom(
                    mask_base_linear,
                    blurred,
                    mask_adj.glow_amount,
                    global.is_raw_image,
                    global.exposure + mask_adj.exposure,
                    global.brightness + mask_adj.brightness,
                    global.contrast + mask_adj.contrast,
                    global.whites + mask_adj.whites,
                );
            }
            if let Some(blurred) = clarity_rgb {
                mask_base_linear = apply_halation(
                    mask_base_linear,
                    blurred,
                    mask_adj.halation_amount,
                    global.is_raw_image,
                    global.exposure + mask_adj.exposure,
                    global.brightness + mask_adj.brightness,
                    global.contrast + mask_adj.contrast,
                    global.whites + mask_adj.whites,
                );
            }

            let mut mask_adjusted_linear = apply_all_mask_adjustments(mask_base_linear, &mask_adj);
            if mask_adj.flare_amount > 0.0 {
                let mut flare_color = flare_map
                    .map(|m| sample_flare_linear(m, FLARE_MAP_SIZE, width, height, x, y))
                    .unwrap_or([0.0, 0.0, 0.0]);
                flare_color = [
                    flare_color[0] * 1.4,
                    flare_color[1] * 1.4,
                    flare_color[2] * 1.4,
                ];
                flare_color = [
                    flare_color[0] * flare_color[0],
                    flare_color[1] * flare_color[1],
                    flare_color[2] * flare_color[2],
                ];
                let mask_linear_luma = get_luma([
                    mask_adjusted_linear[0].max(0.0),
                    mask_adjusted_linear[1].max(0.0),
                    mask_adjusted_linear[2].max(0.0),
                ]);
                let mask_perceptual_luma = if mask_linear_luma <= 1.0 {
                    mask_linear_luma.max(0.0).powf(1.0 / 2.2)
                } else {
                    1.0 + (mask_linear_luma - 1.0).max(0.0).powf(1.0 / 2.2)
                };
                let protection = 1.0 - smoothstep(0.7, 1.8, mask_perceptual_luma);
                mask_adjusted_linear = [
                    mask_adjusted_linear[0] + flare_color[0] * mask_adj.flare_amount * protection,
                    mask_adjusted_linear[1] + flare_color[1] * mask_adj.flare_amount * protection,
                    mask_adjusted_linear[2] + flare_color[2] * mask_adj.flare_amount * protection,
                ];
            }
            composite_linear = [
                mix(composite_linear[0], mask_adjusted_linear[0], influence),
                mix(composite_linear[1], mask_adjusted_linear[1], influence),
                mix(composite_linear[2], mask_adjusted_linear[2], influence),
            ];
        }
        if any_active {
            active_pixels += 1;
        }

        let mut final_rgb = if global.tonemapper_mode == 1 {
            agx_full_transform(composite_linear, all_adjustments)
        } else {
            linear_to_srgb(composite_linear)
        };
        final_rgb = apply_all_curves(
            final_rgb,
            &all_adjustments.global.luma_curve,
            all_adjustments.global.luma_curve_count,
            &all_adjustments.global.red_curve,
            all_adjustments.global.red_curve_count,
            &all_adjustments.global.green_curve,
            all_adjustments.global.green_curve_count,
            &all_adjustments.global.blue_curve,
            all_adjustments.global.blue_curve_count,
        );
        for i in 0..std::cmp::min(all_adjustments.mask_count, 9) as usize {
            let influence = get_mask_influence(i, idx, &masks);
            if influence <= 0.001 {
                continue;
            }
            let mask_adj = all_adjustments.mask_adjustments[i];
            let mask_curved_srgb = apply_all_curves(
                final_rgb,
                &mask_adj.luma_curve,
                mask_adj.luma_curve_count,
                &mask_adj.red_curve,
                mask_adj.red_curve_count,
                &mask_adj.green_curve,
                mask_adj.green_curve_count,
                &mask_adj.blue_curve,
                mask_adj.blue_curve_count,
            );
            final_rgb = [
                mix(final_rgb[0], mask_curved_srgb[0], influence),
                mix(final_rgb[1], mask_curved_srgb[1], influence),
                mix(final_rgb[2], mask_curved_srgb[2], influence),
            ];
        }

        if global.grain_amount > 0.0 {
            let coord = [x as f32, y as f32];
            let amount = global.grain_amount * 0.5;
            let grain_frequency = (1.0 / global.grain_size.max(0.1)) / scale;
            let roughness = global.grain_roughness;
            let luma = get_luma(final_rgb).max(0.0);
            let luma_mask = smoothstep(0.0, 0.15, luma) * (1.0 - smoothstep(0.6, 1.0, luma));
            let base_coord = [coord[0] * grain_frequency, coord[1] * grain_frequency];
            let rough_coord = [
                coord[0] * grain_frequency * 0.6,
                coord[1] * grain_frequency * 0.6,
            ];
            let noise_base = gradient_noise(base_coord);
            let noise_rough = gradient_noise([rough_coord[0] + 5.2, rough_coord[1] + 1.3]);
            let noise_val = mix(noise_base, noise_rough, roughness);
            final_rgb = [
                final_rgb[0] + noise_val * amount * luma_mask,
                final_rgb[1] + noise_val * amount * luma_mask,
                final_rgb[2] + noise_val * amount * luma_mask,
            ];
        }

        if global.vignette_amount != 0.0 {
            let v_amount = global.vignette_amount;
            let v_mid = global.vignette_midpoint;
            let v_round = 1.0 - global.vignette_roundness;
            let v_feather = global.vignette_feather * 0.5;
            let safe_height = std::cmp::max(height, 1);
            let safe_width = std::cmp::max(width, 1);
            let aspect = safe_height as f32 / safe_width as f32;
            let uv_centered = [
                ((x as f32 / safe_width as f32) - 0.5) * 2.0,
                ((y as f32 / safe_height as f32) - 0.5) * 2.0,
            ];
            let uv_round = [
                uv_centered[0].signum() * uv_centered[0].abs().powf(v_round),
                uv_centered[1].signum() * uv_centered[1].abs().powf(v_round),
            ];
            let d = (uv_round[0] * uv_round[0] + (uv_round[1] * aspect).powi(2)).sqrt() * 0.5;
            let vignette_mask = smoothstep(v_mid - v_feather, v_mid + v_feather, d);
            if v_amount < 0.0 {
                let m = 1.0 + v_amount * vignette_mask;
                final_rgb = [final_rgb[0] * m, final_rgb[1] * m, final_rgb[2] * m];
            } else {
                final_rgb = [
                    mix(final_rgb[0], 1.0, v_amount * vignette_mask),
                    mix(final_rgb[1], 1.0, v_amount * vignette_mask),
                    mix(final_rgb[2], 1.0, v_amount * vignette_mask),
                ];
            }
        }

        if global.show_clipping == 1 {
            if final_rgb[0] > 0.998 || final_rgb[1] > 0.998 || final_rgb[2] > 0.998 {
                final_rgb = [1.0, 0.0, 0.0];
            } else if final_rgb[0] < 0.002 || final_rgb[1] < 0.002 || final_rgb[2] < 0.002 {
                final_rgb = [0.0, 0.0, 1.0];
            }
        }

        let dither_amount = 1.0 / 255.0;
        let d = dither_noise(x, y) * dither_amount;
        final_rgb = [final_rgb[0] + d, final_rgb[1] + d, final_rgb[2] + d];

        out[base] = final_rgb[0];
        out[base + 1] = final_rgb[1];
        out[base + 2] = final_rgb[2];
    }

    let mean_influence = if influence_count == 0 {
        0.0
    } else {
        (influence_sum / influence_count as f64) as f32
    };
    (
        out,
        MaskCompositeStats {
            mask_count: std::cmp::min(all_adjustments.mask_count, 9),
            active_pixels,
            max_influence,
            mean_influence,
        },
    )
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

fn curves_active(adjustments: &AllAdjustments) -> bool {
    let g = adjustments.global;
    if g.luma_curve_count != 0
        || g.red_curve_count != 0
        || g.green_curve_count != 0
        || g.blue_curve_count != 0
    {
        return true;
    }
    for i in 0..std::cmp::min(adjustments.mask_count, 9) as usize {
        let m = adjustments.mask_adjustments[i];
        if m.luma_curve_count != 0
            || m.red_curve_count != 0
            || m.green_curve_count != 0
            || m.blue_curve_count != 0
        {
            return true;
        }
    }
    false
}

fn hsl_active(adjustments: &AllAdjustments) -> bool {
    let g_hsl = adjustments.global.hsl;
    for entry in g_hsl {
        let (h, s, l) = hsl_fields(&entry);
        if !near_zero(h) || !near_zero(s) || !near_zero(l) {
            return true;
        }
    }
    for i in 0..std::cmp::min(adjustments.mask_count, 9) as usize {
        let m_hsl = adjustments.mask_adjustments[i].hsl;
        for entry in m_hsl {
            let (h, s, l) = hsl_fields(&entry);
            if !near_zero(h) || !near_zero(s) || !near_zero(l) {
                return true;
            }
        }
    }
    false
}

fn color_grading_active(adjustments: &AllAdjustments) -> bool {
    let g = adjustments.global;
    if !color_grade_is_zero(g.color_grading_shadows)
        || !color_grade_is_zero(g.color_grading_midtones)
        || !color_grade_is_zero(g.color_grading_highlights)
        || !near_zero(g.color_grading_blending)
        || !near_zero(g.color_grading_balance)
    {
        return true;
    }
    for i in 0..std::cmp::min(adjustments.mask_count, 9) as usize {
        let m = adjustments.mask_adjustments[i];
        if !color_grade_is_zero(m.color_grading_shadows)
            || !color_grade_is_zero(m.color_grading_midtones)
            || !color_grade_is_zero(m.color_grading_highlights)
            || !near_zero(m.color_grading_blending)
            || !near_zero(m.color_grading_balance)
        {
            return true;
        }
    }
    false
}

fn color_calibration_active(adjustments: &AllAdjustments) -> bool {
    !color_calibration_is_zero(adjustments.global.color_calibration)
}

fn global_local_contrast_active(adjustments: &AllAdjustments) -> bool {
    let g = adjustments.global;
    !near_zero(g.sharpness)
        || !near_zero(g.clarity)
        || !near_zero(g.structure)
        || !near_zero(g.centré)
}

fn global_advanced_effects_active(adjustments: &AllAdjustments) -> bool {
    let g = adjustments.global;
    !near_zero(g.dehaze)
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
}

fn mask_dehaze_active(adjustments: &AllAdjustments) -> bool {
    for i in 0..std::cmp::min(adjustments.mask_count, 9) as usize {
        if !near_zero(adjustments.mask_adjustments[i].dehaze) {
            return true;
        }
    }
    false
}

fn main_tonal_color_active(adjustments: &AllAdjustments) -> bool {
    let g = adjustments.global;
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
        return true;
    }
    for i in 0..std::cmp::min(adjustments.mask_count, 9) as usize {
        let m = adjustments.mask_adjustments[i];
        if !near_zero(m.brightness)
            || !near_zero(m.contrast)
            || !near_zero(m.highlights)
            || !near_zero(m.shadows)
            || !near_zero(m.whites)
            || !near_zero(m.blacks)
            || !near_zero(m.temperature)
            || !near_zero(m.tint)
            || !near_zero(m.saturation)
            || !near_zero(m.vibrance)
        {
            return true;
        }
    }
    false
}

fn parity_dashboard(flags: CubeclParityFlags) -> String {
    format!(
        "flags:dehaze={} glow={} halation={} vignette={} grain={} ca={} brightness={} contrast={} highlights={} shadows={} whites={} blacks={} temperature={} tint={} saturation={} vibrance={} clipping={}",
        flags.dehaze as u8,
        flags.glow as u8,
        flags.halation as u8,
        flags.vignette as u8,
        flags.grain as u8,
        flags.chromatic_aberration as u8,
        flags.brightness as u8,
        flags.contrast as u8,
        flags.highlights as u8,
        flags.shadows as u8,
        flags.whites as u8,
        flags.blacks as u8,
        flags.temperature as u8,
        flags.tint as u8,
        flags.saturation as u8,
        flags.vibrance as u8,
        flags.clipping_overlay as u8
    )
}

fn unsupported_reason(
    adjustments: &AllAdjustments,
    flags: CubeclParityFlags,
) -> Option<&'static str> {
    let g = adjustments.global;

    if g.show_clipping != 0 && !flags.clipping_overlay {
        return Some("clipping overlay is disabled in CubeCL by feature flag");
    }
    if (!near_zero(g.dehaze) || mask_dehaze_active(adjustments)) && !flags.dehaze {
        return Some("dehaze is disabled in CubeCL by feature flag");
    }
    if !near_zero(g.glow_amount) && !flags.glow {
        return Some("glow is disabled in CubeCL by feature flag");
    }
    if !near_zero(g.halation_amount) && !flags.halation {
        return Some("halation is disabled in CubeCL by feature flag");
    }
    if (!near_zero(g.vignette_amount)
        || !near_zero(g.vignette_midpoint)
        || !near_zero(g.vignette_roundness)
        || !near_zero(g.vignette_feather))
        && !flags.vignette
    {
        return Some("vignette is disabled in CubeCL by feature flag");
    }
    if (!near_zero(g.grain_amount) || !near_zero(g.grain_size) || !near_zero(g.grain_roughness))
        && !flags.grain
    {
        return Some("grain is disabled in CubeCL by feature flag");
    }
    if (!near_zero(g.chromatic_aberration_red_cyan)
        || !near_zero(g.chromatic_aberration_blue_yellow))
        && !flags.chromatic_aberration
    {
        return Some("chromatic aberration is disabled in CubeCL by feature flag");
    }
    if (!near_zero(g.brightness)
        || (0..std::cmp::min(adjustments.mask_count, 9) as usize)
            .any(|i| !near_zero(adjustments.mask_adjustments[i].brightness)))
        && !flags.brightness
    {
        return Some("brightness is disabled in CubeCL by feature flag");
    }
    if (!near_zero(g.contrast)
        || (0..std::cmp::min(adjustments.mask_count, 9) as usize)
            .any(|i| !near_zero(adjustments.mask_adjustments[i].contrast)))
        && !flags.contrast
    {
        return Some("contrast is disabled in CubeCL by feature flag");
    }
    if (!near_zero(g.highlights)
        || (0..std::cmp::min(adjustments.mask_count, 9) as usize)
            .any(|i| !near_zero(adjustments.mask_adjustments[i].highlights)))
        && !flags.highlights
    {
        return Some("highlights is disabled in CubeCL by feature flag");
    }
    if (!near_zero(g.shadows)
        || (0..std::cmp::min(adjustments.mask_count, 9) as usize)
            .any(|i| !near_zero(adjustments.mask_adjustments[i].shadows)))
        && !flags.shadows
    {
        return Some("shadows is disabled in CubeCL by feature flag");
    }
    if (!near_zero(g.whites)
        || (0..std::cmp::min(adjustments.mask_count, 9) as usize)
            .any(|i| !near_zero(adjustments.mask_adjustments[i].whites)))
        && !flags.whites
    {
        return Some("whites is disabled in CubeCL by feature flag");
    }
    if (!near_zero(g.blacks)
        || (0..std::cmp::min(adjustments.mask_count, 9) as usize)
            .any(|i| !near_zero(adjustments.mask_adjustments[i].blacks)))
        && !flags.blacks
    {
        return Some("blacks is disabled in CubeCL by feature flag");
    }
    if (!near_zero(g.temperature)
        || (0..std::cmp::min(adjustments.mask_count, 9) as usize)
            .any(|i| !near_zero(adjustments.mask_adjustments[i].temperature)))
        && !flags.temperature
    {
        return Some("temperature is disabled in CubeCL by feature flag");
    }
    if (!near_zero(g.tint)
        || (0..std::cmp::min(adjustments.mask_count, 9) as usize)
            .any(|i| !near_zero(adjustments.mask_adjustments[i].tint)))
        && !flags.tint
    {
        return Some("tint is disabled in CubeCL by feature flag");
    }
    if (!near_zero(g.saturation)
        || (0..std::cmp::min(adjustments.mask_count, 9) as usize)
            .any(|i| !near_zero(adjustments.mask_adjustments[i].saturation)))
        && !flags.saturation
    {
        return Some("saturation is disabled in CubeCL by feature flag");
    }
    if (!near_zero(g.vibrance)
        || (0..std::cmp::min(adjustments.mask_count, 9) as usize)
            .any(|i| !near_zero(adjustments.mask_adjustments[i].vibrance)))
        && !flags.vibrance
    {
        return Some("vibrance is disabled in CubeCL by feature flag");
    }

    for i in 0..std::cmp::min(adjustments.mask_count, 9) as usize {
        let m = adjustments.mask_adjustments[i];
        if m.luma_noise_reduction > 100.0 || m.color_noise_reduction > 100.0 {
            return Some("mask noise reduction > 100 is not yet implemented in CubeCL");
        }
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
    mask_bitmaps: &[ImageBuffer<Luma<u8>, Vec<u8>>],
    lut: Option<&Lut>,
    wgsl_fallback_pixels: Option<&[u8]>,
) -> Result<CubeclRunResult, String> {
    let total_start = Instant::now();
    let (width, height) = base_image.dimensions();
    let flags = cubecl_parity_flags();
    let parity_note = parity_dashboard(flags);

    let fallback_result = |reason: String| -> Result<CubeclRunResult, String> {
        if let Some(fallback) = wgsl_fallback_pixels {
            return Ok(CubeclRunResult {
                pixels: fallback.to_vec(),
                timings: CubeclTimings {
                    total: total_start.elapsed(),
                    flare_threshold: Duration::ZERO,
                    flare_blur: Duration::ZERO,
                    main: Duration::ZERO,
                    mask_composite: Duration::ZERO,
                },
                mask_stats: MaskCompositeStats::default(),
                used_wgsl_fallback: true,
                fallback_reason: Some(reason),
                parity_dashboard: parity_note.clone(),
            });
        }
        Err(format!(
            "CubeCL path cannot execute this adjustment set without WGSL fallback: {}",
            reason
        ))
    };

    let lut_buffer = if all_adjustments.global.has_lut != 0 {
        let Some(lut_ref) = lut else {
            return fallback_result(
                "LUT is enabled but CubeCL LUT data was not provided".to_string(),
            );
        };
        let lut_buffer = match CubeclLutBuffer::from_lut(lut_ref) {
            Ok(buffer) => buffer,
            Err(error) => return fallback_result(error),
        };
        Some(lut_buffer)
    } else {
        None
    };

    if let Some(reason) = unsupported_reason(&all_adjustments, flags) {
        return fallback_result(reason.to_string());
    }

    let rgba_f32 = base_image.to_rgba32f();
    let input = rgba_f32.as_raw();
    let client = cubecl::wgpu::WgpuRuntime::client(&*CUBECL_DEVICE);

    let mut flare_threshold_time = Duration::ZERO;
    let mut flare_blur_time = Duration::ZERO;
    let mut mask_composite_time = Duration::ZERO;

    let any_mask_flare = (0..std::cmp::min(all_adjustments.mask_count, 9) as usize)
        .any(|i| !near_zero(all_adjustments.mask_adjustments[i].flare_amount));
    let flare_map = if all_adjustments.global.flare_amount > 0.0 || any_mask_flare {
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

    let needs_cpu_blend_path = all_adjustments.global.tonemapper_mode == 1
        || curves_active(&all_adjustments)
        || hsl_active(&all_adjustments);
    let needs_cpu_blend_path = needs_cpu_blend_path
        || color_grading_active(&all_adjustments)
        || color_calibration_active(&all_adjustments)
        || global_local_contrast_active(&all_adjustments)
        || global_advanced_effects_active(&all_adjustments)
        || main_tonal_color_active(&all_adjustments)
        || all_adjustments.global.show_clipping != 0;

    let main_start = Instant::now();
    let mut output = if needs_cpu_blend_path {
        run_copy_kernel_tiled(&client, input, width, height)
    } else if let Some(flare_map) = flare_map.as_ref() {
        run_main_flare_composite_kernel_tiled(
            &client,
            input,
            Some(flare_map),
            width,
            height,
            all_adjustments.global.flare_amount
        )
    } else {
        run_copy_kernel_tiled(&client, input, width, height)
    };
    let main_time = main_start.elapsed();
    let mut mask_stats = MaskCompositeStats::default();

    if needs_cpu_blend_path {
        let mask_start = Instant::now();
        let mut sharpness_blur: Option<Vec<f32>> = None;
        let mut clarity_blur: Option<Vec<f32>> = None;
        let mut structure_blur: Option<Vec<f32>> = None;
        let need_blurs = !near_zero(all_adjustments.global.sharpness)
            || !near_zero(all_adjustments.global.clarity)
            || !near_zero(all_adjustments.global.structure)
            || !near_zero(all_adjustments.global.centré)
            || !near_zero(all_adjustments.global.glow_amount)
            || !near_zero(all_adjustments.global.halation_amount)
            || (0..std::cmp::min(all_adjustments.mask_count, 9) as usize).any(|i| {
                let m = all_adjustments.mask_adjustments[i];
                !near_zero(m.sharpness)
                    || !near_zero(m.clarity)
                    || !near_zero(m.structure)
                    || !near_zero(m.glow_amount)
                    || !near_zero(m.halation_amount)
            });
        if need_blurs {
            let scale = f32::max(std::cmp::min(width, height) as f32 / 1080.0_f32, 0.1_f32);
            let sharpness_radius = f32::max((2.0_f32 * scale).ceil(), 1.0_f32) as u32;
            let clarity_radius = f32::max((8.0_f32 * scale).ceil(), 1.0_f32) as u32;
            let structure_radius = f32::max((40.0_f32 * scale).ceil(), 1.0_f32) as u32;
            sharpness_blur = Some(run_gaussian_blur_kernel(
                &client,
                input,
                width,
                height,
                sharpness_radius,
            ));
            clarity_blur = Some(run_gaussian_blur_kernel(
                &client,
                input,
                width,
                height,
                clarity_radius,
            ));
            structure_blur = Some(run_gaussian_blur_kernel(
                &client,
                input,
                width,
                height,
                structure_radius,
            ));
        }
        let blended = blend_mask_adjustments(
            input,
            &all_adjustments,
            mask_bitmaps,
            flare_map.as_deref(),
            sharpness_blur.as_deref(),
            clarity_blur.as_deref(),
            structure_blur.as_deref(),
            width,
            height,
        );
        output = blended.0;
        mask_stats = blended.1;
        mask_composite_time = mask_start.elapsed();
    } else if all_adjustments.mask_count > 0 {
        let mask_start = Instant::now();
        let blended = run_mask_composite_kernel(
            &client,
            &output,
            input,
            width,
            height,
            &all_adjustments,
            mask_bitmaps,
            flare_map.as_deref(),
        );
        output = blended.0;
        mask_stats = blended.1;
        mask_composite_time = mask_start.elapsed();
    }

    if let Some(lut_buffer) = lut_buffer.as_ref() {
        apply_lut_in_place_srgb(
            &mut output,
            lut_buffer,
            all_adjustments.global.lut_intensity,
        );
    }

    Ok(CubeclRunResult {
        pixels: f32_rgba_to_u8(&output),
        timings: CubeclTimings {
            total: total_start.elapsed(),
            flare_threshold: flare_threshold_time,
            flare_blur: flare_blur_time,
            main: main_time,
            mask_composite: mask_composite_time,
        },
        mask_stats,
        used_wgsl_fallback: false,
        fallback_reason: None,
        parity_dashboard: parity_note,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn known_size2_lut() -> CubeclLutBuffer {
        // LUT texel order is X-major within Y-major within Z-major.
        let rgb_data = vec![
            0.1, 0.2, 0.3, // (0,0,0)
            0.4, 0.5, 0.6, // (1,0,0)
            0.3, 0.9, 0.7, // (0,1,0)
            0.7, 0.1, 0.2, // (1,1,0)
            0.2, 0.7, 0.4, // (0,0,1)
            0.6, 0.3, 0.9, // (1,0,1)
            0.8, 0.2, 0.5, // (0,1,1)
            0.9, 0.8, 0.1, // (1,1,1)
        ];
        CubeclLutBuffer::from_lut(&Lut {
            size: 2,
            data: rgb_data,
        })
        .expect("valid LUT")
    }

    fn assert_rgb_close(actual: [f32; 3], expected: [f32; 3], eps: f32) {
        assert!(
            (actual[0] - expected[0]).abs() <= eps
                && (actual[1] - expected[1]).abs() <= eps
                && (actual[2] - expected[2]).abs() <= eps,
            "actual={:?} expected={:?} eps={}",
            actual,
            expected,
            eps
        );
    }

    #[test]
    fn cubecl_lut_tetrahedral_golden_case_rgb_gt_gb_gt_b() {
        let lut = known_size2_lut();
        let sampled = sample_lut_tetrahedral([0.7, 0.4, 0.2], &lut);
        assert_rgb_close(sampled, [0.47, 0.39, 0.33], 1.0e-6);
    }

    #[test]
    fn cubecl_lut_tetrahedral_golden_case_b_gt_g_gt_r() {
        let lut = known_size2_lut();
        let sampled = sample_lut_tetrahedral([0.2, 0.3, 0.8], &lut);
        assert_rgb_close(sampled, [0.38, 0.57, 0.33], 1.0e-6);
    }
}
