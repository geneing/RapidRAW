use crate::image_processing::apply_orientation;
use anyhow::{anyhow, Result};
use image::{DynamicImage, ImageBuffer, Rgba};
use rawler::{
    decoders::{Orientation, RawDecodeParams},
    formats::tiff::Value,
    imgop::develop::{DemosaicAlgorithm, Intermediate, ProcessingStep, RawDevelop},
    rawimage::{RawImage, RawImageData, RawPhotometricInterpretation},
    rawsource::RawSource,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

pub fn develop_raw_image(
    file_bytes: &[u8],
    fast_demosaic: bool,
    highlight_compression: f32,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<DynamicImage> {
    let (developed_image, orientation) = develop_internal(
        file_bytes,
        fast_demosaic,
        highlight_compression,
        cancel_token,
    )?;
    Ok(apply_orientation(developed_image, orientation))
}

fn is_linear_raw_format(raw_image: &RawImage) -> bool {
    matches!(raw_image.photometric, RawPhotometricInterpretation::LinearRaw)
}

fn log_dng_info(raw_image: &RawImage) {
    log::info!("--- DNG Debug Information ---");
    log::info!("  Make: {}, Model: {}", raw_image.make, raw_image.model);
    log::info!("  Photometric Interpretation: {:?}", raw_image.photometric);
    log::info!(
        "  Data Type: {}",
        match raw_image.data {
            RawImageData::Integer(_) => "Integer",
            RawImageData::Float(_) => "Float",
        }
    );
    log::info!("  Bits Per Sample: {}", raw_image.bps);
    log::info!("  Black Level: {:?}", raw_image.blacklevel);
    log::info!("  White Level: {:?}", raw_image.whitelevel);

    log::info!("--- Relevant DNG Tags ---");
    let tags_to_check = [
        (50879, "ColorimetricReference"),
        (50940, "ProfileToneCurve"),
        (50712, "LinearizationTable"),
        (50706, "DNGVersion"),
    ];

    for (id, name) in tags_to_check {
        if let Some(val) = raw_image.dng_tags.get(&id) {
            log::info!("  - {}({}): {:?}", name, id, val);
        } else {
            log::info!("  - {}({}): Not present", name, id);
        }
    }
    log::info!("-----------------------------");
}

fn needs_srgb_ungamma(raw_image: &RawImage) -> bool {
    if !is_linear_raw_format(raw_image) {
        return false;
    }

    if let Some(value) = raw_image.dng_tags.get(&50879) {
        let reference = match value {
            Value::Short(v) if !v.is_empty() => Some(v[0] as u32),
            Value::Long(v) if !v.is_empty() => Some(v[0]),
            _ => None,
        };

        if let Some(1) = reference {
            log::debug!("Heuristic: ColorimetricReference is Scene-Referred. Data is linear. NO un-gamma.");
            return false;
        } else if let Some(0) = reference {
            log::debug!("Heuristic: ColorimetricReference is Output-Referred. Applying un-gamma.");
            return true;
        }
    }

    if matches!(raw_image.data, RawImageData::Float(_)) {
        log::debug!("Heuristic: Data is Float32. Assuming linear data. NO un-gamma.");
        return false;
    }

    if raw_image.dng_tags.contains_key(&50940) {
        log::debug!("Heuristic: ProfileToneCurve found. Assuming linear base data. NO un-gamma.");
        return false;
    }

    if raw_image.bps >= 16 {
        log::debug!("Heuristic: 16-bit (or higher) Integer data with no explicit Gamma tag. Assuming linear. NO un-gamma.");
        return false;
    }

    log::debug!("Heuristic: Fallback. Low-bit-depth Integer LinearRaw with no markers. Assuming sRGB. Applying un-gamma.");
    true
}

#[inline]
fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(3.0)
    }
}

fn develop_internal(
    file_bytes: &[u8],
    fast_demosaic: bool,
    highlight_compression: f32,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<(DynamicImage, Orientation)> {
    let check_cancel = || -> Result<()> {
        if let Some((tracker, generation)) = &cancel_token {
            if tracker.load(Ordering::SeqCst) != *generation {
                return Err(anyhow!("Load cancelled"));
            }
        }
        Ok(())
    };

    check_cancel()?;

    let source = RawSource::new_from_slice(file_bytes);
    let decoder = rawler::get_decoder(&source)?;

    check_cancel()?;
    let mut raw_image: RawImage = decoder.raw_image(&source, &RawDecodeParams::default(), false)?;

    let metadata = decoder.raw_metadata(&source, &RawDecodeParams::default())?;
    let orientation = metadata
        .exif
        .orientation
        .map(Orientation::from_u16)
        .unwrap_or(Orientation::Normal);

    let is_linear_format = is_linear_raw_format(&raw_image);

    if is_linear_format {
        log_dng_info(&raw_image);
    }

    let should_ungamma = needs_srgb_ungamma(&raw_image);

    if is_linear_format {
        if should_ungamma {
            log::info!("Detected Linear Raw DNG (sRGB) - will apply degamma");
        } else {
            log::info!("Detected Linear Raw DNG (Linear) - skipping degamma");
        }
    }

    let original_white_level = raw_image
        .whitelevel
        .0
        .get(0)
        .cloned()
        .unwrap_or(u16::MAX as u32) as f32;
    let original_black_level = raw_image
        .blacklevel
        .levels
        .get(0)
        .map(|r| r.as_f32())
        .unwrap_or(0.0);

    for level in raw_image.whitelevel.0.iter_mut() {
        *level = u32::MAX;
    }

    let mut developer = RawDevelop::default();

    if is_linear_format {
        developer.steps.retain(|&step| {
            step != ProcessingStep::SRgb
                && step != ProcessingStep::Demosaic
                && step != ProcessingStep::Calibrate
        });
    } else if fast_demosaic {
        developer.demosaic_algorithm = DemosaicAlgorithm::Speed;
        developer.steps.retain(|&step| step != ProcessingStep::SRgb);
    } else {
        developer.steps.retain(|&step| step != ProcessingStep::SRgb);
    }

    check_cancel()?;
    let mut developed_intermediate = developer.develop_intermediate(&raw_image)?;

    drop(raw_image);

    let denominator = (original_white_level - original_black_level).max(1.0);
    let rescale_factor = (u32::MAX as f32 - original_black_level) / denominator;
    let safe_highlight_compression = highlight_compression.max(1.01);

    check_cancel()?;

    match &mut developed_intermediate {
        Intermediate::Monochrome(pixels) => {
            pixels.data.iter_mut().for_each(|p| {
                let mut linear_val = *p * rescale_factor;
                if should_ungamma {
                    linear_val = srgb_to_linear(linear_val.clamp(0.0, 1.0));
                }
                *p = linear_val;
            });
        }
        Intermediate::ThreeColor(pixels) => {
            pixels.data.iter_mut().for_each(|p| {
                let mut r = (p[0] * rescale_factor).max(0.0);
                let mut g = (p[1] * rescale_factor).max(0.0);
                let mut b = (p[2] * rescale_factor).max(0.0);

                if should_ungamma {
                    r = srgb_to_linear(r.clamp(0.0, 1.0));
                    g = srgb_to_linear(g.clamp(0.0, 1.0));
                    b = srgb_to_linear(b.clamp(0.0, 1.0));
                    let lum = r * 0.2126 + g * 0.7152 + b * 0.0722;
                    r = (lum + (r - lum) * 2.0).max(0.0);
                    g = (lum + (g - lum) * 2.0).max(0.0);
                    b = (lum + (b - lum) * 2.0).max(0.0);
                }

                let max_c = r.max(g).max(b);

                let (final_r, final_g, final_b) = if max_c > 1.0 {
                    let min_c = r.min(g).min(b);
                    let compression_factor = (1.0
                        - (max_c - 1.0) / (safe_highlight_compression - 1.0))
                        .max(0.0)
                        .min(1.0);
                    let compressed_r = min_c + (r - min_c) * compression_factor;
                    let compressed_g = min_c + (g - min_c) * compression_factor;
                    let compressed_b = min_c + (b - min_c) * compression_factor;
                    let compressed_max = compressed_r.max(compressed_g).max(compressed_b);

                    if compressed_max > 1e-6 {
                        let rescale = max_c / compressed_max;
                        (
                            compressed_r * rescale,
                            compressed_g * rescale,
                            compressed_b * rescale,
                        )
                    } else {
                        (max_c, max_c, max_c)
                    }
                } else {
                    (r, g, b)
                };

                p[0] = final_r;
                p[1] = final_g;
                p[2] = final_b;
            });
        }
        Intermediate::FourColor(pixels) => {
            pixels.data.iter_mut().for_each(|p| {
                p.iter_mut().for_each(|c| {
                    let mut linear_val = *c * rescale_factor;
                    if should_ungamma {
                        linear_val = srgb_to_linear(linear_val.clamp(0.0, 1.0));
                    }
                    *c = linear_val;
                });
            });
        }
    }

    let (width, height) = {
        let dim = developed_intermediate.dim();
        (dim.w as u32, dim.h as u32)
    };

    check_cancel()?;

    let dynamic_image = match developed_intermediate {
        Intermediate::ThreeColor(pixels) => {
            let buffer = ImageBuffer::<Rgba<f32>, _>::from_fn(width, height, |x, y| {
                let p = pixels.data[(y * width + x) as usize];
                Rgba([p[0], p[1], p[2], 1.0])
            });
            DynamicImage::ImageRgba32F(buffer)
        }
        Intermediate::Monochrome(pixels) => {
            let buffer = ImageBuffer::<Rgba<f32>, _>::from_fn(width, height, |x, y| {
                let p = pixels.data[(y * width + x) as usize];
                Rgba([p, p, p, 1.0])
            });
            DynamicImage::ImageRgba32F(buffer)
        }
        _ => {
            return Err(anyhow!("Unsupported intermediate format for conversion"));
        }
    };

    Ok((dynamic_image, orientation))
}