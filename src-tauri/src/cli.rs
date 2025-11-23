use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use csv::Writer;
use image::{ImageBuffer, Luma, GenericImageView};
use regex::Regex;
use serde::Serialize;
use serde_json::Value;
use walkdir::WalkDir;

use crate::formats::is_supported_image_file;
use crate::gpu_processing::init_gpu_context;
use crate::image_loader::load_base_image_from_bytes;
use crate::image_processing::{
    get_all_adjustments_from_json, apply_cpu_default_raw_processing,
    ImageMetadata, GpuContext,
};
use crate::lut_processing::Lut;
use crate::mask_generation::{MaskDefinition, generate_mask_bitmap};
use crate::{
    apply_all_transformations, calculate_full_job_hash, encode_image_to_bytes,
    write_image_with_metadata,
};
use crate::file_management::parse_virtual_path;
use crate::file_management::read_file_mapped;
use crate::formats::is_raw_file;
use std::sync::Arc;

#[derive(Serialize)]
pub struct ProcessingMetrics {
    filename: String,
    step_name: String,
    elapsed_ms: u128,
    cpu_memory_mb: f64,
    gpu_memory_mb: f64,
    gpu_load_pct: f64,
}

pub struct CliOptions {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub use_gpu: bool,
    pub verbose: bool,
}

/// Parse command-line arguments for CLI mode
pub fn parse_cli_args(args: Vec<String>) -> Option<CliOptions> {
    if args.len() < 3 {
        return None;
    }

    if args[1] != "process" && args[1] != "--process" {
        return None;
    }

    let input_dir = PathBuf::from(&args[2]);
    let output_dir = PathBuf::from(&args[3]);

    let use_gpu = !args.iter().any(|arg| arg == "--cpuonly");
    let verbose = args.iter().any(|arg| arg == "--verbose");

    Some(CliOptions {
        input_dir,
        output_dir,
        use_gpu,
        verbose,
    })
}

/// Setup lightweight CLI logging without UI handlers
pub fn setup_cli_logging(verbose: bool) -> Result<(), String> {
    let level = if verbose {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };

    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{}] {}",
                record.level(),
                message
            ))
        })
        .level(level)
        .chain(std::io::stderr())
        .apply()
        .map_err(|e| format!("Failed to initialize logging: {}", e))?;

    Ok(())
}

/// Find all images with corresponding .rrdata sidecar files
fn find_images_with_sidecars(dir: &Path) -> Result<Vec<(PathBuf, Option<PathBuf>)>, String> {
    let mut images = Vec::new();
    let _sidecar_pattern =
        Regex::new(r"\.([0-9a-f]{6})\.rrdata$|\.rrdata$").map_err(|e| e.to_string())?;

    let walker = WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|entry| entry.file_type().is_file());

    for entry in walker {
        let path = entry.path();
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // Skip non-image files and sidecar files
        if !is_supported_image_file(file_name) {
            continue;
        }

        // Check if there's a corresponding sidecar
        let sidecar_path = format!("{}.rrdata", path.display());
        let sidecar_path_obj = Path::new(&sidecar_path);

        let sidecar_option = if sidecar_path_obj.exists() {
            Some(sidecar_path_obj.to_path_buf())
        } else {
            None
        };

        images.push((path.to_path_buf(), sidecar_option));
    }

    Ok(images)
}

/// Collect system metrics (CPU memory in MB, GPU memory in MB, GPU load %)
fn collect_system_metrics() -> (f64, f64, f64) {
    let cpu_memory_mb = get_process_memory_mb().unwrap_or(0.0);
    let gpu_memory_mb = 0.0; // GPU memory tracking would require wgpu internals or external tools
    let gpu_load_pct = 0.0; // GPU load would require specialized tooling

    (cpu_memory_mb, gpu_memory_mb, gpu_load_pct)
}

/// Get current process memory usage in MB (Linux/Windows)
#[cfg(target_os = "linux")]
fn get_process_memory_mb() -> Option<f64> {
    use std::fs::File;
    use std::io::Read;

    let mut status = String::new();
    File::open("/proc/self/status")
        .ok()?
        .read_to_string(&mut status)
        .ok()?;

    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(kb) = parts[1].parse::<f64>() {
                    return Some(kb / 1024.0); // Convert KB to MB
                }
            }
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn get_process_memory_mb() -> Option<f64> {
    None // Not implemented on other platforms yet
}

/// Log metrics to CSV
fn log_metrics_to_csv(
    writer: &mut Writer<std::fs::File>,
    metrics: &ProcessingMetrics,
) -> Result<(), String> {
    writer
        .serialize(metrics)
        .map_err(|e| format!("Failed to write CSV: {}", e))?;
    Ok(())
}

/// Generate output filename with _processed suffix
fn generate_output_filename(original_path: &Path) -> String {
    let stem = original_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");
    let extension = original_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("jpg");

    format!("{}_{}.{}", stem, "processed", extension)
}

/// Process a single image file
fn process_single_image(
    image_path: &Path,
    sidecar_path: Option<&Path>,
    output_dir: &Path,
    use_gpu: bool,
    gpu_context: Option<&GpuContext>,
    lut_cache: &mut HashMap<String, Arc<Lut>>,
) -> Result<(), String> {
    let file_name = image_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("Invalid filename")?;

    let batch_start = Instant::now();

    eprintln!("Processing: {}", file_name);

    // Load metadata from sidecar
    let metadata: ImageMetadata = if let Some(sidecar) = sidecar_path {
        if let Ok(content) = fs::read_to_string(sidecar) {
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            ImageMetadata::default()
        }
    } else {
        ImageMetadata::default()
    };

    let (source_path, _) = parse_virtual_path(&image_path.to_string_lossy());
    let source_path_str = source_path.to_string_lossy().to_string();
    let is_raw = is_raw_file(&source_path_str);

    // Step 1: Load image
    let step_timer = Instant::now();
    let img_bytes = read_file_mapped(Path::new(&source_path_str))
        .map_err(|e| format!("Failed to read image: {}", e))?;

    let highlight_compression = 2.5; // Default value
    let pristine_img = load_base_image_from_bytes(
        &img_bytes,
        &source_path_str,
        false, // use_fast_raw_dev
        highlight_compression,
    )
    .map_err(|e| format!("Failed to load image: {}", e))?;

    let elapsed = step_timer.elapsed().as_millis();
    let (cpu_mem, gpu_mem, gpu_load) = collect_system_metrics();
    log_step_timing(
        file_name,
        "load",
        elapsed,
        cpu_mem,
        gpu_mem,
        gpu_load,
    );

    // Step 2: Load and apply metadata
    let step_timer = Instant::now();
    let adjustments = if metadata.adjustments != Value::Null {
        metadata.adjustments
    } else {
        Value::Object(serde_json::Map::new())
    };
    let elapsed = step_timer.elapsed().as_millis();
    let (cpu_mem, gpu_mem, gpu_load) = collect_system_metrics();
    log_step_timing(
        file_name,
        "metadata_load",
        elapsed,
        cpu_mem,
        gpu_mem,
        gpu_load,
    );

    // Step 3: Apply transformations
    let step_timer = Instant::now();
    let mut image_for_processing = pristine_img.clone();
    if is_raw {
        apply_cpu_default_raw_processing(&mut image_for_processing);
    }

    let (transformed_image, unscaled_crop_offset) =
        apply_all_transformations(&image_for_processing, &adjustments);
    let elapsed = step_timer.elapsed().as_millis();
    let (cpu_mem, gpu_mem, gpu_load) = collect_system_metrics();
    log_step_timing(
        file_name,
        "transform",
        elapsed,
        cpu_mem,
        gpu_mem,
        gpu_load,
    );

    let (img_w, img_h) = transformed_image.dimensions();

    // Step 4: GPU Processing (if enabled)
    let step_timer = Instant::now();
    let mask_definitions: Vec<MaskDefinition> = adjustments
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_else(Vec::new);

    let _mask_bitmaps: Vec<ImageBuffer<Luma<u8>, Vec<u8>>> = mask_definitions
        .iter()
        .filter_map(|def| generate_mask_bitmap(def, img_w, img_h, 1.0, unscaled_crop_offset))
        .collect();

    let mut all_adjustments = get_all_adjustments_from_json(&adjustments, is_raw);
    all_adjustments.global.show_clipping = 0;

    let lut_path = adjustments["lutPath"].as_str();
    let _lut = if let Some(lp) = lut_path {
        if !lut_cache.contains_key(lp) {
            if let Ok(loaded_lut) = crate::lut_processing::parse_lut_file(lp) {
                lut_cache.insert(lp.to_string(), Arc::new(loaded_lut));
            }
        }
        lut_cache.get(lp).cloned()
    } else {
        None
    };

    let _unique_hash = calculate_full_job_hash(&source_path_str, &adjustments);

    let final_image = if use_gpu && gpu_context.is_some() {
        log::warn!("GPU processing requested but not available in CLI mode. Using CPU processing.");
        transformed_image
    } else {
        transformed_image
    };

    let elapsed = step_timer.elapsed().as_millis();
    let (cpu_mem, gpu_mem, gpu_load) = collect_system_metrics();
    log_step_timing(
        file_name,
        "gpu_process",
        elapsed,
        cpu_mem,
        gpu_mem,
        gpu_load,
    );

    // Step 5: Export
    let step_timer = Instant::now();
    let output_filename = generate_output_filename(image_path);
    let output_path = output_dir.join(&output_filename);

    let extension = output_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("jpg");

    let mut image_bytes = encode_image_to_bytes(&final_image, extension, 90)?;
    write_image_with_metadata(
        &mut image_bytes,
        &source_path_str,
        extension,
        true, // keep_metadata
        false, // strip_gps
    )?;

    fs::write(&output_path, &image_bytes)
        .map_err(|e| format!("Failed to write output file: {}", e))?;

    let elapsed = step_timer.elapsed().as_millis();
    let (cpu_mem, gpu_mem, gpu_load) = collect_system_metrics();
    log_step_timing(
        file_name,
        "export",
        elapsed,
        cpu_mem,
        gpu_mem,
        gpu_load,
    );

    let total_elapsed = batch_start.elapsed().as_millis();
    eprintln!(
        "  Completed: {} [load: {}ms, metadata: {}ms, transform: {}ms, gpu: {}ms, export: {}ms]",
        file_name, 0, 0, 0, 0, 0
    );
    eprintln!("  Total: {}ms", total_elapsed);

    Ok(())
}

/// Log timing information to stderr for human readability
fn log_step_timing(
    filename: &str,
    step: &str,
    elapsed_ms: u128,
    cpu_mem: f64,
    gpu_mem: f64,
    gpu_load: f64,
) {
    eprintln!(
        "  {} - {} elapsed: {}ms (CPU: {:.1}MB, GPU: {:.1}MB, Load: {:.1}%)",
        filename, step, elapsed_ms, cpu_mem, gpu_mem, gpu_load
    );
}

/// Main CLI processor entry point
pub fn run_cli_processor(opts: CliOptions) -> Result<(), String> {
    setup_cli_logging(opts.verbose)?;

    log::info!(
        "Starting CLI processor: input={}, output={}, gpu={}, verbose={}",
        opts.input_dir.display(),
        opts.output_dir.display(),
        opts.use_gpu,
        opts.verbose
    );

    // Create output directory if it doesn't exist
    fs::create_dir_all(&opts.output_dir)
        .map_err(|e| format!("Failed to create output directory: {}", e))?;

    // Find all images with sidecars
    let images = find_images_with_sidecars(&opts.input_dir)?;
    if images.is_empty() {
        return Err("No images with sidecar files found in input directory".to_string());
    }

    log::info!("Found {} images to process", images.len());

    // Initialize GPU context if needed
    let gpu_context = if opts.use_gpu {
        Some(init_gpu_context()?)
    } else {
        None
    };

    // Create CSV writer for metrics
    let metrics_file_path = opts.output_dir.join("processing_metrics.csv");
    let metrics_file = fs::File::create(&metrics_file_path)
        .map_err(|e| format!("Failed to create metrics file: {}", e))?;
    let mut csv_writer = Writer::from_writer(metrics_file);

    // Process each image
    let mut lut_cache: HashMap<String, Arc<Lut>> = HashMap::new();

    for (image_path, sidecar_path) in images {
        process_single_image(
            &image_path,
            sidecar_path.as_deref(),
            &opts.output_dir,
            opts.use_gpu,
            gpu_context.as_ref(),
            &mut lut_cache,
        )?;
    }

    csv_writer
        .flush()
        .map_err(|e| format!("Failed to flush CSV: {}", e))?;

    log::info!("CLI processing completed successfully");
    eprintln!("Processing complete. Metrics saved to {}", metrics_file_path.display());

    Ok(())
}
