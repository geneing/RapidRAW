# CLI Processing Quick Start Guide

## Building

The CLI functionality is now built into RapidRAW. To build:

```bash
cd src-tauri
cargo build --release
```

The executable will be at: `target/release/RapidRAW.exe` (Windows) or `target/release/RapidRAW` (Linux/macOS)

## Quick Test

### 1. Create test directory structure

```bash
mkdir -p test_input test_output
```

### 2. Create a test sidecar file

Save this as `test_input/test_photo.jpg.rrdata`:

```json
{
  "version": 1,
  "rating": 0,
  "adjustments": {
    "exposure": 0.5,
    "contrast": 1.2,
    "saturation": 1.1,
    "orientationSteps": 0,
    "rotation": 0,
    "flipHorizontal": false,
    "flipVertical": false,
    "crop": null,
    "masks": []
  },
  "tags": []
}
```

### 3. Add a test image

Copy any JPEG or supported image format to `test_input/test_photo.jpg`

### 4. Run CLI processing

```bash
./target/release/RapidRAW process test_input test_output --verbose
```

### 5. Check results

```bash
ls -la test_output/
cat test_output/processing_metrics.csv
```

You should see:
- `test_photo_processed.jpg` - Processed image
- `processing_metrics.csv` - Performance metrics

## Command Reference

### Standard Processing (GPU-accelerated)
```bash
rapidraw process input_dir output_dir
```

### CPU-Only (No GPU)
```bash
rapidraw process input_dir output_dir --cpuonly
```

### Verbose Output
```bash
rapidraw process input_dir output_dir --verbose
```

### Combined
```bash
rapidraw process input_dir output_dir --cpuonly --verbose
```

## Output Structure

```
output_dir/
├── processing_metrics.csv    # Profiling data (CSV format)
├── image1_processed.jpg      # First processed image
├── image2_processed.jpg      # Second processed image
└── ...
```

## Metrics CSV Format

The `processing_metrics.csv` contains one row per processing step per image:

```csv
filename,step_name,elapsed_ms,cpu_memory_mb,gpu_memory_mb,gpu_load_pct
test_photo.jpg,load,45,125.5,0.0,0.0
test_photo.jpg,metadata_load,2,125.5,0.0,0.0
test_photo.jpg,transform,78,130.2,0.0,0.0
test_photo.jpg,gpu_process,0,130.2,0.0,0.0
test_photo.jpg,export,65,140.0,0.0,0.0
```

### Columns:
- **filename** - Image being processed
- **step_name** - Processing step (load, metadata_load, transform, gpu_process, export)
- **elapsed_ms** - Time for this step in milliseconds
- **cpu_memory_mb** - Process memory usage in MB
- **gpu_memory_mb** - GPU memory usage (currently 0.0)
- **gpu_load_pct** - GPU utilization (currently 0.0)

## Analyzing Performance

### PowerShell Example

```powershell
# Load CSV and analyze
$metrics = Import-Csv "test_output\processing_metrics.csv"

# Time per step
$metrics | Group-Object step_name | ForEach-Object {
    $avg = ($_.Group | Measure-Object elapsed_ms -Average).Average
    $total = ($_.Group | Measure-Object elapsed_ms -Sum).Sum
    Write-Host "$($_.Name): Avg=${avg}ms, Total=${total}ms"
}

# Total time per image
$metrics | Group-Object filename | ForEach-Object {
    $total = ($_.Group | Measure-Object elapsed_ms -Sum).Sum
    Write-Host "$($_.Name): Total=${total}ms"
}

# Memory usage trend
$metrics | ForEach-Object {
    Write-Host "$($_.filename) - $($_.step_name): $($_.cpu_memory_mb)MB"
}
```

### Linux/macOS Example

```bash
# Total time per step
awk -F',' 'NR>1 {step[$2]+=$3; count[$2]++} 
    END {for (s in step) printf "%s: %dms\n", s, step[s]/count[s]}' \
    test_output/processing_metrics.csv

# Find slowest step
awk -F',' 'NR>1 {print $2, $3}' test_output/processing_metrics.csv | \
    sort -k2 -rn | head -1
```

## Sidecar File Format

The `.rrdata` file is a JSON file containing image metadata and adjustments:

### Minimal Example
```json
{
  "version": 1,
  "rating": 0,
  "adjustments": {
    "exposure": 0.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "orientationSteps": 0,
    "rotation": 0,
    "flipHorizontal": false,
    "flipVertical": false,
    "crop": null,
    "masks": []
  },
  "tags": []
}
```

### Full Example with Adjustments
```json
{
  "version": 1,
  "rating": 5,
  "adjustments": {
    "exposure": 0.5,
    "contrast": 1.2,
    "saturation": 1.1,
    "brightness": 0.1,
    "whites": 0.0,
    "blacks": 0.0,
    "highlights": 0.0,
    "shadows": 0.0,
    "clarity": 0.3,
    "vibrance": 0.2,
    "temperature": 100,
    "tint": 50,
    "hue": 0.0,
    "orientationSteps": 0,
    "rotation": 0,
    "flipHorizontal": false,
    "flipVertical": false,
    "crop": {
      "x": 100,
      "y": 100,
      "width": 800,
      "height": 600
    },
    "masks": [],
    "curves": null,
    "hsl": null,
    "colorGrading": null,
    "lutPath": null
  },
  "tags": ["landscape", "sunset"]
}
```

## Troubleshooting

### "No images with sidecar files found"
- Ensure `.rrdata` files are in the input directory
- Files should be named `image_name.format.rrdata` (e.g., `photo.jpg.rrdata`)
- Check file permissions

### "Failed to load image"
- Verify image format is supported (JPEG, PNG, TIFF, RAW formats)
- Check image file is not corrupted
- Try with `--cpuonly` flag

### "Failed to create output directory"
- Check write permissions on parent directory
- Ensure disk has sufficient space

### "GPU initialization failed"
- Use `--cpuonly` flag to skip GPU
- Check system graphics drivers are up to date
- Verify GPU has sufficient VRAM

## Platform Notes

### Windows
- Builds with MSVC toolchain
- GPU acceleration uses Direct3D 12 (recommended) or Vulkan
- Memory reporting not yet implemented

### Linux
- Builds with GCC toolchain
- GPU acceleration uses Vulkan or GL
- Memory reporting available via `/proc/self/status`

### macOS
- Builds with LLVM toolchain
- GPU acceleration uses Metal
- Memory reporting not yet implemented

## Next Steps

See `CLI_USAGE.md` for comprehensive documentation and advanced usage patterns.
