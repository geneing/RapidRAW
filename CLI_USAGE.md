# RapidRAW CLI Processing Mode

## Overview

RapidRAW now supports a command-line interface (CLI) for batch processing images. This enables:
- Automated image processing workflows
- Unit testing of image processing
- Performance profiling of the rendering pipeline
- Headless processing on servers

## Usage

### Basic Command

```bash
rapidraw process <input_directory> <output_directory> [options]
```

### Options

- `--cpuonly` - Process using CPU only, skip GPU acceleration
- `--verbose` - Enable debug-level logging output

### Examples

#### Standard GPU-accelerated processing
```bash
rapidraw process C:\photos\input C:\photos\output
```

#### CPU-only processing (useful for servers without GPU)
```bash
rapidraw process /mnt/photos/input /mnt/photos/output --cpuonly
```

#### Verbose logging for debugging
```bash
rapidraw process ~/photos/input ~/photos/output --verbose
```

## Input Format

The CLI looks for image files in the input directory that have corresponding `.rrdata` sidecar files. The sidecar files contain the processing settings (adjustments, masks, etc.).

### Expected Structure

```
input_directory/
├── photo1.jpg          (optional - if no .rrdata, default settings used)
├── photo1.jpg.rrdata   (contains adjustments in JSON)
├── photo2.cr3
└── photo2.cr3.rrdata
```

### Sidecar File Format

The `.rrdata` file is a JSON file containing `ImageMetadata`:

```json
{
  "version": 1,
  "rating": 4,
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

## Output

### Processed Images

Output images are saved with a `_processed` suffix:
- Input: `photo.jpg` → Output: `photo_processed.jpg`
- Input: `photo.cr3` → Output: `photo_processed.jpg` (or .png, .tiff based on format)

### Metrics CSV

A `processing_metrics.csv` file is generated with per-step profiling data:

```csv
filename,step_name,elapsed_ms,cpu_memory_mb,gpu_memory_mb,gpu_load_pct
photo.jpg,load,45,125.5,0.0,0.0
photo.jpg,metadata_load,2,125.5,0.0,0.0
photo.jpg,transform,78,130.2,450.0,85.5
photo.jpg,gpu_process,420,135.0,512.0,95.0
photo.jpg,export,65,140.0,450.0,50.0
```

## Processing Steps Tracked

Each image goes through the following steps (all times logged):

1. **load** - Load image file from disk
2. **metadata_load** - Load and parse .rrdata sidecar file
3. **transform** - Apply geometric transformations (crop, rotate, flip)
4. **gpu_process** - Apply GPU-accelerated effects (color grading, curves, etc.)
5. **export** - Encode and write output image

## Error Handling

The CLI uses **fail-fast** error handling:
- Processing stops on the first error
- Error message is printed to stderr
- Exit code is 1 on failure, 0 on success

## Performance Profiling

The CSV output is designed for downstream analysis:

```powershell
# Example: Calculate average time per step
Import-Csv processing_metrics.csv | Group-Object step_name | ForEach-Object {
    $avg = ($_.Group | Measure-Object elapsed_ms -Average).Average
    Write-Host "$($_.Name): ${avg}ms"
}
```

## Examples

### Create a sidecar file for batch processing

```powershell
$metadata = @{
    version = 1
    rating = 0
    adjustments = @{
        exposure = 0.0
        contrast = 1.0
        saturation = 1.0
        orientationSteps = 0
        rotation = 0
        flipHorizontal = $false
        flipVertical = $false
        crop = $null
        masks = @()
    }
    tags = @()
} | ConvertTo-Json

$metadata | Out-File "C:\photos\photo.jpg.rrdata"
```

### Run CLI processing

```bash
cd C:\RapidRAW
.\target\release\RapidRAW.exe process C:\photos\input C:\photos\output --verbose
```

### Analyze metrics

```bash
# Find slowest processing steps
sort -t',' -k3 -rn processing_metrics.csv | head -10
```

## Technical Details

### GPU Context Initialization

- GPU context is initialized once at startup if `--cpuonly` is not specified
- If GPU initialization fails, processing stops with an error
- Use `--cpuonly` flag to force CPU-only processing

### File Discovery

- Recursively scans input directory for supported image formats
- Matches `.rrdata` files to their source images
- Supports virtual copies (e.g., `photo.jpg.a1b2c3.rrdata`)

### Memory Tracking

- CPU memory: Read from `/proc/self/status` on Linux (not available on Windows/macOS)
- GPU memory: Currently not tracked (requires wgpu internals)
- GPU load: Currently set to 0.0 (requires specialized tools)

## Future Enhancements

- [ ] GPU memory tracking
- [ ] GPU load percentage calculation
- [ ] Parallel image processing (currently sequential)
- [ ] Output format selection (currently auto-detected from extension)
- [ ] Batch export template variables support
- [ ] Streaming metrics output
