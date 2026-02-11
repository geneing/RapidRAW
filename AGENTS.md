# AGENTS.md

## Executive Summary

**RapidRAW** is a cross-platform, GPU-accelerated, non-destructive RAW image editor targeting photographers who value speed, simplicity, and a modern workflow. It is implemented as a hybrid desktop application using a Rust backend (via Tauri) and a React/TypeScript frontend. The project is actively developed and supports Windows, macOS, and Linux.

- **Purpose:** High-performance, user-friendly RAW photo editing with advanced masking, adjustment, and export features.
- **Primary Entry Points:**
  - Backend: `src-tauri/src/main.rs` (Rust, Tauri)
  - Frontend: `src/main.tsx` and `src/App.tsx` (React)
- **Key Dependencies:**
  - Rust: `tauri`, `wgpu`, `cubecl`, `image`, `rayon`, `serde`, `rawler`, `ort`, `tokio`, `reqwest`, `mimalloc`
  - JS: `react`, `@tauri-apps/api`, `@tauri-apps/plugin-*`, `framer-motion`, `konva`, `@dnd-kit/core`, `lucide-react`, `@clerk/clerk-react`
- **Supported Platforms:** Windows, macOS, Linux

## High-Level Architecture Diagram (Textual)

- **Frontend (React/TypeScript):**
  - UI components (adjustments, panels, modals)
  - State management (hooks, context)
  - Communicates with backend via Tauri IPC
- **Backend (Rust/Tauri):**
  - Image loading, processing, and export (`image_processing.rs`, `raw_processing.rs`, `gpu_processing.rs`, `cubecl_processing.rs`, `image_loader.rs`)
  - Mask and adjustment management (`mask_generation.rs`, `src/utils/adjustments.tsx`)
  - AI/ONNX-based features (`ai_processing.rs`, `ai_connector.rs`, `ort`)
  - File, metadata, and tagging (`file_management.rs`, `formats.rs`, `exif_processing.rs`, `tagging.rs`)
  - Specialty pipelines (denoise, inpaint, panorama, lens, negative) (`denoising.rs`, `inpainting.rs`, `panorama_stitching.rs`, `lens_correction.rs`, `negative_conversion.rs`)
- **Data Flow:**
  - User actions in UI -> Tauri IPC -> Rust backend for processing -> Results returned to UI for display
- **External Services:**
  - Optional: AI model downloads and remote assets via HTTP

## Architecture and Modules

### Component Map

- **src-tauri/src/**
  - `main.rs`: Application entry, module orchestration, Tauri commands
  - `image_processing.rs`: Core image operations, metadata, adjustment application
  - `gpu_processing.rs`: GPU-accelerated processing (via wgpu)
  - `cubecl_processing.rs`: Experimental/benchmark CubeCL GPU processing path, comparison, and fallback logic
  - `raw_processing.rs`: RAW file decoding (via `rawler`)
  - `image_loader.rs`: Image loading and compositing utilities
  - `mask_generation.rs`: Mask and selection logic, supports additive and subtractive submasks
  - `ai_processing.rs`, `ai_connector.rs`: AI features (ONNX models, mask generation)
  - `denoising.rs`, `inpainting.rs`: Heavy compute pipelines
  - `panorama_stitching.rs`: Panorama processing
  - `lens_correction.rs`, `negative_conversion.rs`: Specialty transforms
  - `lut_processing.rs`: LUT management
  - `preset_converter.rs`: Preset ingestion and conversion
  - `exif_processing.rs`, `file_management.rs`, `formats.rs`: File I/O and metadata
  - `tagging.rs`: Tagging and classification
  - `culling.rs`: Photo culling logic
- **src/components/adjustments/**
  - `Basic.tsx`, `Color.tsx`, `Curves.tsx`, `Details.tsx`, `Effects.tsx`: Modular adjustment panels
- **src/components/panel/**
  - `Editor`, `MainLibrary`, `CommunityPage`, right-side panels for crop, presets, masks, AI, export, and metadata
- **src/utils/adjustments.tsx**
  - Central adjustment and mask data model, enums for adjustment types, mask modes, and helpers

### Modular Structure of Filters

- Each adjustment or filter is a React component (frontend) and a corresponding Rust function (backend).
- Adjustments are passed as JSON objects (`serde_json::Value`) between frontend and backend, allowing flexible stacking and masking.
- Masks are defined as composable structures (`MaskDefinition`, `SubMask`) with additive and subtractive logic, supporting complex selections.


### Testing Infrastructure

**Rust Backend:**
- Unit and integration tests are defined in Rust modules using the standard `#[cfg(test)]` and `#[test]` attributes.
- Key modules (e.g., `image_processing.rs`, `raw_processing.rs`, `mask_generation.rs`) include test functions for core algorithms and edge cases.
- Tests cover image loading, adjustment application, mask logic, and file I/O.
- Build and release workflows exist, but CI coverage for tests is not confirmed here.

**Summary:**
- The backend has basic automated test coverage for core logic, but the frontend lacks formal automated testing.

---

### Performance and Processing Optimizations

RapidRAW employs several large-scale architectural optimizations to maximize image processing speed and responsiveness:

- **Multithreading (CPU Parallelism):**
  - The Rust backend uses the `rayon` crate to parallelize CPU-bound image operations across all available cores.
  - Batch operations (e.g., thumbnail generation, export, batch adjustments) are distributed across threads for high throughput.

- **GPU Acceleration:**
  - The core image pipeline leverages `wgpu` for GPU-accelerated processing.
  - Custom WGSL shaders are used for fast, parallelizable operations (e.g., exposure, curves, LUTs, masking).
  - A parallel CubeCL compute path exists for shader-porting/benchmark work and can be enabled for comparison or preferred output.
  - GPU context management is handled via a global context, with device and queue reuse to avoid costly reinitialization.

### CubeCL Implementation (Current State)

- **Status:**
  - CubeCL is integrated as an additional GPU path in the Rust backend, focused on parity testing and performance benchmarking against WGSL.
  - Implementation lives in `src-tauri/src/cubecl_processing.rs` and is invoked from `src-tauri/src/gpu_processing.rs`.
- **Runtime Modes:**
  - `RAPIDRAW_CUBECL_MODE=off|benchmark|cubecl` controls behavior.
  - `off`: WGSL-only execution.
  - `benchmark`: Run WGSL + CubeCL, log timings and pixel diff, keep WGSL output.
  - `cubecl`: Prefer CubeCL output when CubeCL successfully executes.
- **Timing and Comparison Logging:**
  - Compare logs include WGSL duration, CubeCL total duration, CubeCL stage durations (threshold/blur/main), and mismatch metrics (count, max diff, mean diff).
  - Tolerance is controlled by `RAPIDRAW_CUBECL_MATCH_TOLERANCE`.
- **Parity and Fallback:**
  - Current CubeCL coverage includes a subset of blur and flare processing kernels plus simplified main compositing.
  - For unsupported adjustment sets, CubeCL returns WGSL fallback output (when provided) and records fallback reason; this preserves correctness during incremental porting.
  - Full parity with `shader.wgsl` (masks, LUT, AgX, curves, HSL, color grading, vignette/grain, etc.) is not complete.
- **Testing:**
  - Unit tests verify identity tolerance matching and fallback behavior in `gpu_processing.rs`.
  - CubeCL path is currently intended for controlled benchmarking/verification rather than default production rendering.

### CubeCL Unsupported Controls and TODO Playbooks

Below is the actionable implementation backlog for CubeCL parity with WGSL. Each item includes a step-by-step plan suitable for another LLM.


4. **Curves (global + per-mask, luma + RGB channels)**
   - **Current state:** Not implemented. CubeCL bails out when any curve count is non-zero.
   - **TODO plan:**
     - Port point interpolation helpers:
       - `interpolate_cubic_hermite`
       - `apply_curve`
       - `is_default_curve`
       - `apply_all_curves`.
     - Match WGSL default-curve optimization behavior.
     - Apply global curves after tonemap and before LUT.
     - Apply per-mask curves after global curves with mask influence blending.
     - Remove/relax curve gate in `unsupported_reason`.
     - Add tests for:
       - Identity curve.
       - S-curve contrast.
       - RGB-only split tone curves.

5. **HSL panel**
   - **Current state:** Not implemented in CubeCL pipeline; currently should be treated as unsupported.
   - **TODO plan:**
     - Add explicit HSL non-default detection in `unsupported_reason` first (safety).
     - Port HSL helpers from WGSL:
       - `rgb_to_hsv`, `hsv_to_rgb`, `get_raw_hsl_influence`, range definitions.
     - Port `apply_hsl_panel` exactly, including gray short-circuit and influence normalization.
     - Apply HSL at the same order inside global and mask adjustment stacks.
     - Remove the HSL gate after implementation and validation.
     - Add tests per hue range (reds/oranges/…/magentas) with tolerance checks.

6. **Color grading (shadows/midtones/highlights + blending + balance)**
   - **Current state:** Not implemented. CubeCL bails out when color grading values are non-default.
   - **TODO plan:**
     - Port `apply_color_grading` from WGSL.
     - Preserve crossover, feather, and mask math exactly.
     - Reuse HSV helper and tint application behavior.
     - Integrate into global and mask adjustment functions in the same stage order as WGSL.
     - Remove/relax color grading gate in `unsupported_reason`.
     - Add tests for isolated shadows/midtones/highlights and balance extremes.

7. **Color calibration**
   - **Current state:** Not implemented. CubeCL bails out when calibration values are non-default.
   - **TODO plan:**
     - Port `apply_color_calibration` (hue matrix, channel masks, saturation mix, shadows tint).
     - Match numerical safeguards (`max`, epsilon usage).
     - Insert at the same pipeline point as WGSL (before HSL/color grading/creative color).
     - Remove/relax calibration gate in `unsupported_reason`.
     - Add tests for each channel’s hue/saturation controls and shadows tint.

8. **Local contrast blur path (sharpness, clarity, structure, centre local contrast)**
   - **Current state:** Not implemented. CubeCL bails out if `sharpness/clarity/structure` are non-zero.
   - **TODO plan:**
     - Port `apply_local_contrast`, `apply_centre_local_contrast`, and `apply_centre_tonal_and_color`.
     - Implement separate blur buffers equivalent to WGSL:
       - sharpness blur radius scaling
       - clarity blur radius scaling
       - structure blur radius scaling.
     - Match raw vs non-raw handling and mode-specific dampening.
     - Integrate centre control into tonal/color path and local contrast path.
     - Remove/relax local contrast gate in `unsupported_reason`.
     - Add tests for each control independently and combined.

9. **Advanced effects bundle**
   - **Current state:** Grouped as unsupported (`dehaze`, `glow`, `halation`, `vignette`, `grain`, chromatic aberration).
   - **TODO plan (common):**
     - Split the current single gate into per-effect gates for incremental rollout.
     - Port one effect at a time and validate before enabling the next.
   - **Per-effect step plan:**
     - `dehaze`:
       - Port `apply_dehaze`.
       - Validate with haze-heavy images and negative values.
     - `glow`:
       - Port `apply_glow_bloom`.
       - Ensure blurred source matches WGSL source texture and stage ordering.
     - `halation`:
       - Port `apply_halation`.
       - Validate highlight-only behavior and red fringe tint response.
     - `vignette`:
       - Port vignette geometry math (midpoint, roundness, feather, aspect compensation).
       - Validate darken and brighten branches (`vignette_amount < 0` vs `> 0`).
     - `grain`:
       - Port `hash`, `gradient_noise`, and grain blend logic.
       - Ensure deterministic output for stable tests.
     - `chromatic aberration`:
       - Port `apply_ca_correction` sampling offsets and channel recombine.
       - Validate edge cases near borders.

10. **Main tonal/color controls parity (brightness/contrast/highlights/shadows/whites/blacks/temperature/tint/saturation/vibrance)**
   - **Current state:** Partially implemented; currently gated as unsupported for non-zero values.
   - **TODO plan:**
     - Port WGSL equivalents exactly:
       - `apply_white_balance`
       - `apply_filmic_exposure`
       - `apply_tonal_adjustments`
       - `apply_highlights_adjustment`
       - `apply_creative_color`.
     - Verify operation order in global and mask paths.
     - Validate non-raw and raw branches separately.
     - Remove/relax this gate only after all above controls pass parity thresholds.
     - Add parameter sweep tests over positive/negative ranges for each control.

11. **Clipping overlay**
   - **Current state:** Not implemented. CubeCL bails out when `show_clipping != 0`.
   - **TODO plan:**
     - Port highlight/shadow clipping logic and warning colors from WGSL.
     - Apply at final RGB stage after vignette/grain and before final store (matching WGSL).
     - Remove/relax clipping gate in `unsupported_reason`.
     - Add tests for near-threshold edge values.

12. **Parity roll-out and safety process (cross-cutting TODO)**
   - Keep fallback enabled by default while any unsupported gate remains.
   - Introduce one feature flag per control to allow staged enablement.
   - After each control:
     - run WGSL-vs-CubeCL diff tests on small synthetic images and real RAW-derived previews,
     - log mismatch rate and max/mean diffs,
     - only then relax the corresponding gate.
   - Add/maintain a parity dashboard note in logs (per control enabled/disabled).

- **Hybrid Processing Pipeline:**
  - The architecture allows dynamic selection between CPU and GPU code paths depending on operation type, image size, and hardware capabilities.
  - Fallbacks to CPU are provided for unsupported hardware or headless environments.

- **Memory and Data Flow Optimizations:**
  - Image data is processed in high-precision (32F) buffers on the GPU, with conversion to lower-precision formats only for display and export.
  - Intermediate results are cached and reused where possible to avoid redundant computation.
  - Mask and adjustment data are passed as compact JSON structures, reducing IPC overhead between frontend and backend.

- **Asynchronous and Non-blocking Design:**
  - Long-running operations (e.g., AI-based masking, batch exports) are executed asynchronously, with progress updates sent to the frontend to keep the UI responsive.
  - Tokio and async Rust are used for I/O-bound and network operations (e.g., model downloads, file I/O).

These strategies collectively enable RapidRAW to deliver real-time feedback and high throughput, even on large RAW files and complex adjustment stacks.

### Format and Metadata Passing

- Adjustments and masks are serialized as JSON (`serde_json::Value`) for flexibility
- Image data is passed as file paths or binary buffers, not directly through IPC
- Metadata (EXIF, tags) is extracted via Rust (`kamadak-exif`, `little_exif`)

### Mask and Modifier Implementation

- Masks are composable, with submasks supporting additive and subtractive logic
- Each mask can have its own set of adjustments, enabling local edits
- Mask logic is implemented in both Rust (processing) and TS (UI and model)

## Methodology and Coverage

- **Directories by Size:**
  - `src-tauri/` (7551.3 MB): Rust backend, main logic, models
  - `node_modules/` (169 MB): JS dependencies
  - `dist/` (8 MB): Frontend build output
  - `public/` (6.7 MB): Assets
  - `src/` (1 MB): React frontend
  - `packaging/` (0.8 MB): Packaging assets
- **Recently Modified (git status):**
  - `src-tauri/Cargo.toml`
  - `src-tauri/rawler` (submodule or nested repo)
- **Files Examined:**
  - `src-tauri/Cargo.toml`
  - `package.json`
  - `src-tauri/src/main.rs` (lines 1-120)
  - `src-tauri/src/gpu_processing.rs` (lines 1-160)
  - `src-tauri/src/cubecl_processing.rs`
  - `src/App.tsx` (lines 1-120)

- **Dependency Map:**
  - Rust: See `src-tauri/Cargo.toml` for runtime dependencies (Tauri 2.9, wgpu 28, image 0.25, ort 2.0.0-rc, reqwest 0.12, tokio 1.x, etc.)
  - JS: See `package.json` for runtime and dev dependencies (React 19, Tauri API 2.9, framer-motion, konva, Tailwind, Vite 7, etc.)

---

*Updated by Codex on 2026-02-09 based on local repository state.*
