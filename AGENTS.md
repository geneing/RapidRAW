# AGENTS.md

## Executive Summary

**RapidRAW** is a cross-platform, GPU-accelerated, non-destructive RAW image editor targeting photographers who value speed, simplicity, and a modern workflow. It is implemented as a hybrid desktop application using a Rust backend (via Tauri) and a React/TypeScript frontend. The project is actively developed and supports Windows, macOS, and Linux.

- **Purpose:** High-performance, user-friendly RAW photo editing with advanced masking, adjustment, and export features.
- **Primary Entry Points:**
  - Backend: `src-tauri/src/main.rs` (Rust, Tauri)
  - Frontend: `src/main.tsx` and `src/App.tsx` (React)
- **Key Dependencies:**
  - Rust: `tauri`, `wgpu`, `image`, `rayon`, `serde`, `rawler`, `ort`, `tokio`, `reqwest`, `mimalloc`
  - JS: `react`, `@tauri-apps/api`, `@tauri-apps/plugin-*`, `framer-motion`, `konva`, `@dnd-kit/core`, `lucide-react`, `@clerk/clerk-react`
- **Supported Platforms:** Windows, macOS, Linux

## High-Level Architecture Diagram (Textual)

- **Frontend (React/TypeScript):**
  - UI components (adjustments, panels, modals)
  - State management (hooks, context)
  - Communicates with backend via Tauri IPC
- **Backend (Rust/Tauri):**
  - Image loading, processing, and export (`image_processing.rs`, `raw_processing.rs`, `gpu_processing.rs`, `image_loader.rs`)
  - Mask and adjustment management (`mask_generation.rs`, `src/utils/adjustments.tsx`)
  - AI/ONNX-based features (`ai_processing.rs`, `ai_connector.rs`, `ort`)
  - File, metadata, and tagging (`file_management.rs`, `formats.rs`, `exif_processing.rs`, `tagging.rs`)
  - Specialty pipelines (denoise, inpaint, panorama, lens, negative) (`denoising.rs`, `inpainting.rs`, `panorama_stitching.rs`, `lens_correction.rs`, `negative_conversion.rs`)
- **Data Flow:**
  - User actions in UI -> Tauri IPC -> Rust backend for processing -> Results returned to UI for display
- **External Services:**
  - Optional: AI model downloads and remote assets via HTTP

## Risk Snapshot: Top 5 Maintainability and Security Risks

1. **Large, Monolithic Files:** Key files (e.g., `App.tsx`, `main.rs`, `image_processing.rs`) are very large, increasing cognitive load and risk of bugs.
2. **Complex Data Serialization:** Heavy use of `serde_json::Value` and dynamic structures for adjustments and masks can lead to runtime errors and weak type safety.
3. **GPU and Native Code Complexity:** Use of `wgpu`, ONNX, and custom GPU code increases risk of subtle bugs, platform-specific issues, and security vulnerabilities.
4. **Rapid API Surface Growth:** Frequent expansion of core modules (AI, panorama, denoise, inpainting, lens, negative conversion) may outpace documentation and test coverage.
5. **Dependency Surface:** Many dependencies (Rust and JS) increase supply chain and update risks; some (e.g., ONNX, wgpu) are complex and security-sensitive.

## Architecture and Modules

### Component Map

- **src-tauri/src/**
  - `main.rs`: Application entry, module orchestration, Tauri commands
  - `image_processing.rs`: Core image operations, metadata, adjustment application
  - `gpu_processing.rs`: GPU-accelerated processing (via wgpu)
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

### Data Model

- **Image Data:**
  - Rust: `DynamicImage`, `RgbaImage`, `Rgb32FImage` (from the `image` crate)
  - JS: Image data is not directly manipulated; adjustments and masks are sent to the backend for processing
- **Metadata:**
  - `ImageMetadata` struct (Rust): version, rating, adjustments (as JSON), tags
- **Masks:**
  - `MaskDefinition` (Rust/TS): id, name, visible, invert, opacity, adjustments, sub_masks
  - `SubMask`: id, type, visible, mode (additive or subtractive), parameters
- **Adjustments:**
  - Enum-based (TS): `BasicAdjustment`, `ColorAdjustment`, `DetailsAdjustment`, etc.
  - Passed as partial objects for composability

---

### Objects for Image Data

**Rust Backend:**
- The primary image data objects are from the `image` crate:
  - `DynamicImage`: Used for generic image operations and conversions between formats.
  - `RgbaImage`: Stores 8-bit RGBA pixel data, used for display and export.
  - `Rgb32FImage`: Stores 32-bit floating-point RGB data, used for high-precision processing and GPU operations.
- Additional types:
  - `GrayImage`, `Luma`, and custom buffer types for masks and intermediate results.
- GPU processing uses custom wrappers and `wgpu` textures for efficient parallel computation.
- RAW decoding uses `RawImage` and related types from the `rawler` crate, with conversion to `DynamicImage` for further processing.

**Frontend (TypeScript):**
- Image data is not directly manipulated in JS/TS. Instead, the frontend manages references (file paths, IDs) and adjustment and mask objects, which are serialized and sent to the backend for processing.
- Thumbnails and previews are handled as binary blobs or base64-encoded images received from the backend.

---

### Color Science Implementation

RapidRAW implements a modern color science pipeline inspired by Blender's AgX and darktable:

- **Tone Mapping:**
  - Supports both "Basic" and "AgX" tone mappers, selectable in the UI.
  - AgX is a filmic-like tone mapping curve designed for perceptual accuracy and pleasing rolloff in highlights and shadows.
  - Tone mapping is implemented in both CPU (Rust) and GPU (WGSL shaders) code paths.

- **Color Adjustments:**
  - Modular adjustment stack includes exposure, contrast, white balance, HSL, color grading, and LUTs.
  - Color operations are performed in linear RGB or perceptual spaces as appropriate.
  - LUT support allows for creative looks and film emulation.

- **Color Management:**
  - Internal processing is performed in high bit-depth (32F) linear RGB.
  - Output is converted to sRGB or display-referred color spaces for export and preview.
  - EXIF and metadata parsing (via `kamadak-exif`, `little_exif`) is used to extract white balance and camera color data.

- **Implementation Details:**
  - Color science logic is split between Rust (core processing, metadata) and GPU shaders (real-time preview, fast transforms).
  - The frontend exposes color controls and visualizations, but all color math is performed in the backend for accuracy and performance.

---

### Testing Infrastructure

**Rust Backend:**
- Unit and integration tests are defined in Rust modules using the standard `#[cfg(test)]` and `#[test]` attributes.
- Key modules (e.g., `image_processing.rs`, `raw_processing.rs`, `mask_generation.rs`) include test functions for core algorithms and edge cases.
- Tests cover image loading, adjustment application, mask logic, and file I/O.
- Build and release workflows exist, but CI coverage for tests is not confirmed here.

**Frontend (TypeScript):**
- No explicit test files or frameworks (e.g., Jest, React Testing Library) were observed in the top-level `package.json` scripts.
- Testing is likely manual or ad hoc, with reliance on type safety and runtime validation.

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
  - GPU context management is handled via a global context, with device and queue reuse to avoid costly reinitialization.

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
  - `src/App.tsx` (lines 1-120)

- **Dependency Map:**
  - Rust: See `src-tauri/Cargo.toml` for runtime dependencies (Tauri 2.9, wgpu 28, image 0.25, ort 2.0.0-rc, reqwest 0.12, tokio 1.x, etc.)
  - JS: See `package.json` for runtime and dev dependencies (React 19, Tauri API 2.9, framer-motion, konva, Tailwind, Vite 7, etc.)

---

*Updated by Codex on 2026-02-07 based on local repository state.*
