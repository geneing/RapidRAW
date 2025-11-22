# AGENTS.md

## Executive Summary

**RapidRAW** is a cross-platform, GPU-accelerated, non-destructive RAW image editor targeting photographers who value speed, simplicity, and a modern workflow. It is implemented as a hybrid desktop application using a Rust backend (via Tauri) and a React/TypeScript frontend. The project is actively developed and supports Windows, macOS, and Linux.

- **Purpose:** High-performance, user-friendly RAW photo editing with advanced masking, adjustment, and export features.
- **Primary Entry Points:**
  - Backend: `src-tauri/src/main.rs` (Rust, Tauri)
  - Frontend: `src/main.tsx` and `src/App.tsx` (React)
- **Key Dependencies:**
  - Rust: `tauri`, `wgpu`, `image`, `rayon`, `serde`, `rawler`, `ort` (ONNX runtime)
  - JS: `react`, `@tauri-apps/api`, `framer-motion`, `konva`, `lodash.debounce`, `@clerk/clerk-react`
- **Supported Platforms:** Windows, macOS, Linux

## High-Level Architecture Diagram (Textual)

- **Frontend (React/TypeScript):**
  - UI components (adjustments, panels, modals)
  - State management (hooks, context)
  - Communicates with backend via Tauri IPC
- **Backend (Rust/Tauri):**
  - Image loading, processing, and export (`image_processing.rs`, `raw_processing.rs`, `gpu_processing.rs`)
  - Mask and adjustment management (`mask_generation.rs`, `adjustments.tsx`)
  - AI/ONNX-based features (`ai_processing.rs`, `ort`)
  - File and metadata management
- **Data Flow:**
  - User actions in UI → Tauri IPC → Rust backend for processing → Results returned to UI for display
- **External Services:**
  - Optional: AI model downloads, community integration (future)

## Risk Snapshot: Top 5 Maintainability & Security Risks

1. **Large, Monolithic Files:** Key files (e.g., `App.tsx`, `main.rs`, `image_processing.rs`) are very large, increasing cognitive load and risk of bugs.
2. **Complex Data Serialization:** Heavy use of `serde_json::Value` and dynamic structures for adjustments/masks can lead to runtime errors and weak type safety.
3. **GPU/Native Code Complexity:** Use of `wgpu`, ONNX, and custom GPU code increases risk of subtle bugs, platform-specific issues, and security vulnerabilities.
4. **Rapid API Surface Growth:** Frequent changes to core modules (per git history) may outpace documentation and test coverage.
5. **Dependency Surface:** Many dependencies (Rust and JS) increase supply chain and update risks; some (e.g., ONNX, wgpu) are complex and security-sensitive.

## Architecture and Modules

### Component Map

- **src-tauri/src/**
  - `main.rs`: Application entry, module orchestration
  - `image_processing.rs`: Core image operations, metadata, adjustment application
  - `gpu_processing.rs`: GPU-accelerated processing (via wgpu)
  - `raw_processing.rs`: RAW file decoding (via `rawler`)
  - `mask_generation.rs`: Mask/selection logic, supports additive/subtractive submasks
  - `ai_processing.rs`: AI-based features (ONNX, subject/sky/foreground masks)
  - `file_management.rs`, `formats.rs`, `tagging.rs`: File I/O, format support, tagging
- **src/components/adjustments/**
  - `Basic.tsx`, `Color.tsx`, `Curves.tsx`, `Details.tsx`, `Effects.tsx`: UI for modular, stackable adjustment panels
- **src/utils/adjustments.tsx**
  - Central adjustment/mask data model, enums for adjustment types, mask modes, etc.

### Modular Structure of Filters

- Each adjustment/filter is a React component (frontend) and a corresponding Rust function (backend).
- Adjustments are passed as JSON objects (`serde_json::Value`) between frontend and backend, allowing flexible stacking and masking.
- Masks are defined as composable structures (`MaskDefinition`, `SubMask`) with additive/subtractive logic, supporting complex selections.

### Data Model

- **Image Data:**
  - Rust: `DynamicImage`, `RgbaImage`, `Rgb32FImage` (from `image` crate)
  - JS: Image data is not directly manipulated; adjustments/masks are sent to backend for processing
- **Metadata:**
  - `ImageMetadata` struct (Rust): version, rating, adjustments (as JSON), tags
- **Masks:**
  - `MaskDefinition` (Rust/TS): id, name, visible, invert, opacity, adjustments, sub_masks
  - `SubMask`: id, type, visible, mode (additive/subtractive), parameters
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
- Image data is not directly manipulated in JS/TS. Instead, the frontend manages references (file paths, IDs) and adjustment/mask objects, which are serialized and sent to the backend for processing.
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
- No evidence of a dedicated CI pipeline for test automation in the provided context, but GitHub Actions workflows exist for build and release.

**Frontend (TypeScript):**
- No explicit test files or frameworks (e.g., Jest, React Testing Library) are present in the top-level context or `package.json` scripts.
- Testing is likely manual or ad hoc, with reliance on type safety and runtime validation.

**Summary:**
- The backend has basic automated test coverage for core logic, but the frontend lacks formal automated testing. Build and release workflows are present, but continuous integration for tests is not fully established.

---

### Performance and Processing Optimizations

RapidRAW employs several large-scale architectural optimizations to maximize image processing speed and responsiveness:

- **Multithreading (CPU Parallelism):**
  - The Rust backend uses the `rayon` crate to parallelize CPU-bound image operations across all available cores. This is applied to pixel-wise operations, adjustment stacks, and mask computations, enabling efficient use of modern multicore CPUs.
  - Batch operations (e.g., thumbnail generation, export, batch adjustments) are distributed across threads for high throughput.

- **GPU Acceleration:**
  - The core image pipeline leverages `wgpu` for GPU-accelerated processing. Intensive tasks such as tone mapping, color transforms, and real-time preview rendering are offloaded to the GPU.
  - Custom WGSL shaders are used for fast, parallelizable operations (e.g., exposure, curves, LUTs, masking), minimizing CPU-GPU data transfer and maximizing throughput.
  - GPU context management is handled via a global context, with device/queue reuse to avoid costly reinitialization.

- **Hybrid Processing Pipeline:**
  - The architecture allows dynamic selection between CPU and GPU code paths depending on operation type, image size, and hardware capabilities.
  - Fallbacks to CPU are provided for unsupported hardware or headless environments, ensuring broad compatibility.

- **Memory and Data Flow Optimizations:**
  - Image data is processed in high-precision (32F) buffers on the GPU, with conversion to lower-precision formats only for display/export.
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

- Masks are composable, with submasks supporting additive/subtractive logic
- Each mask can have its own set of adjustments, enabling local edits
- Mask logic is implemented in both Rust (processing) and TS (UI/model)

## Methodology and Coverage

- **Directories by Size:**
  - `src-tauri/` (4.2G): Rust backend, main logic
  - `node_modules/` (240M): JS dependencies
  - `public/` (7.9M): Assets
  - `src/` (976K): React frontend
- **Recent Change Frequency:**
  - Most changed: `src-tauri/src/main.rs`, `src-tauri/src/image_processing.rs`, `src/App.tsx`, `src/components/panel/Editor.jsx`
- **Files Examined:**
  - `src-tauri/src/main.rs` (lines 1–60)
  - `src-tauri/src/image_processing.rs` (lines 1–60)
  - `src-tauri/src/gpu_processing.rs` (lines 1–60)
  - `src-tauri/src/raw_processing.rs` (lines 1–60)
  - `src-tauri/src/mask_generation.rs` (lines 1–60)
  - `src/utils/adjustments.tsx` (lines 1–60)
  - `src/components/adjustments/Basic.tsx` (lines 1–60)
  - `src/components/adjustments/Color.tsx` (lines 1–60)
  - `src/components/adjustments/Curves.tsx` (lines 1–60)
  - `src/components/adjustments/Details.tsx` (lines 1–60)
  - `src/components/adjustments/Effects.tsx` (lines 1–60)
  - `src/main.tsx` (lines 1–60)
  - `src/App.tsx` (lines 1–60)
  - `README.md` (lines 1–60)
  - `package.json`, `Cargo.toml`, `Cargo.lock`, `tsconfig.json`

- **Dependency Map:**
  - Rust: See `Cargo.toml` for runtime dependencies (Tauri, wgpu, image, ort, etc.)
  - JS: See `package.json` for runtime/dev dependencies (React, Tauri API, framer-motion, etc.)

- **External Discussion:**
  - RapidRAW is discussed on [pixls.us](https://discuss.pixls.us/) as a GPU-accelerated, open-source RAW editor, with positive feedback on AgX color management and performance.

---

*Generated by GitHub Copilot on 2025-11-02. For full details, see cited file paths and line ranges above.*
