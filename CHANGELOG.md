# 3D Object Reconstruction 0.2.0 — Commercial Version (08 Jun 2026)

## New Features

- **ONNX-Based Depth Estimation:** Replaced the PyTorch FoundationStereo model with a deployable ONNX model (`foundationstereo_onnx.py`), enabling faster and more portable inference without requiring PyTorch research dependencies.
- **Theseus GPU Pose Optimizer:** Extracted the Theseus-based pose optimization into a dedicated `optimizers/` module (`theseus_optimizer.py`), supporting configurable outer/inner iterations, correspondence weighting, and Huber robust loss.
- **USD and USDZ Export:** Added end-of-pipeline export to USD and USDZ formats (`utils/postprocessing.py`), configurable via `base.yaml` (`export_usd`, `export_usdz` flags).
- **ARM64 / Jetson Orin Support:** Added `docker/Dockerfile.aarch64` for native ARM64 builds; `deploy.sh` now auto-detects the host architecture and selects the appropriate Dockerfile.
- **End-to-End Test Suite:** Added a `tests/` directory with `pytest`-based E2E and unit tests, including markers (`unit`, `e2e`, `slow`, `gpu`) and a `test_reconstruction.sh` shell harness.
- **Granular CLI Logging:** Added a custom `PROGRESS` log level (between `INFO` and `WARNING`) for pipeline step visibility without library noise; new `--verbose`, `--debug`, and `--stage` flags give fine-grained control over log output.

## Improvements

- **Texture Color Fusion:** New configurable color fusion parameters in `base.yaml` (`alpha`, `beta`, `choose_top_n`, `max_angle`, `frame_color_remap`) for improved texture quality on textureless and reflective objects.
- **RoMa Correspondence Clipping:** Added `min_correspondences` and `max_correspondences` config knobs for RoMa feature matching to prevent OOM on high-density scenes and improve stability on textureless objects.
- **Bundle Tracking Progress:** Added `tqdm` progress bars to the bundle tracking loop with per-frame status updates (`loading frame`, `eroding mask`, `running BundleTrack`).
- **Logging Consistency:** Replaced `print` statements with structured `logging` calls throughout the pipeline; `roma_outdoor` stdout is now captured and forwarded to the logger.
- **Docker Base Image Update:** Upgraded from `deepstream:7.1-triton-multiarch` to `pytorch:25.04-py3`; added CUDA arch targets `10.0` and `12.0`; PCL build now runs in `/tmp` for a cleaner image layer.
- **Dependency Pinning:** All Python dependencies pinned to exact versions for reproducible builds; Python minimum raised from `3.8` to `3.10`.

## Bug Fixes

- **PyTorch 2.1+ Deprecation:** Replaced `torch.set_default_tensor_type(torch.cuda.FloatTensor)` with `torch.set_default_device("cuda")` to fix deprecation warnings on PyTorch ≥ 2.1.
- **`igl.signed_distance` Signature:** Fixed 4-return-value unpacking to match the current libigl API (previously caused a `ValueError` at runtime).
- **RoMa Float32 Precision:** Added `torch.set_float32_matmul_precision('highest')` before RoMa model load to disable TF32 and fix accuracy regressions on Ampere+ GPUs.
- **EGL Platform Initialization:** Changed `PYOPENGL_PLATFORM` from a hard `os.environ` set to `os.environ.setdefault`, preventing the variable from overriding Docker-level settings on local runs.
- **SAM2 Frame Sorting:** Simplified frame name sorting to consistent lexicographic order, removing a fragile numeric parse that failed on non-standard filename patterns.

---

# 3D Object Reconstruction 0.1.0 (18 Jul 2025)

## New Features

- **End-to-End 3D Reconstruction Workflow:** Initial release of the 3D Object Reconstruction workflow, providing a complete workflow to convert stereo video inputs into high-quality 3D assets.
- **State-of-the-Art Model Integration:** The workflow integrates several cutting-edge models for robust and accurate reconstruction:
    - **FoundationStereo:** A transformer-based model for high-accuracy stereo depth estimation.
    - **SAM2 (Segment Anything Model 2):** Used for precise and consistent object segmentation in video sequences.
    - **RoMA (Robust Matching):** Employs robust feature matching to establish reliable correspondences between images.
    - **BundleSDF:** Implements neural 6-DoF tracking and 3D reconstruction for unknown objects, ensuring geometric accuracy.
- **Sample Inference Data:** Includes a sample dataset of a retail item with corresponding configuration files, allowing users to quickly test and validate the reconstruction workflow.
- **Docker Compose-Based Deployment:**
    - **Simplified Setup:** A single script (`deploy.sh`) automates the entire setup process, including downloading model weights, building container images, and managing external dependencies.
    - **Pre-configured Environment:** The Dockerfile is based on DeepStream base images and includes all necessary components to run the workflow out-of-the-box.
- **Interactive Jupyter Notebook:**
    - **Step-by-Step Guidance:** A demo notebook (`3d_object_reconstruction_demo.ipynb`) provides an interactive, step-by-step guide through the reconstruction process.
    - **Easy to Use:** Designed for ease of use, allowing users to experiment with the workflow and visualize results in real-time.
- **Command-Line Interface (CLI):**
    - **Automated Workflows:** Provides a CLI for running the reconstruction workflow, enabling batch processing and integration into automated workflows.

## Improvements

- **High-Quality Mesh and Texture Generation:** The workflow is optimized to produce production-ready 3D meshes with photorealistic textures, suitable for digital twin creation, synthetic data generation, and more.
- **Performance:** Achieves rapid processing, with the capability to generate a complete 3D asset in under 30 minutes on an NVIDIA RTX A6000 GPU.
- **Extensibility:** The modular architecture allows for customization and integration of new models or components.

## Bug Fixes

- No major bug fixes in this initial release.
