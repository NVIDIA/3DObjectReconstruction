# 3D Object Reconstruction 0.1.0 (18 Jul 2025)

## New Features

- **End-to-End 3D Reconstruction Pipeline:** Initial release of the 3D Object Reconstruction framework, providing a complete workflow to convert stereo video inputs into high-quality 3D assets.
- **State-of-the-Art Model Integration:** The pipeline integrates several cutting-edge models for robust and accurate reconstruction:
    - **FoundationStereo:** A transformer-based model for high-accuracy stereo depth estimation.
    - **SAM2 (Segment Anything Model 2):** Used for precise and consistent object segmentation in video sequences.
    - **RoMA (Robust Matching):** Employs robust feature matching to establish reliable correspondences between images.
    - **BundleSDF:** Implements neural 6-DoF tracking and 3D reconstruction for unknown objects, ensuring geometric accuracy.
- **Sample Inference Data:** Includes a sample dataset of a retail item with corresponding configuration files, allowing users to quickly test and validate the reconstruction pipeline.
- **Docker Compose-Based Deployment:**
    - **Simplified Setup:** A single script (`deploy.sh`) automates the entire setup process, including downloading model weights, building container images, and managing external dependencies.
    - **Pre-configured Environment:** The Dockerfile is based on DeepStream base images and includes all necessary components to run the workflow out-of-the-box.
- **Interactive Jupyter Notebook:**
    - **Step-by-Step Guidance:** A demo notebook (`3d_object_reconstruction_demo.ipynb`) provides an interactive, step-by-step guide through the reconstruction process.
    - **Easy to Use:** Designed for ease of use, allowing users to experiment with the pipeline and visualize results in real-time.
- **Command-Line Interface (CLI):**
    - **Automated Workflows:** Provides a CLI for running the reconstruction pipeline, enabling batch processing and integration into automated workflows.

## Improvements

- **High-Quality Mesh and Texture Generation:** The pipeline is optimized to produce production-ready 3D meshes with photorealistic textures, suitable for digital twin creation, synthetic data generation, and more.
- **Performance:** Achieves rapid processing, with the capability to generate a complete 3D asset in under 30 minutes on an NVIDIA RTX A6000 GPU.
- **Extensibility:** The modular architecture allows for customization and integration of new models or components.

## Bug Fixes

- No major bug fixes in this initial release.
