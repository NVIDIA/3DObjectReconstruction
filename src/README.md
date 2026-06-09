# NVIDIA 3D Object Reconstruction Package

A comprehensive framework for high-quality 3D object reconstruction from RGB-D input using neural implicit surfaces, bundle adjustment, and advanced feature matching.

## Installation

```bash
pip install nvidia-3d-object-reconstruction
```

## Quick Start

```python
from nvidia.reconstruction3d.object.networks import NVBundleSDF
from nvidia.reconstruction3d.object.configs.schema import NVBundleSDFConfig

# Initialize configuration
config = NVBundleSDFConfig()

# Create reconstruction pipeline
pipeline = NVBundleSDF(
    config_nerf=config.nerf,
    cfg_bundletrack=config.bundletrack, 
    roma_config=config.roma
)

# Run the reconstruction pipeline
pipeline.run_track(reader)
pipeline.run_global_sdf(reader)
pipeline.run_texture_bake(reader)
```

## Package Components

### Networks (`nvidia.reconstruction3d.object.networks`)

- **NVBundleSDF**: Main reconstruction pipeline
- **FoundationStereoProcessor**: Stereo depth estimation
- **Sam2Infer**: SAM2-based object segmentation
- **FeatureMatchingInfer**: RoMa feature matching
- **NerfRunner**: Neural Radiance Field implementation

### Configuration (`nvidia.reconstruction3d.object.configs`)

- **NVBundleSDFConfig**: Main configuration schema
- **BundleTrackConfig**: Bundle adjustment settings
- **NeRFConfig**: Neural field parameters
- **FoundationStereoConfig**: Stereo depth settings
- **SAM2Config**: Segmentation parameters

### Utilities (`nvidia.reconstruction3d.object.utils`)

- **preprocessing**: Data preprocessing functions
- **structures**: Data structure utilities

## Individual Component Usage

### Stereo Depth Estimation

```python
from nvidia.reconstruction3d.object.networks import FoundationStereoProcessor

processor = FoundationStereoProcessor(config, rgb_path, output_path)
processor.run()
```

### Object Segmentation

```python
from nvidia.reconstruction3d.object.networks import Sam2Infer

sam2 = Sam2Infer(config)
sam2.run(rgb_path, mask_path)
```

### Feature Matching

```python
from nvidia.reconstruction3d.object.networks import FeatureMatchingInfer

matcher = FeatureMatchingInfer(config)
```

## Configuration Management

```python
from nvidia.reconstruction3d.object.configs.schema import (
    NVBundleSDFConfig,
    FoundationStereoConfig,
    SAM2Config,
    BundleTrackConfig,
    NeRFConfig
)

# Create and customize configuration
config = NVBundleSDFConfig()
config.nerf.n_step = 5000
config.foundation_stereo.scale = 0.5
config.sam2.bbox = [1144, 627, 2227, 2232]
```

## Command Line Interface

```bash
# Run reconstruction pipeline
nvidia-3d-reconstruct --config config.yaml --data-path /path/to/data

# Get help
nvidia-3d-reconstruct --help
```

## Key Features

- **BundleTrack**: Camera pose tracking and bundle adjustment
- **FoundationStereo**: Advanced stereo depth estimation  
- **SAM2**: Object segmentation using Segment Anything Model 2
- **Neural Implicit Surfaces**: High-quality 3D reconstruction using NeRF
- **Texture Baking**: Photorealistic texture generation

## Requirements

- **GPU**: NVIDIA GPU with CUDA support (minimum requirements: Compute Capability 7.0 with at least 24GB VRAM)
- **Memory**: 32GB+ RAM recommended
- **Storage**: 100GB+ free space recommended
- **OS**: Ubuntu 22.04+

## License

NVIDIA License (Non-Commercial) - see LICENSE file for details.

**Important**: This software is for non-commercial use only. This package incorporates third-party components under different licenses including CC BY-NC-SA 4.0. Review the complete LICENSE file for all terms and attributions.

## Support

For issues and questions, please visit the project repository. 