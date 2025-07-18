"""
NVIDIA 3D Object Reconstruction Package.

A comprehensive framework for high-quality 3D object reconstruction from RGB-D input
using neural implicit surfaces, bundle adjustment, and advanced feature matching.

Key Features:
- BundleTrack for camera pose tracking
- FoundationStereo for depth estimation  
- SAM2 for object segmentation
- Neural Implicit Surface representation
- Texture baking for photorealistic results

Example Usage:
    >>> from nvidia.reconstruction3d.object.networks import NVBundleSDF
    >>> from nvidia.reconstruction3d.object.configs.schema import NVBundleSDFConfig
    >>> 
    >>> config = NVBundleSDFConfig()
    >>> pipeline = NVBundleSDF(config.nerf, config.bundletrack, config.roma)
    >>> pipeline.run_track(reader)
    >>> pipeline.run_global_sdf(reader)
"""

__version__ = "1.0.0"
__author__ = "NVIDIA Corporation"
__email__ = "support@nvidia.com"

# Main pipeline imports
from .networks.nvbundlesdf import NVBundleSDF
from .configs.schema import NVBundleSDFConfig

# Individual component imports  
from .networks.foundationstereo import FoundationStereoProcessor, run_depth_estimation
from .networks.sam2infer import Sam2Infer, run_mask_extraction
from .networks.roma import FeatureMatchingInfer
from .dataloader.reconstruction_dataloader import ReconstructionDataLoader

__all__ = [
    'NVBundleSDF',
    'NVBundleSDFConfig', 
    'FoundationStereoProcessor',
    'run_depth_estimation',
    'Sam2Infer', 
    'run_mask_extraction',
    'FeatureMatchingInfer',
    'ReconstructionDataLoader',
    '__version__'
] 