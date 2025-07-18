"""
Neural Network Models for 3D Object Reconstruction.

This module contains all the neural network components used in the reconstruction pipeline:
- FoundationStereoProcessor: Stereo depth estimation
- NVBundleSDF: Main reconstruction pipeline combining BundleTrack and NeRF
- Sam2Infer: SAM2-based object segmentation
- FeatureMatchingInfer: RoMa-based feature matching
- NerfRunner: Neural Radiance Field implementation
- Tool utilities: Point cloud processing, mesh operations
"""

from .foundationstereo import FoundationStereoProcessor, FoundationStereoNet, run_depth_estimation
from .nvbundlesdf import NVBundleSDF, vis_camera_poses
from .sam2infer import Sam2Infer, run_mask_extraction
from .roma import FeatureMatchingInfer
from .nerf_runner import NerfRunner, ModelRendererOffscreen
from .tool import (
    PointCloudProcessor,
    MeshProcessor, 
    TensorUtils,
    PoseUtils,
    compute_scene_bounds,
    set_seed
)

__all__ = [
    # Main pipeline
    'NVBundleSDF',
    
    # Individual processors
    'FoundationStereoProcessor',
    'FoundationStereoNet', 
    'run_depth_estimation',
    'Sam2Infer',
    'run_mask_extraction',
    'FeatureMatchingInfer',
    'NerfRunner',
    
    # Utility classes
    'PointCloudProcessor',
    'MeshProcessor',
    'TensorUtils', 
    'PoseUtils',
    
    # Utility functions
    'compute_scene_bounds',
    'set_seed',
    'ModelRendererOffscreen',
    'vis_camera_poses'
] 