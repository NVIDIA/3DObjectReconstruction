"""
NVBundleSDF: A Framework for BundleTrack and Neural Implicit Surface Representation

This module implements the BundleTrack framework that combines traditional bundle adjustment tracking
with Neural Implicit Surface representation (SDF) for high-quality 3D reconstruction from RGB-D input.

Key components:
- Bundle adjustment for camera pose tracking and optimization
- Feature correspondence matching using RoMa (Robust Matcher)
- Neural Implicit Surface representation for 3D scene modeling
- Texture baking for final mesh generation

The pipeline processes frames sequentially for tracking and then trains a global neural model
for high-quality 3D reconstruction and texture generation.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import PIL
import numpy as np
import ruamel.yaml
import torch
import trimesh
from trimesh.scene import Scene
import open3d as o3d
import tempfile

# Local imports
from romatch import roma_outdoor
from .tool import (
    PointCloudProcessor,
    MeshProcessor,
    TensorUtils,
    PoseUtils,
    GL_CAM_TO_CV_CAM
)
from .nerf_runner import NerfRunner, trimesh_split, build_texture_from_images, transform_pts, preprocess_data
from .roma import FeatureMatchingInfer
from .tool import set_seed

# Import BundleTrack library (now properly installed in site-packages)
import my_cpp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize YAML parser
yaml = ruamel.yaml.YAML()

# Type aliases for better readability
Frame = my_cpp.Frame
FramePair = Tuple[Frame, Frame]
FramePairs = List[FramePair]
ConfigDict = Dict[str, Any]
ImageArray = np.ndarray
PoseMatrix = np.ndarray
DepthMap = np.ndarray
MaskArray = np.ndarray

# Ensure consistent tensor types for NVBundleSDF operations
def set_nvbundlesdf_precision(use_full_precision: bool = True) -> torch.dtype:
    """Configure PyTorch tensor types for NVBundleSDF operations.
    
    This function sets the default tensor type and precision for all NVBundleSDF operations.
    Using full precision (float32) is recommended for pose tracking to ensure numerical stability.
    
    Args:
        use_full_precision: Whether to use full precision (float32) or allow mixed precision.
            Defaults to True for maximum stability.
            
    Returns:
        torch.dtype: The current default tensor type after configuration.
        
    Note:
        When use_full_precision is True:
        - Forces float32 precision for all tensors
        - Uses CUDA FloatTensor if CUDA is available
        - Ensures maximum numerical stability for pose tracking
    """
    if use_full_precision:
        # Force full precision (safest option for pose tracking)
        torch.set_default_dtype(torch.float32)
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        logger.info("Set PyTorch to use full precision (float32) for NVBundleSDF")
    return torch.get_default_dtype()

# Define required tensor types for different operations
ROMA_TYPES = {
    'rgbAs': torch.float32,
    'rgbBs': torch.float32,
    'warp': torch.float32,
    'certainty': torch.float32,
    'matches': torch.float32
}

# Update tensor type utilities
def ensure_tensor_type(tensor: torch.Tensor, target_dtype: torch.dtype = torch.float32, 
                      name: str = "tensor") -> torch.Tensor:
    """Ensure a tensor has the specified data type.
    
    This is a wrapper function for backward compatibility that ensures tensors
    have the correct data type for NVBundleSDF operations.
    
    Args:
        tensor: Input tensor to convert
        target_dtype: Desired data type for the tensor
        name: Name of the tensor for logging purposes
        
    Returns:
        torch.Tensor: Tensor converted to the target data type
        
    Raises:
        TypeError: If input is not a torch.Tensor
    """
    return TensorUtils.ensure_tensor_type(tensor, target_dtype, name)

def ensure_tensor_types(tensors_dict: Dict[str, torch.Tensor], 
                       dtypes_dict: Dict[str, torch.dtype]) -> Dict[str, torch.Tensor]:
    """Ensure multiple tensors have their specified data types.
    
    This is a wrapper function for backward compatibility that ensures multiple
    tensors have the correct data types for NVBundleSDF operations.
    
    Args:
        tensors_dict: Dictionary of tensors to convert
        dtypes_dict: Dictionary mapping tensor names to their target data types
        
    Returns:
        Dict[str, torch.Tensor]: Dictionary of tensors converted to their target data types
        
    Raises:
        KeyError: If a tensor name in dtypes_dict is not found in tensors_dict
        TypeError: If any input is not a torch.Tensor
    """
    return TensorUtils.ensure_tensor_types(tensors_dict, dtypes_dict)

class NVBundleSDF:
    """NVBundleSDF framework for 3D reconstruction combining bundle adjustment with neural implicit surfaces.
    
    This class implements the complete reconstruction pipeline:
    1. Camera pose tracking using BundleTrack
    2. Neural implicit surface representation
    3. Mesh extraction and texture baking
    
    Attributes:
        cfg_nerf (ConfigDict): Configuration dictionary for neural model parameters
        debug_dir (Path): Directory for saving debug outputs
        translation (Optional[np.ndarray]): Translation vector for scene normalization
        sc_factor (Optional[float]): Scale factor for scene normalization
        cnt (int): Frame counter
        roma (FeatureMatchingInfer): Feature matcher instance
        K (Optional[np.ndarray]): Camera intrinsics matrix
    """
    
    def __init__(self, config_nerf: ConfigDict, cfg_bundletrack: ConfigDict, roma_config: ConfigDict, cfg_texture: ConfigDict, logger: logging.Logger = None) -> None:
        """Initialize the NVBundleSDF framework.
        
        Args:
            config_nerf: Configuration dictionary for neural model parameters
            cfg_bundletrack: Configuration dictionary for BundleTrack
            roma_config: Configuration dictionary for RoMa
            logger: Logger instance for logging
            
        Raises:
            RuntimeError: If configuration files cannot be loaded
        """
        # Set full precision for pose tracking operations
        set_seed(42)

        self.orig_dtype = set_nvbundlesdf_precision(use_full_precision=True)
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        # Store configuration
        self.cfg_nerf = config_nerf
        self.roma_config = roma_config
        self.cfg_bundletrack = cfg_bundletrack
        self.cfg_track = cfg_bundletrack  # For backward compatibility
        self.cfg_texture = cfg_texture
        self.debug_dir = Path(config_nerf['save_dir'])
        self.debug_dir.mkdir(exist_ok=True, parents=True)

        # Initialize state variables
        self.translation: Optional[np.ndarray] = None
        self.sc_factor: Optional[float] = None
        self.cnt: int = -1
        self.K: Optional[np.ndarray] = None
        
        # Initialize feature matcher
        self.roma = FeatureMatchingInfer(roma_config, logger)
    


    def run_track(self, reader):
        """Run tracking on all frames from the input reader."""
        # Move bundler initialization here
        cfg_bundletrack_str = {}
        for key, value in self.cfg_bundletrack.items():
            if isinstance(value, Path):
                cfg_bundletrack_str[key] = str(value)
            elif isinstance(value, dict):
                cfg_bundletrack_str[key] = {k: str(v) if isinstance(v, Path) else v for k, v in value.items()}
            else:
                cfg_bundletrack_str[key] = value
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            yaml.dump(cfg_bundletrack_str, temp_file)
            temp_file_path = temp_file.name
            
        yml = my_cpp.YamlLoadFile(temp_file_path)
        if yml is None:
            raise RuntimeError(f"Failed to load YAML file: {temp_file_path}")
        
        self.logger.info("Initializing BundleTrack...")
        try:
            self.bundler = my_cpp.Bundler(yml)
        except Exception as e:
            self.logger.error(f"Error creating Bundler: {str(e)}")
            raise
        
        # Ensure we're using full precision for tracking
        set_nvbundlesdf_precision(use_full_precision=True)
        cfg_bundletrack = self.cfg_bundletrack
        
        # Copy camera intrinsics
        K = reader.K.copy()
        
        self.logger.info(f"Processing {len(reader.color_files)} frames for tracking...")
        
        # Process all frames
        for i in range(0, len(reader.color_files), 1):
            # Load frame data
            color = reader.get_color(i)
            depth = reader.get_depth(i)
            mask = reader.get_mask(i)
            id_str = reader.id_strs[i]
            
            
            # Apply mask erosion if configured
            if cfg_bundletrack['erode_mask'] > 0:
                kernel_size = cfg_bundletrack['erode_mask']
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask = cv2.erode(mask.astype(np.uint8), kernel)
            
            # Initialize pose
            pose_in_model = np.eye(4)
            
            # Process frame
            self.run_bundletrack(color, depth, K, id_str, mask, None, pose_in_model)
            
            # Save keyframes
            keyframe_path = Path(self.cfg_bundletrack['debug_dir']) / id_str / "keyframes.yml"
            if keyframe_path.exists():
                shutil.copy(keyframe_path, Path(self.cfg_bundletrack['debug_dir']) / "keyframes.yml")
                self.logger.info(f"Copied keyframes to {Path(self.cfg_bundletrack['debug_dir']) / 'keyframes.yml'} for frame {id_str}")
            else:
                self.logger.warning(f"Keyframes file not found at {keyframe_path}")

        # Copy the keyframes.yml file to the debug_dir
        final_keyframes = Path(self.cfg_bundletrack['debug_dir']) / "keyframes.yml"
        if final_keyframes.exists():
            shutil.copy(final_keyframes, Path(self.debug_dir) / "keyframes.yml")
        else:
            self.logger.warning(f"Expected keyframes file {final_keyframes} not found – skipping copy")

        self.logger.info(f"Copied keyframes to {Path(self.debug_dir) / 'keyframes.yml'}")

        # At the end, explicitly delete bundler to free memory
        del self.bundler
        self.bundler = None  # Clear reference

    def make_frame(self, color: ImageArray, depth: DepthMap, K: np.ndarray, 
                  id_str: str, mask: Optional[MaskArray] = None, 
                  occ_mask: Optional[MaskArray] = None, 
                  pose_in_model: Optional[PoseMatrix] = None) -> Frame:
        """Create a Frame object for BundleTrack processing.
        
        This method creates a new frame object with the provided RGB-D data and optional masks.
        The frame is used for tracking and bundle adjustment in the BundleTrack pipeline.
        
        Args:
            color: RGB image array of shape (H, W, 3)
            depth: Depth map array of shape (H, W)
            K: Camera intrinsics matrix of shape (3, 3)
            id_str: Unique identifier for the frame
            mask: Optional foreground mask array of shape (H, W)
            occ_mask: Optional occlusion mask array of shape (H, W)
            pose_in_model: Optional initial pose matrix of shape (4, 4)
            
        Returns:
            Frame: Initialized Frame object ready for BundleTrack processing
            
        Raises:
            ValueError: If input arrays have incompatible shapes
            TypeError: If input arrays have incorrect data types
        """
        # Get image dimensions
        H, W = color.shape[:2]
        roi = [0, W-1, 0, H-1]
        
        # Default pose if not provided
        if pose_in_model is None:
            pose_in_model = np.eye(4)

        # Create frame
        frame = Frame(
            color, depth, roi, pose_in_model, 
            self.cnt, id_str, K, self.bundler.yml
        )
        
        # Set masks if provided
        if mask is not None:
            frame._fg_mask = my_cpp.cvMat(mask)
        if occ_mask is not None:
            frame._occ_mask = my_cpp.cvMat(occ_mask)
            
        return frame

    def run_bundletrack(self, color: ImageArray, depth: DepthMap, K: np.ndarray,
                       id_str: str, mask: Optional[MaskArray] = None,
                       occ_mask: Optional[MaskArray] = None,
                       pose_in_model: Optional[PoseMatrix] = None) -> None:
        """Process a single frame through the BundleTrack algorithm.
        
        This method handles the complete processing pipeline for a single frame:
        1. Depth preprocessing if configured
        2. Frame creation and initialization
        3. Frame processing through BundleTrack
        4. Result saving and logging
        
        Args:
            color: RGB image array of shape (H, W, 3)
            depth: Depth map array of shape (H, W)
            K: Camera intrinsics matrix of shape (3, 3)
            id_str: Unique identifier for the frame
            mask: Optional foreground mask array of shape (H, W)
            occ_mask: Optional occlusion mask array of shape (H, W)
            pose_in_model: Optional initial pose matrix of shape (4, 4)
            
        Raises:
            ValueError: If input arrays have incompatible shapes
            RuntimeError: If frame processing fails
        """
        # Increment frame counter
        self.cnt += 1
        
        # Apply depth preprocessing if configured
        if mask is not None and self.cfg_bundletrack['depth_processing']["percentile"] < 100:
            self.logger.info("Applying depth preprocessing...")
            percentile = self.cfg_bundletrack['depth_processing']["percentile"]
            valid = (depth >= 0.1) & (mask > 0)
            
            if valid.sum() > 0:
                thres = np.percentile(depth[valid], percentile)
                depth[depth >= thres] = 0
                self.logger.info(f"Depth threshold: {thres:.4f}, remaining valid points: {(depth > 0).sum()}")
        
        # Create frame object
        frame = self.make_frame(color, depth, K, id_str, mask, occ_mask, pose_in_model)
        
        # Create debug directory
        os.makedirs(f"{self.bundler.yml['debug_dir']}/{frame._id_str}", exist_ok=True)

        # Process frame
        self.logger.info(f"Processing frame {id_str} (ID: {self.cnt})...")
        self.process_new_frame(frame)
        self.logger.info(f"Frame {id_str} processing complete")
        
        # Save results
        self.bundler.saveNewframeResult()

    def process_new_frame(self, frame: Frame) -> None:
        """Process a new frame through bundle adjustment pipeline.
        
        This method handles the core frame processing logic:
        1. Frame initialization and validation
        2. Feature matching and correspondence finding
        3. Pose optimization and bundle adjustment
        4. Keyframe selection and management
        
        Args:
            frame: The frame to process
            
        Raises:
            RuntimeError: If frame processing fails
            ValueError: If frame validation fails
        """
        self.logger.info(f"Processing frame {frame._id_str}")
        
        # Set up frame for processing
        self.bundler._newframe = frame
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Handle first vs. subsequent frames differently
        if frame._id > 0:
            # For subsequent frames, set reference to most recent frame
            ref_frame = self.bundler._frames[list(self.bundler._frames.keys())[-1]]
            frame._ref_frame_id = ref_frame._id
            frame._pose_in_model = ref_frame._pose_in_model
        else:
            # For first frame, set as first frame
            self.bundler._firstframe = frame
        
        # Apply mask to invalidate pixels
        frame.invalidatePixelsByMask(frame._fg_mask)
        
        # For first frame with identity pose, set new coordinate system
        if frame._id == 0 and np.abs(np.array(frame._pose_in_model) - np.eye(4)).max() <= 1e-4:
            frame.setNewInitCoordinate()
        
        # Validate frame has sufficient foreground points
        n_fg = (np.array(frame._fg_mask) > 0).sum()
        self.logger.info(f"Foreground points: {n_fg}")
        if n_fg < 100:
            self.logger.warning(f"Frame {frame._id_str} has insufficient foreground points: {n_fg}")
            frame._status = Frame.FAIL
            self.bundler.forgetFrame(frame)
            return
        
        # Validate frame has sufficient valid points
        n_valid = frame.countValidPoints()
        self.logger.info(f"Valid points: {n_valid}")
        
        # Apply cloud denoising if configured
        if self.cfg_track["depth_processing"]["denoise_cloud"]:
            frame.pointCloudDenoise()
            self.logger.info(f"After denoising, valid points: {frame.countValidPoints()}")
        
        # Compare with first frame for validation
        n_valid_first = self.bundler._firstframe.countValidPoints()
        if n_valid < n_valid_first / 40.0:
            self.logger.warning(
                f"Frame {frame._id_str} has too few valid points ({n_valid}) "
                f"compared to first frame ({n_valid_first})"
            )
            frame._status = Frame.FAIL
            self.bundler.forgetFrame(frame)
            return
        
        # For first frame, simply add as keyframe
        if frame._id == 0:
            self.bundler.checkAndAddKeyframe(frame)  # First frame is always keyframe
            self.bundler._frames[frame._id] = frame
            return
        
        # For subsequent frames, find correspondences with reference frame
        min_match_with_ref = self.cfg_track["feature_corres"]["min_match_with_ref"]
        self.find_corres([(frame, ref_frame)])
        
        # Check if correspondence finding failed
        if frame._status == Frame.FAIL:
            self.logger.warning(f"Failed to find correspondences for frame {frame._id_str}")
            self.bundler.forgetFrame(frame)
            return
        
        # Check if sufficient matches were found
        matches = self.bundler._fm._matches[(frame, ref_frame)]
        if len(matches) < min_match_with_ref:
            # Try alternative reference frames
            self.logger.info(f"Insufficient matches with current reference frame, trying alternatives...")
            visibles = []
            for kf in self.bundler._keyframes:
                visible = my_cpp.computeCovisibility(frame, kf)
                visibles.append(visible)
            
            visibles = np.array(visibles)
            ids = np.argsort(visibles)[::-1]
            found = False
            for id in ids:
                kf = self.bundler._keyframes[id]
                self.logger.info(f"Trying alternative reference frame {kf._id_str}")
                ref_frame = kf
                frame._ref_frame_id = kf._id
                frame._pose_in_model = kf._pose_in_model
                self.find_corres([(frame, ref_frame)])
                
                if len(self.bundler._fm._matches[(frame, kf)]) >= min_match_with_ref:
                    self.logger.info(f"Found suitable reference frame: {kf._id_str}")
                    found = True
                    break
            
            if not found:
                self.logger.warning(f"No suitable reference frame found for {frame._id_str}")
                frame._status = Frame.FAIL
                self.bundler.forgetFrame(frame)
                return
        
        # Update pose using Procrustes alignment
        self.logger.info(f"Updating pose for frame {frame._id_str}\nBefore:\n{frame._pose_in_model.round(3)}")
        offset = self.bundler._fm.procrustesByCorrespondence(frame, ref_frame)
        frame._pose_in_model = offset @ frame._pose_in_model
        self.logger.info(f"After:\n{frame._pose_in_model.round(3)}")
        
        # Manage frame window size
        window_size = self.cfg_track["bundle"]["window_size"]
        if len(self.bundler._frames) - len(self.bundler._keyframes) > window_size:
            for k in self.bundler._frames:
                f = self.bundler._frames[k]
                is_forget = self.bundler.forgetFrame(f)
                if is_forget:
                    self.logger.info(f"Window size exceeded, forgetting frame {f._id_str}")
                    break
        
        # Add frame to tracked frames
        self.bundler._frames[frame._id] = frame
        
        # Select keyframes for bundle adjustment
        self.bundler.selectKeyFramesForBA()
        local_frames = self.bundler._local_frames
        
        # Find correspondences between all local frames
        pairs = self.bundler.getFeatureMatchPairs(local_frames)
        self.find_corres(pairs)
        
        if frame._status == Frame.FAIL:
            self.bundler.forgetFrame(frame)
            return
        
        # Run optimization
        find_matches = False
        self.bundler.optimizeGPU(local_frames, find_matches)
        
        if frame._status == Frame.FAIL:
            self.bundler.forgetFrame(frame)
            return
        
        # Check if frame should be added as a keyframe
        self.bundler.checkAndAddKeyframe(frame)

    def find_corres(self, frame_pairs: FramePairs) -> None:
        """Find correspondences between pairs of frames using RoMa feature matching.
        
        This method implements the feature matching pipeline:
        1. Image pair preprocessing
        2. RoMa feature matching
        3. Correspondence validation and RANSAC filtering
        4. Result visualization and storage
        
        Args:
            frame_pairs: List of frame pairs to match
            
        Raises:
            RuntimeError: If feature matching fails
            ValueError: If input frame pairs are invalid
        """
        self.logger.info(f"Finding correspondences between {len(frame_pairs)} frame pairs")
        
        # Check if matching with reference frame
        is_match_ref = (
            len(frame_pairs) == 1 and 
            frame_pairs[0][0]._ref_frame_id == frame_pairs[0][1]._id and 
            self.bundler._newframe == frame_pairs[0][0]
        )
        
        # Get processed image pairs
        imgs, tfs, query_pairs = self.bundler._fm.getProcessedImagePairs(frame_pairs)
        
        if len(query_pairs) == 0:
            self.logger.info("No valid query pairs found")
            return
        
        # Prepare images for RoMa with correct precision
        img_shape = np.array(imgs[0]).shape
        array_imgs = np.empty((len(imgs), *img_shape), dtype=np.uint8)
        
        for i, img in enumerate(imgs):
            array_imgs[i] = np.array(img)
        
        # Ensure arrays are in float32 for RoMa
        rgbAs = ensure_tensor_type(torch.tensor(array_imgs[::2]), torch.float32, 'rgbAs')
        rgbBs = ensure_tensor_type(torch.tensor(array_imgs[1::2]), torch.float32, 'rgbBs')
        
        # Run RoMa feature matching with float32 precision
        # with torch.cuda.amp.autocast(enabled=False):
        self.logger.info("Running RoMa feature matching...")
        set_seed(42)
        corres = self.roma._process_batch(rgbAs.cpu().numpy(), rgbBs.cpu().numpy())
        self.logger.info(f"Found correspondences: {len(corres)} pairs, shape: {corres[0].shape}")
        
        # Process correspondences
        for i_pair in range(len(query_pairs)):
            if i_pair < len(corres):
                cur_corres = corres[i_pair][:, :4]
                frame_pair = query_pairs[i_pair]
                tfA = np.array(tfs[i_pair * 2])
                tfB = np.array(tfs[i_pair * 2 + 1])
                
                # Transform correspondences back to original image coordinates
                cur_corres[:, :2] = transform_pts(cur_corres[:, :2], np.linalg.inv(tfA))
                cur_corres[:, 2:4] = transform_pts(cur_corres[:, 2:4], np.linalg.inv(tfB))
                
                # Store matches in the raw_matches dictionary
                self.bundler._fm._raw_matches[frame_pair] = cur_corres.round().astype(np.uint16)
            else:
                self.logger.warning(f"Missing correspondences for pair {i_pair}")
        
        # Free memory
        del corres, array_imgs
        
        # Apply minimum match threshold for reference frame
        min_match_with_ref = self.cfg_track["feature_corres"]["min_match_with_ref"]
        
        if is_match_ref and len(self.bundler._fm._raw_matches[frame_pairs[0]]) < min_match_with_ref:
            self.logger.warning(f"Insufficient matches ({len(self.bundler._fm._raw_matches[frame_pairs[0]])}) "
                         f"with reference frame (required: {min_match_with_ref})")
            self.bundler._fm._raw_matches[frame_pairs[0]] = []
            self.bundler._newframe._status = Frame.FAIL
            return
        
        # Convert raw matches to correspondences
        self.bundler._fm.rawMatchesToCorres(query_pairs)
        
        # Visualize correspondences before RANSAC
        for pair in query_pairs:
            self.bundler._fm.vizCorresBetween(pair[0], pair[1], 'before_ransac')
        
        # Run RANSAC for robust matching
        self.logger.info("Running RANSAC for robust correspondence estimation...")
        self.bundler._fm.runRansacMultiPairGPU(query_pairs)
        
        # Visualize correspondences after RANSAC
        for pair in query_pairs:
            self.bundler._fm.vizCorresBetween(pair[0], pair[1], 'after_ransac')

    def run_global_sdf(self, reader):
        """
        Run global SDF optimization on tracked frames.
        
        Args:
            reader: Data reader providing input frames
        """
        self.logger.info("Running global SDF optimization...")
        
        # Load keyframes from tracking
        keyframes_path = Path(self.debug_dir) / "keyframes.yml"
        with open(keyframes_path, 'r') as f:
            keyframes = yaml.load(f)
        
        keys = list(keyframes.keys())
        self.logger.info(f"Found {len(keys)} keyframes for SDF optimization")
        
        # Extract camera poses
        cam_in_obs = []
        for k in keys:
            cam_in_ob = np.array(keyframes[k]['cam_in_ob']).reshape(4, 4)
            cam_in_obs.append(cam_in_ob)
        cam_in_obs = np.array(cam_in_obs)
        
        # Extract frame IDs
        frame_ids = []
        for k in keys:
            key_frames_id = k.replace('keyframe_', '')
            frame_ids.append(key_frames_id)
        self.logger.info(f"Frame IDs for SDF: {frame_ids}")
        
        # Load RGB-D data for keyframes
        rgb_list = []
        id_list = []
        self.K = reader.K.copy()
        for frame_id in frame_ids:
            id = reader.id_strs.index(frame_id)
            rgb_list.append(reader.color_files[id])
            id_list.append(frame_id)
        reader.color_files = rgb_list
        reader.id_strs = id_list

        # Transform camera poses to OpenGL convention
        glcam_in_obs = cam_in_obs @ GL_CAM_TO_CV_CAM
        
        # Compute scene bounds for normalization
        self.logger.info("Computing scene bounds...")
        # Load RGB images first
        rgb_images = [reader.get_color(i) for i in range(len(reader.color_files))]
        sc_factor, translation_cvcam, pcd_real_scale, pcd = PointCloudProcessor.compute_scene_bounds(
            rgb_images, 
            [reader.get_depth(i) for i in range(len(reader.color_files))], 
            [reader.get_mask(i) for i in range(len(reader.color_files))], 
            glcam_in_obs, 
            self.K,
            translation_cvcam=None, 
            sc_factor=None, 
            eps=0.06, 
            min_samples=1, 
            cluster=True
        )
        print(sc_factor, translation_cvcam, len(pcd.points))
        # save the pcd_normalized
        pcd_normalized_path = Path(self.debug_dir) / "pcd_normalized.ply"
        o3d.io.write_point_cloud(str(pcd_normalized_path), pcd_real_scale)
        
        # Store normalization parameters
        self.cfg_nerf['translation'] = translation_cvcam
        self.cfg_nerf['sc_factor'] = sc_factor

        with open(f'{self.debug_dir}/normalization.yml','w') as ff:
            tmp = {
            'translation_cvcam':translation_cvcam.tolist(),
            'sc_factor':float(sc_factor),
            }
            yaml.dump(tmp,ff)

        self.logger.info(f"Scene normalization: scale={sc_factor}, translation={translation_cvcam}")
        
        # Preprocess data for SDF
        poses = glcam_in_obs
        poses[:, :3, 3] += translation_cvcam
        poses[:, :3, 3] *= sc_factor
        
        # Initialize SDF model
        self.logger.info("Initializing SDF model...")
        sdf = NerfRunner(
            self.cfg_nerf, reader, occ_masks=None, poses=poses,
            K=self.K, build_octree_pcd=pcd
        )
        
        # Train SDF model
        self.logger.info("Training SDF model...")
        sdf.train()
        # sdf.save_weights(str(Path(self.debug_dir) / "model_latest.pth"))
        
        # Extract mesh from trained model
        self.logger.info("Extracting mesh from SDF model...")
        mesh, sigma, query_pts = sdf.extract_mesh(
            voxel_size=self.cfg_nerf['mesh_resolution'],
            isolevel=0,
            mesh_savepath='',
            return_sigma=True
        )
        
        # Post-process mesh
        mesh.merge_vertices()
        ms = trimesh_split(mesh, min_edge=1000)
        
        # Find largest connected component
        largest_size = 0
        largest = None
        for m in ms:
            if m.vertices.shape[0] > largest_size:
                largest_size = m.vertices.shape[0]
                largest = m
        mesh = largest
        mesh_path = Path(self.debug_dir) / "mesh_cleaned.obj"
        mesh.export(str(mesh_path))
        self.logger.info(f"Saved cleaned mesh to {mesh_path}")

    def run_texture_bake(self, reader) -> None:
        """Run texture baking on the reconstructed mesh.
        
        This method implements the texture baking pipeline:
        1. Load keyframes and camera poses
        2. Compute scene bounds and normalization
        3. Load pre-trained SDF model
        4. Generate and apply textures to mesh
        
        Args:
            reader: Data reader providing input frames
            cfg_texture: Configuration dictionary for texture baking
            
        Raises:
            FileNotFoundError: If required files are missing
            RuntimeError: If texture baking fails
        """
        self.logger.info("Running texture baking...")
        cfg_texture = self.cfg_texture
        
        # Load keyframes from tracking
        keyframes_path = Path(self.debug_dir) / "keyframes.yml"
        with open(keyframes_path, 'r') as f:
            keyframes = yaml.load(f)
        
        keys = list(keyframes.keys())
        self.logger.info(f"Found {len(keys)} keyframes for texture baking")
        
        # Extract camera poses
        cam_in_obs = []
        for k in keys:
            cam_in_ob = np.array(keyframes[k]['cam_in_ob']).reshape(4, 4)
            cam_in_obs.append(cam_in_ob)
        cam_in_obs = np.array(cam_in_obs)
        
        # Extract frame IDs
        frame_ids = []
        for k in keys:
            key_frames_id = k.replace('keyframe_', '')
            frame_ids.append(key_frames_id)
        self.logger.info(f"Frame IDs for texture baking: {frame_ids}")
        
        # Transform camera poses to OpenGL convention
        glcam_in_obs = cam_in_obs @ GL_CAM_TO_CV_CAM
        
        # Load RGB-D data for keyframes
        rgb_list = []
        id_list = []
        self.K = reader.K.copy()
        for frame_id in frame_ids:
            id = reader.id_strs.index(frame_id)
            rgb_list.append(reader.color_files[id])
            id_list.append(frame_id)
            self.cnt += 1
        reader.color_files = rgb_list
        reader.id_strs = id_list
        
        # Compute scene bounds for normalization
        self.logger.info("Computing scene bounds for texture baking...")
        # Load RGB images first
        ocree_pcd_path = f"{self.cfg_nerf['save_dir']}/build_octree_cloud.ply"
        if os.path.exists(ocree_pcd_path):
            pcd = o3d.io.read_point_cloud(str(ocree_pcd_path))
        else:
            pcd = np.zeros((1,3)) # place holder
        if 'sc_factor' not in self.cfg_nerf:
            assert os.path.exists(f'{self.debug_dir}/normalization.yml')
            with open(f'{self.debug_dir}/normalization.yml','r') as f:
                tmp = yaml.load(f)
                sc_factor = tmp['sc_factor']
                translation_cvcam = tmp['translation_cvcam']
                self.cfg_nerf['sc_factor'] = sc_factor
                self.cfg_nerf['translation'] = translation_cvcam
        sc_factor = self.cfg_nerf['sc_factor']
        translation_cvcam = self.cfg_nerf['translation']
        # Store normalization parameters
        #self.cfg_nerf['translation'] = translation_cvcam
        #self.cfg_nerf['sc_factor'] = sc_factor
        self.logger.info(f"Scene normalization: scale={sc_factor}, translation={translation_cvcam}")
        
        # Preprocess data for SDF
        poses = glcam_in_obs
        poses[:, :3, 3] += translation_cvcam
        poses[:, :3, 3] *= sc_factor
        
        # Initialize SDF model
        self.logger.info("Initializing SDF model for texture baking...")
        sdf = NerfRunner(
            self.cfg_nerf, reader, occ_masks=None, poses=poses,
            K=self.K, build_octree_pcd=pcd, # place holder
        )
        
        # Load pre-trained weights
        model_path = Path(self.debug_dir) / "model_latest.pth"
        self.logger.info(f"Loading pre-trained SDF weights from {model_path}")
        sdf.load_weights(str(model_path))
        
        # Prepare for texture baking
        _ids = torch.arange(len(reader)).long().cuda()
        
        optimized_cvcam_in_obs, offset = PoseUtils.get_optimized_poses_in_real_world(
            poses, sdf.models['pose_array'], sc_factor, translation_cvcam
        )

        # Get optimized poses if available
        sdf_pose = poses
        if sdf.models['pose_array'] is not None:
            self.logger.info("Using optimized poses from SDF training")
            _ids = torch.arange(len(reader)).long().cuda()
            sdf_pose = sdf.models['pose_array'].get_matrices(_ids)
            sdf_pose = sdf_pose.detach().cpu().numpy()
            sdf_pose = sdf_pose @ poses
        
        # Load the cleaned mesh
        mesh_path = Path(self.debug_dir) / "mesh_cleaned.obj"
        self.logger.info(f"Loading mesh from {mesh_path}")
        mesh = trimesh.load(str(mesh_path))
        mesh = MeshProcessor.smooth_mesh(mesh, self.cfg_nerf, self.debug_dir)

        # Run texture baking
        Height, Width = reader.H, reader.W
        self.logger.info(f"Baking texture with resolution {cfg_texture['texture_res']}")

        mesh = build_texture_from_images(
            mesh, reader, sdf_pose, 
            Height=Height, Width=Width, K=self.K,
            sc_factor=sc_factor, far=reader.far, 
            tex_res=cfg_texture['texture_res'],debug_dir=self.debug_dir
        )
        
        # Rescale the mesh to the real world
        mesh = MeshProcessor.mesh_to_real_world(mesh, pose_offset=offset, translation=translation_cvcam, sc_factor=sc_factor)
        
        # Save textured mesh
        textured_mesh_path = Path(self.debug_dir) / "textured_mesh.obj"
        mesh.export(str(textured_mesh_path))
        self.logger.info(f"Saved textured mesh to {textured_mesh_path}")


def vis_camera_poses(file_path,reader=None,scale=0.05,eps=0.06,image_scale=100,default_object_scale=0.1):
    """Visualize camera pose from a file"""
    def create_camera_geometry(pose, img=None, scale=0.05):
        """Create a WIS3D-style camera using Trimesh primitives."""
        geometries = []

        # Draw axis lines (red=X, green=Y, blue=Z)
        for i, color in enumerate([[255, 0, 0], [0, 255, 0], [0, 0, 255]]):
            direction = pose[:3, i] * scale
            points = np.array([pose[:3, 3], pose[:3, 3] + direction])
            line = trimesh.load_path(points)
            line.colors = np.tile(color + [255], (len(line.entities), 1))
            geometries.append(line)

        # Draw camera frustum pyramid
        origin = pose[:3, 3]
        z = pose[:3, 2] * scale * 1.5
        center = origin + z

        x_axis = pose[:3, 0]  # right
        y_axis = pose[:3, 1]  # up

        # Adjust the width/height scale of the camera frustum if the image is provided
        if img is not None:
            H, W = img.shape[:2]
            half_height = scale * 0.5
            aspect = W / H
            half_width = half_height * aspect
        else:
            half_width = half_height = scale * 0.4  # square fallback

        # Define the corners of the camera frustum
        corners = [
            center + x_axis * half_width + y_axis * half_height,   # top-right
            center - x_axis * half_width + y_axis * half_height,   # top-left
            center - x_axis * half_width - y_axis * half_height,   # bottom-left
            center + x_axis * half_width - y_axis * half_height    # bottom-right
        ]

        # If an image is provided, show its pixels in the camera frustum as point cloud
        if img is not None:
            H, W = img.shape[:2]
            # Generate normalized grid in quad space (UV)
            us = np.linspace(0, 1, W)
            vs = np.linspace(0, 1, H)
            uu, vv = np.meshgrid(us, vs)
            uu = 1.0 - uu[..., None]  # shape (H, W, 1)
            vv = 1.0 - vv[..., None]  # shape (H, W, 1)
            # A: top-left, B: top-right, C: bottom-right, D: bottom-left
            A = corners[1]  # top-left
            B = corners[0]  # top-right
            C = corners[3]  # bottom-right
            D = corners[2]  # bottom-left
            # Bilinear interpolation for each pixel
            quad_points = (
                (1 - uu) * (1 - vv) * B +
                uu * (1 - vv) * A +
                (1 - uu) * vv * C +
                uu * vv * D
            )

            # Flatten to (N, 3)
            points = quad_points.reshape(-1, 3)
            colors = img.reshape(-1, 3)
            pcd = trimesh.PointCloud(vertices=points, colors=colors)
            geometries.append(pcd)

        # Lines from origin to each corner
        for c in corners:
            line = trimesh.load_path(np.array([origin, c]))
            line.colors = np.array([[255, 255, 0, 255]])  # yellow
            geometries.append(line)

        # Base square
        for i in range(4):
            line = trimesh.load_path(np.array([corners[i], corners[(i + 1) % 4]]))
            line.colors = np.array([[255, 255, 0, 255]])
            geometries.append(line)

        return geometries

    with open(file_path, 'r') as f:
        keyframes = yaml.load(f)
    
    keys = list(keyframes.keys())
    print(f"Selected {len(keys)} keyframes")
    
    # Extract camera poses
    cam_in_obs = []
    for k in keys:
        cam_in_ob = np.array(keyframes[k]['cam_in_ob']).reshape(4, 4)
        cam_in_obs.append(cam_in_ob)
    poses = np.array(cam_in_obs)
    
    # Plot
    scene = Scene()
    for i, pose in enumerate(poses):
        if reader is None:
            for g in create_camera_geometry(pose, scale=scale):
                scene.add_geometry(g)
        else:
            # Draw camera frustum with image if provided from the reader
            try:
                id = reader.id_strs.index(keys[i].replace('keyframe_', ''))
                img = PIL.Image.fromarray(reader.get_color(id)[...,::-1])
                w, h = img.size
                # Define the resolution of the image (image_scale) to be shown in the camera frustum
                if w >= h:
                    new_w = image_scale
                    new_h = int(h * (image_scale / w))
                else:
                    new_h = image_scale
                    new_w = int(w * (image_scale / h))
                # Resize
                img = np.array(img.resize((new_w, new_h), resample=PIL.Image.BICUBIC))
                for g in create_camera_geometry(pose, img=img, scale=scale):
                    scene.add_geometry(g)
            except Exception as exc:   # noqa: BLE001   
                logger.warning("Visualisation fallback triggered: %s", exc)
                for g in create_camera_geometry(pose, scale=scale):
                    scene.add_geometry(g)

    if reader is  None:
        # use a small box (length=default_object_scale) to replace the object
        unitbox = trimesh.primitives.Box(extents=[default_object_scale] * 3)
        scene.add_geometry(unitbox)
    else:
        # Extract frame IDs
        factor = max(1,int(len(keys)//5))
        selec_id = list(range(len(keys)))[::factor]
        cam_in_obs = poses[selec_id,...]
        keys = [keys[i] for i in selec_id]
        frame_ids = []
        for k in keys:
            key_frames_id = k.replace('keyframe_', '')
            frame_ids.append(key_frames_id)
            
        # Load RGB-D data for keyframes
        id_list = []
        K = reader.K.copy()
        for frame_id in frame_ids:
            id = reader.id_strs.index(frame_id)
            id_list.append(id)
        GL_CAM_TO_CV_CAM = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

        # Transform camera poses to OpenGL convention
        glcam_in_obs = cam_in_obs @ GL_CAM_TO_CV_CAM
            
        # Load RGB images first
        rgb_images = [reader.get_color(i) for i in id_list]
        _, _, pcd_real_scale, _ = PointCloudProcessor.compute_scene_bounds(
                rgb_images, 
                [reader.get_depth(i) for i in id_list], 
                [reader.get_mask(i) for i in id_list], 
                glcam_in_obs, 
                K,
                translation_cvcam=np.zeros(3), 
                sc_factor=1, 
                eps=eps, 
                min_samples=1, 
                cluster=True
            )
        # Convert open3d point cloud to trimesh point cloud
        points = np.asarray(pcd_real_scale.points)
        colors = np.asarray(pcd_real_scale.colors)
        # Create trimesh point cloud with colors
        pcd = trimesh.PointCloud(vertices=points, colors=colors[:,::-1])
        scene.add_geometry(pcd)
    return scene
