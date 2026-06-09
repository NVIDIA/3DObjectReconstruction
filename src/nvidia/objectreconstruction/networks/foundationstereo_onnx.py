# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
FoundationStereo Network Implementation for 3D Object Reconstruction.

This module provides a wrapper around the FoundationStereo model for stereo
depth estimation. It includes preprocessing utilities, model initialization,
and a high-level processor for batch depth map generation.

Classes:
    InputPadder: Utility class for padding images to required dimensions
    FoundationStereoNet: Wrapper for the FoundationStereo model
    FoundationStereoProcessor: High-level processor for stereo depth estimation

Functions:
    run_depth_estimation: Main entry point for depth estimation pipeline
"""
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Union, Optional
from omegaconf import OmegaConf
import torch
import onnxruntime

# ORT logs ScatterND and other kernel notices to stderr; not controllable via Python warnings.
# 0=verbose, 1=info, 2=warning, 3=error, 4=fatal — use 3 to hide warnings during inference.
onnxruntime.set_default_logger_severity(3)

from torchvision import transforms
import logging
logger = logging.getLogger(__name__)
import PIL
class InputPadder:
    """Pads images to handle neural network dimension requirements.

    Args:
        dims: Input dimensions (H, W)
        mode: Padding mode. Defaults to 'sintel'.
        divis_by: Ensure dimensions are divisible by this value.
            Defaults to 8.
        force_square: Force output to be square. Defaults to False.
    """

    def __init__(
        self,
        dims: tuple,
        mode: str = 'sintel',
        divis_by: int = 8,
        force_square: bool = False,
    ) -> None:
        """Initialize the padder with given dimensions and parameters."""
        self.ht, self.wd = dims[-2:]
        if force_square:
            max_side = max(self.ht, self.wd)
            pad_ht = ((max_side // divis_by) + 1) * divis_by - self.ht
            pad_wd = ((max_side // divis_by) + 1) * divis_by - self.wd
        else:
            pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
            pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by

        if mode == 'sintel':
            self._pad_left = pad_wd // 2
            self._pad_right = pad_wd - pad_wd // 2
            self._pad_top = pad_ht // 2
            self._pad_bottom = pad_ht - pad_ht // 2
        else:
            self._pad_left = pad_wd // 2
            self._pad_right = pad_wd - pad_wd // 2
            self._pad_top = 0
            self._pad_bottom = pad_ht

    def pad(self, *inputs: np.ndarray) -> list[np.ndarray]:
        """Pad input arrays.
        
        Args:
            *inputs: Input numpy arrays of shape [B, C, H, W]
            
        Returns:
            List of padded numpy arrays
            
        Raises:
            ValueError: If inputs are not 4D arrays
        """
        if not all(x.ndim == 4 for x in inputs):
            raise ValueError("All inputs must be 4D arrays")
            
        pad_width = (
            (0, 0),
            (0, 0),
            (self._pad_top, self._pad_bottom),
            (self._pad_left, self._pad_right),
        )
                    
        result = [np.pad(x, pad_width, mode='edge') for x in inputs]
        return result

    def unpad(self, x: np.ndarray) -> np.ndarray:
        """Remove padding from array.
        
        Args:
            x: Input numpy array of shape [B, C, H, W]
            
        Returns:
            Unpadded numpy array
            
        Raises:
            ValueError: If input is not a 4D array
        """
        if x.ndim != 4:
            raise ValueError("Input must be a 4D array")
            
        return x[
            ...,
            self._pad_top:x.shape[-2] - self._pad_bottom,
            self._pad_left:x.shape[-1] - self._pad_right,
        ]



def preprocess_each_image_pair(left_image, right_image, target_size):
    """
    load and preprocess each input image. Here we resize the images. 
    The user may choose to crop the input image instead.
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
  

    transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
    
    sample_transformed_left = transform(left_image) #avoid rgba
    sample_transformed_right = transform(right_image)
    return sample_transformed_left, sample_transformed_right


class FoundationStereoProcessorOnnx:
    """
    High-level processor for stereo depth estimation.

    This class manages the complete pipeline from loading stereo image pairs
    to generating depth maps. It handles image preprocessing, network inference,
    and depth conversion with configurable camera parameters.

    Attributes:
        config (Dict[str, Any]): Configuration parameters
        net (FoundationStereoNet): The stereo network model
        rgb_path (Path): Path to input RGB images
        output_path (Path): Path for output depth maps
        left_images (List[Path]): List of left stereo image paths
        intrinsic (np.ndarray): Camera intrinsic matrix (3x3)
        baseline (float): Baseline distance between cameras
    """

    def __init__(
        self,
        config: Dict[str, Any],
        rgb_path: Path,
        output_path: Path,
        logger: logging.Logger = logger
    ) -> None:
        """
        Initialize the stereo depth estimation processor.

        Args:
            config: Configuration dictionary containing:
                   - pth_path: Path to model weights
                   - intrinsic: Camera intrinsics matrix (3x3)
                   - baseline: Baseline distance between cameras
                   - scale: Resize scale factor for images
            rgb_path: Path to directory containing left stereo images
                     Supports png, jpg, jpeg formats
            output_path: Directory path where depth maps will be saved
                        as .npy files

        Raises:
            RuntimeError: If CUDA is not available
            FileNotFoundError: If rgb_path doesn't exist
        """
        self.config = config

        # Initialize and setup the stereo network
        providers = ['CUDAExecutionProvider','CPUExecutionProvider'] if config['device'] == 'cuda' else ['CPUExecutionProvider']
        session_options = onnxruntime.SessionOptions()
        session_options.log_severity_level = 3
        self.net = onnxruntime.InferenceSession(
            config['onnx_path'], session_options, providers=providers
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available")


        self.rgb_path = Path(rgb_path)
        self.output_path = Path(output_path)

        self.logger = logger

        if not self.rgb_path.exists():
            raise FileNotFoundError(f"RGB path does not exist: {rgb_path}")

        # Discover and sort left stereo images
        self._discover_images()

        # Extract camera parameters from configuration
        self._setup_camera_params()

        

    def _discover_images(self) -> None:
        """Discover and sort left stereo images from the input directory."""
        left_images = []
        supported_formats = ['*.png', '*.jpg', '*.jpeg']

        for ext in supported_formats:
            left_images.extend(self.rgb_path.glob(ext))

        self.left_images = sorted(left_images)

        if not self.left_images:
            self.logger.warning(f"No images found in {self.rgb_path}")

        self.logger.info(f"Found {len(self.left_images)} left images")

    def _setup_camera_params(self) -> None:
        """Extract and setup camera parameters from configuration."""
        self.intrinsic = np.array(self.config['intrinsic']).reshape(3, 3)
        # Scale intrinsics to match resized images
        self.intrinsic[:2] *= self.config['scale']
        self.baseline = self.config['baseline']

        self.logger.info(f"Camera baseline: {self.baseline}")
        self.logger.info(f"Image scale factor: {self.config['scale']}")

    def infer(
        self,
        left_input: Union[str, Path, np.ndarray],
        right_input: Union[str, Path, np.ndarray],
        return_disparity: bool = False
    ) -> np.ndarray:
        """
        Perform stereo depth inference on a single pair of images.

        Args:
            left_input: Path to left stereo image or numpy array
            right_input: Path to right stereo image or numpy array
            return_disparity: If True, returns disparity map instead of depth

        Returns:
            Depth map or disparity map as numpy array of shape [H, W]

        Raises:
            ValueError: If inputs are invalid or incompatible
            RuntimeError: If inference fails
        """
        try:
            # Load images - handle both file paths and numpy arrays
            if isinstance(left_input, (str, Path)):
                left = PIL.Image.open(str(left_input))
                right = PIL.Image.open(str(right_input))
            else:
                # Assume numpy arrays passed directly
                left = PIL.Image.fromarray(left_input)
                right = PIL.Image.fromarray(right_input)

            # Validate image shapes
            if np.array(left).shape != np.array(right).shape:
                raise ValueError(
                    f"Image shapes don't match: {np.array(left).shape} vs {np.array(right).shape}"
                )

            # Resize images according to configuration scale
            scale = self.config['scale']
            h,w = left.height, left.width
            H,W = int(h*scale), int(w*scale)
            if min(H,W) < self.config['onnx_input_min']:
                scale = self.config['onnx_input_min'] / min(h,w)
                self.config['scale'] = scale
                self._setup_camera_params()
            if max(H,W) > self.config['onnx_input_max']:
                scale = self.config['onnx_input_max'] / max(h,w)
                self.config['scale'] = scale
                self._setup_camera_params()
            H,W = int(h*scale), int(w*scale)

            

            left,right = preprocess_each_image_pair(left, right,target_size=(H,W))
            # Convert images to PyTorch tensors and move to GPU
            H, W = left.shape[-2:]

            img0=left.unsqueeze(0)[:,:3,:,:]#ensure rgb
            img1=right.unsqueeze(0)[:,:3,:,:]#ensure rgb

            # Pad images to be divisible by 32 for network processing
            padder = InputPadder(img0.shape, divis_by=32)
            img0, img1 = padder.pad(img0, img1)

            onnx_inputs = {"left_image": img0, "right_image": img1}
            disp = self.net.run(None, onnx_inputs)[0]

            # Remove padding and convert to numpy
            disp = padder.unpad(disp)
            disp = disp.reshape(H, W)
            
            if return_disparity:
                return disp

            # Convert disparity to metric depth using camera parameters
            # Depth = (focal_length * baseline) / disparity
            # Avoid division by zero
            disp_safe = np.where(disp > 0, disp, np.inf)
            depth = self.intrinsic[0, 0] * self.baseline / disp_safe

            return depth

        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Stereo inference failed: {e}") from e

    def run(self) -> None:
        """
        Process all stereo image pairs to generate depth maps.

        Main processing loop that:
        1. Loads left/right stereo image pairs
        2. Uses the infer() method for consistent processing
        3. Saves depth maps as numpy arrays

        For each left image, expects corresponding right image with 'left'
        replaced by 'right' in the filename.

        Output depth maps are saved as {image_name}.npy in the output directory.

        Raises:
            FileNotFoundError: If corresponding right image is not found
            RuntimeError: If processing fails
        """
        if not self.left_images:
            self.logger.warning("No left images found to process")
            return

        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

        successful_count = 0

        for left_path in tqdm(self.left_images, desc="Processing stereo pairs"):
            try:
                base_name = left_path.stem

                # Construct right image path
                right_path = left_path.parent.parent / 'right' / left_path.name.replace('left', 'right')

                if not right_path.exists():
                    self.logger.warning(f"Right image not found: {right_path}")
                    continue

                # Use the infer method for consistent processing
                depth = self.infer(
                    left_path, right_path, return_disparity=False
                )

                # Save depth map as numpy array
                output_file = self.output_path / f"{base_name}.npy"
                np.save(output_file, depth)
                successful_count += 1

            except Exception as e:
                self.logger.error(f"Failed to process {left_path}: {e}")
                continue

        self.logger.info(
            f"Successfully processed {successful_count}/{len(self.left_images)} with scale {self.config['scale']} with ONNX model "
            f"stereo pairs"
        )


def run_depth_estimation_onnx(
    config: Dict[str, Any],
    exp_path: Path,
    rgb_path: Path,
    depth_path: Optional[Path] = None,
    logger: logging.Logger = logger
) -> Optional[bool]:
    """
    Set up and run depth estimation pipeline.

    This function orchestrates the complete depth estimation process:
    1. Sets up output directory structure
    2. Checks if depth maps already exist
    3. Runs FoundationStereo processing if needed
    4. Returns success status

    Args:
        config: Configuration dictionary containing model and camera parameters
        exp_path: Path to experiment directory
        rgb_path: Path to RGB frames directory containing left/right images
        depth_path: Optional custom path for depth output (defaults to exp_path/depth)

    Returns:
        True if successful, False/None if failed

    Example:
        >>> config = {
        ...     'cfg_path': 'model_config.yaml',
        ...     'pth_path': 'weights.pth',
        ...     'intrinsic': [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        ...     'baseline': 0.1,
        ...     'scale': 0.5
        ... }
        >>> success = run_depth_estimation(config, exp_path, rgb_path)
    """
    # Setup depth output directory
    if depth_path is None:
        depth_path = exp_path / 'depth'
    depth_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Depth estimation directory: {depth_path}")

    try:
        # Check if depth images already exist (either all .npy or all .png)
        depth_images_npy = list(depth_path.glob('*.npy'))
        depth_images_png = list(depth_path.glob('*.png'))
        rgb_images = list(rgb_path.glob('*.png'))

        # Check if we have sufficient depth images in either format
        if (depth_images_npy and len(depth_images_npy) >= len(rgb_images)) or \
           (depth_images_png and len(depth_images_png) >= len(rgb_images)):
            logger.info("Depth images already exist, skipping depth estimation")
            return True

        

        # Run depth estimation
        logger.info("Running depth estimation...")

        args =OmegaConf.create(config)

        # Initialize and run processor
        processor = FoundationStereoProcessorOnnx(args, rgb_path, depth_path)
        processor.run()

        logger.info("Depth estimation completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error running depth estimation: {e}")
        return None
