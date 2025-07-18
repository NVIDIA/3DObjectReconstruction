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

import cv2
import torch
import imageio
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append('/FoundationStereo/core')
from foundation_stereo import FoundationStereo
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional
from loguru import logger
from omegaconf import OmegaConf


class InputPadder:
    """
    Utility class for padding images to dimensions divisible by a given factor.

    This class ensures that input images have dimensions that are compatible
    with neural network architectures that require specific divisibility
    constraints (e.g., divisible by 8 or 32).

    Attributes:
        ht (int): Original image height
        wd (int): Original image width
        _pad (List[int]): Padding values [left, right, top, bottom]
    """

    def __init__(
        self,
        dims: Tuple[int, ...],
        mode: str = 'sintel',
        divis_by: int = 8,
        force_square: bool = False
    ) -> None:
        """
        Initialize the InputPadder.

        Args:
            dims: Image dimensions tuple (..., H, W)
            mode: Padding mode, either 'sintel' or other
            divis_by: Factor by which dimensions should be divisible
            force_square: If True, pad to make image square

        Example:
            >>> padder = InputPadder((1, 3, 480, 640), divis_by=32)
            >>> padded_imgs = padder.pad(img1, img2)
        """
        self.ht, self.wd = dims[-2:]

        if force_square:
            max_side = max(self.ht, self.wd)
            pad_ht = ((max_side // divis_by) + 1) * divis_by - self.ht
            pad_wd = ((max_side // divis_by) + 1) * divis_by - self.wd
        else:
            pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
            pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by

        if mode == 'sintel':
            self._pad = [
                pad_wd // 2, pad_wd - pad_wd // 2,
                pad_ht // 2, pad_ht - pad_ht // 2
            ]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply padding to input tensors.

        Args:
            *inputs: Variable number of 4D tensors to pad

        Returns:
            List of padded tensors with same order as inputs

        Raises:
            AssertionError: If any input tensor is not 4-dimensional
        """
        assert all((x.ndim == 4) for x in inputs), \
            "All inputs must be 4-dimensional tensors"
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Remove padding from a tensor.

        Args:
            x: 4D tensor to unpad

        Returns:
            Tensor with padding removed

        Raises:
            AssertionError: If input tensor is not 4-dimensional
        """
        assert x.ndim == 4, "Input must be a 4-dimensional tensor"
        ht, wd = x.shape[-2:]
        c = [
            self._pad[2], ht - self._pad[3],
            self._pad[0], wd - self._pad[1]
        ]
        return x[..., c[0]:c[1], c[2]:c[3]]


class FoundationStereoNet(FoundationStereo):
    """
    Wrapper class for FoundationStereo network.

    This class extends the base FoundationStereo class with additional
    functionality for configuration management, weight loading, and
    simplified inference interface for stereo depth estimation.

    Attributes:
        config (Dict[str, Any]): Model configuration parameters
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the FoundationStereo network.

        Args:
            config: Configuration dictionary containing model parameters
                   including architecture settings and hyperparameters

        Example:
            >>> config = {'hidden_dims': [128, 128], 'corr_levels': 4}
            >>> model = FoundationStereoNet(config)
        """
        super().__init__(config)
        self.config = config

    def load_weights(self) -> None:
        """
        Load pre-trained weights from checkpoint file.

        The checkpoint file path should be specified in config['pth_path'].
        The checkpoint is expected to contain a 'model' key with the
        state dictionary.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            KeyError: If checkpoint doesn't contain 'model' key
            RuntimeError: If state dict loading fails
        """
        try:
            ckpt = torch.load(self.config['pth_path'], weights_only=False)
            self.load_state_dict(ckpt['model'])
            logger.info(f"Loaded weights from {self.config['pth_path']}")
        except FileNotFoundError as e:
            logger.error(f"Checkpoint file not found: {self.config['pth_path']}")
            raise e
        except KeyError as e:
            logger.error(f"Checkpoint missing 'model' key: {e}")
            raise e

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform forward pass for stereo depth estimation.

        Args:
            left: Left stereo image tensor of shape [B, C, H, W]
            right: Right stereo image tensor of shape [B, C, H, W]

        Returns:
            Disparity map tensor of shape [B, 1, H, W] representing
            pixel disparities between left and right images
        """
        return super().forward(left, right, iters=32, test_mode=True)


class FoundationStereoProcessor:
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
        output_path: Path
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
        self.net = FoundationStereoNet(config)
        self.net.load_weights()

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available")

        self.net.cuda()  # Move model to GPU
        self.net.eval()  # Set to evaluation mode

        self.rgb_path = Path(rgb_path)
        self.output_path = Path(output_path)

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
            logger.warning(f"No images found in {self.rgb_path}")

        logger.info(f"Found {len(self.left_images)} left images")

    def _setup_camera_params(self) -> None:
        """Extract and setup camera parameters from configuration."""
        self.intrinsic = np.array(self.config['intrinsic']).reshape(3, 3)
        # Scale intrinsics to match resized images
        self.intrinsic[:2] *= self.config['scale']
        self.baseline = self.config['baseline']

        logger.info(f"Camera baseline: {self.baseline}")
        logger.info(f"Image scale factor: {self.config['scale']}")

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
                left = imageio.imread(str(left_input))
                right = imageio.imread(str(right_input))
            else:
                # Assume numpy arrays passed directly
                left = left_input
                right = right_input

            # Validate image shapes
            if left.shape != right.shape:
                raise ValueError(
                    f"Image shapes don't match: {left.shape} vs {right.shape}"
                )

            # Resize images according to configuration scale
            scale = self.config['scale']
            left = cv2.resize(
                left, fx=scale, fy=scale, dsize=None,
                interpolation=cv2.INTER_LINEAR
            )
            right = cv2.resize(
                right, fx=scale, fy=scale, dsize=None,
                interpolation=cv2.INTER_LINEAR
            )
            H, W = left.shape[:2]

            # Convert images to PyTorch tensors and move to GPU
            img0 = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
            img1 = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)

            # Pad images to be divisible by 32 for network processing
            padder = InputPadder(img0.shape, divis_by=32, force_square=False)
            img0, img1 = padder.pad(img0, img1)

            # Run stereo matching inference
            with torch.no_grad():
                disp = self.net(img0, img1)

            # Remove padding and convert to numpy
            disp = padder.unpad(disp.float())
            disp = disp.data.cpu().numpy().reshape(H, W)

            if return_disparity:
                return disp

            # Convert disparity to metric depth using camera parameters
            # Depth = (focal_length * baseline) / disparity
            # Avoid division by zero
            disp_safe = np.where(disp > 0, disp, np.inf)
            depth = self.intrinsic[0, 0] * self.baseline / disp_safe

            return depth

        except Exception as e:
            logger.error(f"Inference failed: {e}")
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
            logger.warning("No left images found to process")
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
                    logger.warning(f"Right image not found: {right_path}")
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
                logger.error(f"Failed to process {left_path}: {e}")
                continue

        logger.info(
            f"Successfully processed {successful_count}/{len(self.left_images)} "
            f"stereo pairs"
        )


def run_depth_estimation(
    config: Dict[str, Any],
    exp_path: Path,
    rgb_path: Path,
    depth_path: Optional[Path] = None
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

        # Load additional model configuration
        cfg_model = OmegaConf.load(config['cfg_path'])
        args = OmegaConf.merge(OmegaConf.create(config), cfg_model)

        # Initialize and run processor
        processor = FoundationStereoProcessor(args, rgb_path, depth_path)
        processor.run()

        logger.info("Depth estimation completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error running depth estimation: {e}")
        return None
