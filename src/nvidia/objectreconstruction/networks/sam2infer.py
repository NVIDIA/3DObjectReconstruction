"""
SAM2 Inference Module for 3D Object Reconstruction.

This module provides functionality for running SAM2 (Segment Anything Model 2)
inference on image sequences for mask generation. It includes utilities for
processing single images, directories of images, and video sequences.

The module handles PNG image formats and provides compatibility with the
original SAM2 video processing pipeline.
"""

import os
import glob
import logging
import numpy as np
import torch
import warnings
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

from sam2.build_sam import build_sam2_video_predictor
from sam2.utils import misc

# Add monkey patch for torch.load to ensure compatibility with older checkpoints
original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    """
    Patch torch.load to handle compatibility issues with older checkpoints.

    This function removes the weights_only=True parameter if present to ensure
    compatibility with older PyTorch checkpoints that don't support this flag.

    Args:
        *args: Positional arguments passed to torch.load
        **kwargs: Keyword arguments passed to torch.load

    Returns:
        The result of torch.load with modified parameters
    """
    if 'weights_only' in kwargs and kwargs['weights_only'] is True:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)


torch.load = patched_torch_load

# Configure logging
logger = logging.getLogger(__name__)

# Select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"using device: {device}")

# Configure device-specific settings
if device.type == "cuda":
    # Use bfloat16 for better performance on CUDA
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # Enable tfloat32 for Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA "
        "and might give numerically different outputs and sometimes degraded "
        "performance on MPS. See e.g. "
        "https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


def png_compatible_load_video_frames(
    video_path: str,
    image_size: int = 1024,
    offload_video_to_cpu: bool = False,
    img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    async_loading_frames: bool = False,
    compute_device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, int, int]:
    """
    Load video frames from a directory of image files (JPEG and PNG).

    This is a drop-in replacement for misc.load_video_frames that supports
    PNG format images in addition to JPEG.

    Args:
        video_path: Path to the directory containing image files
        image_size: Target size for resizing images
        offload_video_to_cpu: Whether to keep images on CPU
        img_mean: RGB mean values for normalization
        img_std: RGB standard deviation values for normalization
        async_loading_frames: Whether to load frames asynchronously (unused)
        compute_device: Device to load images to

    Returns:
        Tuple containing:
            - images: Tensor of shape (N, 3, H, W) containing loaded images
            - video_height: Original height of the video frames
            - video_width: Original width of the video frames

    Raises:
        FileNotFoundError: If video_path doesn't exist
        RuntimeError: If no images found or unsupported image format
        NotImplementedError: If video_path is not a directory
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file or folder not found: {video_path}")

    if not os.path.isdir(video_path):
        warnings.warn(
            "This implementation only supports directories of image files, "
            "not video files."
        )
        raise NotImplementedError(
            "Only image frames are supported. For video files, "
            "extract frames to a directory first."
        )

    img_folder = video_path
    # Get all supported image files
    frame_names = []
    for ext in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]:
        frame_names.extend([
            p for p in os.listdir(img_folder) if p.endswith(ext)
        ])

    if not frame_names:
        raise RuntimeError(f"No images found in {img_folder}")

    # Sort the filenames
    try:
        # Try to sort based on filename pattern (assuming frame_xxxx format)
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0][4:]))
    except (ValueError, IndexError):
        # Fallback to regular sorting
        frame_names.sort()

    # Load the images
    img_paths = [os.path.join(img_folder, frame_name) for frame_name in frame_names]
    img_mean_tensor = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std_tensor = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    # Load the first image to get dimensions
    first_img = Image.open(img_paths[0])
    video_width, video_height = first_img.size

    # Load all images
    num_frames = len(img_paths)
    images = torch.zeros(
        num_frames, 3, image_size, image_size, dtype=torch.float32
    )

    for n, img_path in enumerate(tqdm(img_paths, desc="Loading frames")):
        img_pil = Image.open(img_path).convert("RGB").resize(
            (image_size, image_size)
        )
        img_np = np.array(img_pil)
        if img_np.dtype == np.uint8:
            img_np = img_np / 255.0
        else:
            raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
        images[n] = torch.from_numpy(img_np).permute(2, 0, 1)

    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean_tensor = img_mean_tensor.to(compute_device)
        img_std_tensor = img_std_tensor.to(compute_device)

    # Normalize by mean and std
    images -= img_mean_tensor
    images /= img_std_tensor

    return images, video_height, video_width


def preprocess_single_image(
    image_path: str,
    image_size: int = 1024,
    img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    compute_device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, int, int]:
    """
    Preprocess a single image for SAM2 inference.

    Args:
        image_path: Path to the image file
        image_size: Size to resize the image to
        img_mean: RGB mean values for normalization
        img_std: RGB standard deviation values for normalization
        compute_device: Device to load the image to

    Returns:
        Tuple containing:
            - preprocessed_image: Preprocessed image tensor of shape (1, 3, H, W)
            - height: Original image height
            - width: Original image width

    Raises:
        FileNotFoundError: If image file doesn't exist
        RuntimeError: If image has unsupported dtype
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load and preprocess the image
    img_pil = Image.open(image_path).convert("RGB")
    width, height = img_pil.size

    # Resize and convert to numpy array
    img_pil = img_pil.resize((image_size, image_size))
    img_np = np.array(img_pil)

    if img_np.dtype == np.uint8:
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {image_path}")

    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Create normalization tensors
    img_mean_tensor = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std_tensor = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    # Move to device
    img_tensor = img_tensor.to(compute_device)
    img_mean_tensor = img_mean_tensor.to(compute_device)
    img_std_tensor = img_std_tensor.to(compute_device)

    # Normalize
    img_tensor -= img_mean_tensor
    img_tensor /= img_std_tensor

    return img_tensor, height, width


def segment_image_with_bbox(
    image_path: str,
    bbox: Union[list, np.ndarray],
    checkpoint_path: str = "/sam2/checkpoints/sam2.1_hiera_large.pt",
    model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    output_path: Optional[str] = None,
    image_size: int = 1024,
    device: torch.device = torch.device("cuda")
) -> np.ndarray:
    """
    Segment an object in a single image using a 2D bounding box.

    Args:
        image_path: Path to the input image
        bbox: 2D bounding box in format [x1, y1, x2, y2]
        checkpoint_path: Path to the SAM2 checkpoint
        model_config: Path to the model configuration file
        output_path: Optional path to save the output mask
        image_size: Size to resize the image to during processing
        device: Device to run inference on

    Returns:
        Binary mask of the segmented object as numpy array

    Raises:
        Various exceptions from SAM2 model initialization and inference
    """
    # Build the SAM2 predictor
    predictor = build_sam2_video_predictor(
        model_config, checkpoint_path, device=device
    )

    # Preprocess the image
    image_tensor, original_height, original_width = preprocess_single_image(
        image_path,
        image_size=image_size,
        compute_device=device
    )

    # Convert bbox to numpy array if it's not already
    bbox = np.array(bbox, dtype=np.float32)

    # Calculate the center point of the bounding box for positive click
    center_point = np.array([
        [bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1] + (bbox[3] - bbox[1]) / 2]
    ], dtype=np.float32)

    # Set positive label
    labels = np.array([1], np.int32)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Initialize the inference state
        tmp_dir = os.path.dirname(image_path)
        inference_state = predictor.init_state(video_path=tmp_dir)
        predictor.reset_state(inference_state)

        # Set frame index and object ID
        frame_idx = 0
        obj_id = 1

        # Add the bounding box
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=bbox,
            labels=labels,
            points=center_point,
        )

        # Get the mask
        mask = (out_mask_logits[0] > 0.0).cpu().numpy()

        # Save the mask if output path is provided
        if output_path:
            mask_image = mask.astype(np.uint8) * 255
            mask_image = Image.fromarray(mask_image[0])
            mask_image.save(output_path)
            logger.debug(f"Saved mask to {output_path}")

        return mask


def process_directory_masks(
    rgb_path: str,
    mask_path: str,
    bbox: Optional[Union[list, np.ndarray]] = None,
    checkpoint_path: str = "/sam2/checkpoints/sam2.1_hiera_large.pt",
    model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: torch.device = torch.device("cuda")
) -> None:
    """
    Process all images in a directory and generate masks using a bounding box.

    The bounding box is applied to the first frame and then propagated through
    all subsequent frames using SAM2's video tracking capabilities.

    Args:
        rgb_path: Path to the directory containing RGB images
        mask_path: Path to save the generated masks
        bbox: Bounding box in format [x1, y1, x2, y2]. If None, will be
              calculated as central 80% of the first frame
        checkpoint_path: Path to the SAM2 checkpoint
        model_config: Path to the model configuration file
        device: Device to run inference on

    Raises:
        Various exceptions from image loading and SAM2 inference
    """
    # Create mask directory if it doesn't exist
    os.makedirs(mask_path, exist_ok=True)

    # Check if masks already exist
    if os.path.exists(mask_path) and any(os.listdir(mask_path)):
        logger.info("Masks already extracted")
        return

    # Get all image files in the directory
    image_files = sorted(glob.glob(os.path.join(rgb_path, "*.png")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(rgb_path, "*.jpg")))

    if not image_files:
        logger.error("No image files found in RGB frames directory")
        return

    # If bbox is None, create a default one
    if bbox is None:
        import cv2

        # Read the first image to get dimensions
        first_image = cv2.imread(image_files[0])
        height, width = first_image.shape[:2]

        # Create a default bounding box (central 80% of the image)
        margin_x = int(width * 0.1)
        margin_y = int(height * 0.1)
        bbox = [margin_x, margin_y, width - margin_x, height - margin_y]

        logger.info(f"Using default bounding box: {bbox}")

    # Convert bbox to numpy array
    bbox = np.array(bbox, dtype=np.float32)

    # Build the SAM2 predictor
    predictor = build_sam2_video_predictor(
        model_config, checkpoint_path, device=device
    )

    logger.info(f"Processing {len(image_files)} frames for mask extraction...")

    # Process all frames with the same bounding box
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Initialize the inference state
        inference_state = predictor.init_state(video_path=rgb_path)
        predictor.reset_state(inference_state)

        # Set frame index and object ID
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object

        # Calculate the center point of the bounding box for positive click
        points = np.array([
            [bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1] + (bbox[3] - bbox[1]) / 2]
        ], dtype=np.float32)

        # Set positive label
        labels = np.array([1], np.int32)

        # Add the bounding box
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=bbox,
            labels=labels,
            points=points,
        )

        # Run propagation throughout the video and collect the results
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in (
            predictor.propagate_in_video(inference_state)
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Get frame names
        frame_names = [os.path.basename(image_file) for image_file in image_files]

        # Render the segmentation results
        for out_frame_idx in range(0, len(frame_names)):
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                mask_image = out_mask.astype(np.uint8) * 255
                mask_image = Image.fromarray(mask_image[0])
                mask_output_path = os.path.join(
                    mask_path, frame_names[out_frame_idx]
                )
                mask_image.save(mask_output_path)
                logger.debug(f"Saved mask for frame {out_frame_idx}")

    logger.info(f"Mask extraction completed. Masks saved to {mask_path}")


# Replace the original function with PNG-compatible version
misc.load_video_frames = png_compatible_load_video_frames


class Sam2Infer:
    """
    SAM2 inference wrapper class.

    This class provides a simple interface for running SAM2 mask extraction
    on directories of images using configuration parameters.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the SAM2 inference wrapper.

        Args:
            config: Configuration dictionary containing:
                - checkpoint_path: Path to SAM2 model checkpoint
                - model_config: Path to model configuration file
                - bbox: Bounding box for segmentation [x1, y1, x2, y2]
                - device: Device to run inference on
        """
        print(config)
        self.checkpoint_path = config['checkpoint_path']
        self.model_config = config['model_config']
        self.bbox = config['bbox']
        self.device = config['device']

    def run(self, rgb_path: str, mask_path: str) -> None:
        """
        Run mask extraction on a directory of images.

        Args:
            rgb_path: Path to directory containing RGB images
            mask_path: Path to directory where masks will be saved
        """
        process_directory_masks(
            rgb_path=rgb_path,
            mask_path=mask_path,
            bbox=self.bbox,
            checkpoint_path=self.checkpoint_path,
            model_config=self.model_config
        )


def run_mask_extraction(
    config: Dict[str, Any],
    exp_path: Path,
    rgb_path: Path,
    mask_path: Path
) -> bool:
    """
    Set up and run mask extraction with error handling.

    Args:
        config: Configuration dictionary for SAM2 inference
        exp_path: Path to experiment directory
        rgb_path: Path to RGB frames directory
        mask_path: Path where masks will be saved

    Returns:
        True if mask extraction was successful, False otherwise
    """
    # Create parent directories if they don't exist
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.mkdir(exist_ok=True)
    logger.info(f"Mask extraction directory: {mask_path}")

    # Check if mask images already exist
    mask_images = list(mask_path.glob('*.png'))
    rgb_images = list(rgb_path.glob('*.png'))
    if mask_images and len(mask_images) == len(rgb_images):
        logger.info("Mask images already exist, skipping mask extraction")
        return True

    # Run mask extraction
    logger.info("Running mask extraction...")
    sam2_infer = Sam2Infer(config)

    try:
        sam2_infer.run(str(rgb_path), str(mask_path))
        logger.info("Mask extraction completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error running mask extraction: {e}")
        return False


if __name__ == "__main__":
    """Example usage of the SAM2 inference module."""
    sam2_checkpoint = (
        "/workspace/3d-object-reconstruction/data/weights/sam2/"
        "sam2.1_hiera_large.pt"
    )
    model_cfg = (
        "/workspace/3d-object-reconstruction/data/weights/sam2/"
        "sam2.1_hiera_l.yaml"
    )

    # Example for processing a video directory with bounding box
    video_dir = "/workspace/3d-object-reconstruction/data/samples/retail_item/left/"
    output_dir = "/workspace/3d-object-reconstruction/data/samples/retail_item/masks/"

    # Define a bounding box [x1, y1, x2, y2]
    bbox = [1144, 627, 2227, 2232]

    # Process the directory
    process_directory_masks(
        rgb_path=video_dir,
        mask_path=output_dir,
        bbox=bbox,
        checkpoint_path=sam2_checkpoint,
        model_config=model_cfg
    )
