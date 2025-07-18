import os
import cv2
import sys
import yaml
import logging

import numpy as np
import open3d as o3d
import torch.nn.functional as F
from tqdm import tqdm

from omegaconf import OmegaConf
from pathlib import Path
from typing import Dict, Any

from ..networks.foundationstereo import FoundationStereoProcessor
from ..networks.sam2infer import Sam2Infer

logger = logging.getLogger("preprocessing")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration file.
    
    Args:
        config_path: Path to the configuration YAML file.
    
    Returns:
        Dict containing configuration parameters.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid.
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)

def setup_experiment_directory(config: Dict[str, Any]) -> tuple[Path, Path]:
    """
    Create and validate experiment directory structure.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        Tuple of (experiment_path, rgb_path)
    """
    exp_path = Path(config['base_path']['base_folder'])
    exp_path.mkdir(exist_ok=True)
    rgb_path = Path(config['base_path']['image_folder'])
    rgb_path.mkdir(exist_ok=True)
    logger.info(f"Using experiment directory: {exp_path}")

    return exp_path, rgb_path

def process_video_frames(config: Dict[str, Any], exp_path: Path, rgb_path: Path) -> None:
    """
    Extract frames from input video if not already processed.
    
    Args:
        config: Configuration dictionary.
        exp_path: Path to experiment directory.
        rgb_path: Path to RGB frames directory.
    """
    if rgb_path.exists() and any(rgb_path.iterdir()):
        logger.info("RGB frames already extracted")
        return

    rgb_path.mkdir(exist_ok=True)
    logger.info("Extracting video frames...")
    read_video(config['video']['input_path'], str(exp_path), config)


def depth2xyzmap(depth:np.ndarray, K, uvs:np.ndarray=None, zmin=0.1):
    invalid_mask = (depth < zmin)
    H, W = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    if invalid_mask.any():
        xyz_map[invalid_mask] = 0
    return xyz_map

def toOpen3dCloud(points, colors=None, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def read_video(input_video_path, base_folder, config=None):
    """
    Read stereo video and split it into left and right frames
    Save the frames in the respective folders
    """
    
    # Validate input video path
    if not os.path.exists(input_video_path):
        logger.error(f"Input video file not found: {input_video_path}")
        return False
    
    # Try different backends in order of preference
    cap = None
    backends_to_try = [
        (cv2.CAP_FFMPEG, "FFmpeg"),
        (cv2.CAP_GSTREAMER, "GStreamer"), 
        (cv2.CAP_ANY, "Default")
    ]
    
    for backend_id, backend_name in backends_to_try:
        logger.info(f"Trying {backend_name} backend...")
        cap = cv2.VideoCapture(input_video_path, backend_id)
        if cap.isOpened():
            logger.info(f"Successfully opened with {backend_name} backend")
            break
        else:
            logger.warning(f"{backend_name} backend failed")
            if cap:
                cap.release()
    
    # Check if any backend worked
    if not cap or not cap.isOpened():
        logger.error(f"Failed to open video file with any backend: {input_video_path}")
        logger.error("This could be due to missing codec support or corrupted file")
        return False
    
    # Get video properties for validation
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video properties: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
    
    if total_frames == 0:
        logger.error("Video contains no frames or frame count could not be determined")
        cap.release()
        return False

    # Set default step if config not provided
    step = 1
    if config and 'video' in config and 'step' in config['video']:
        step = config['video']['step']
    
    logger.info(f"Processing every {step} frame(s)")

    # Read the video and split it into frames
    ret = True
    count = 0
    frames_saved = 0
    
    left_path = os.path.join(base_folder, 'left')
    if not os.path.exists(left_path):
        os.makedirs(left_path)
    
    right_path = os.path.join(base_folder, 'right')
    if not os.path.exists(right_path):
        os.makedirs(right_path)

    # Create progress bar
    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frames")
    
    while ret:
        ret, image = cap.read()

        if not ret:
            break
        
        if image is None:
            logger.warning(f"Frame {count} is None, skipping")
            count += 1
            pbar.update(1)
            continue
            
        h, w = image.shape[:2]
        
        # Validate that we have a stereo image (should be twice as wide)
        if w < 100:  # Minimum reasonable width
            logger.error(f"Image width {w} is too small for stereo video")
            break
            
        # separate the stereo video into left and right
        left = image[:, :w//2]
        right = image[:, w//2:]

        # Save frames based on step interval
        if count % step == 0:
            left_save_path = os.path.join(left_path, '{}.png'.format(str(frames_saved).zfill(6)))
            right_save_path = os.path.join(right_path, '{}.png'.format(str(frames_saved).zfill(6)))
            
            success_left = cv2.imwrite(left_save_path, left)
            success_right = cv2.imwrite(right_save_path, right)
            
            if success_left and success_right:
                frames_saved += 1
            else:
                logger.error(f"Failed to save frame {count}")
        
        count += 1
        
        # Update progress bar with current status
        pbar.set_postfix({
            'saved': frames_saved,
            'step': f'1/{step}' if step > 1 else 'all'
        })
        pbar.update(1)
    
    pbar.close()
    
    cap.release()
    cv2.destroyAllWindows()
    
    if frames_saved > 0:
        logger.info(f"Successfully saved {frames_saved} frame pairs from {count} total frames")
        return True
    else:
        logger.error(f"No frames were saved! Processed {count} frames but none could be saved.")
        return False 