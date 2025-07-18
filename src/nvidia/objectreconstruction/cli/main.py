"""
Command-line interface for NVIDIA 3D Object Reconstruction.

This module provides the main entry point for the CLI tool.
"""

import argparse
import logging
import sys
import os
import shutil
from pathlib import Path
import uuid
import yaml
import time
import torch

from nvidia.objectreconstruction.networks import NVBundleSDF
from nvidia.objectreconstruction.dataloader import ReconstructionDataLoader
from nvidia.objectreconstruction.utils.structures import dataclass_to_dict
from nvidia.objectreconstruction.networks.foundationstereo import run_depth_estimation
from nvidia.objectreconstruction.networks.sam2infer import run_mask_extraction
from nvidia.objectreconstruction.utils.preprocessing import setup_experiment_directory


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def validate_config_file(config_path: str) -> dict:
    """
    Load and validate configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Loaded configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
        ValueError: If config is missing required sections
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")
    
    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must contain a dictionary, got {type(config)}")
    
    # Validate required sections
    required_sections = ['bundletrack', 'nerf', 'roma', 'sam2', 'foundation_stereo', 'texture_bake']
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Configuration missing required sections: {missing_sections}")
    
    return config

def validate_data_path(data_path: str) -> Path:
    """
    Validate and return data path.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Path: Validated path object
        
    Raises:
        FileNotFoundError: If data path doesn't exist
        NotADirectoryError: If data path is not a directory
    """
    path = Path(data_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    if not path.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_path}")
    
    return path

def main():
    """Main CLI entry point with comprehensive error handling."""
    parser = argparse.ArgumentParser(
        description="NVIDIA 3D Object Reconstruction Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nvidia-3d-reconstruct --help
  nvidia-3d-reconstruct --config config.yaml --data-path /path/to/data
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="/workspace/3d-object-reconstruction/data/configs/base.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="/workspace/3d-object-reconstruction/data/samples/retail_item/",
        help="Path to input data directory"
    )
    parser.add_argument(
        "--output-path", 
        type=str, 
        default=f"/workspace/3d-object-reconstruction/data/output/{uuid.uuid4()}",
        help="Path to output directory for reconstruction results"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version="nvidia-3d-object-reconstruction 0.1.0"
    )
    
    # Parse arguments with error handling
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # argparse calls sys.exit on error, catch and re-raise
        return e.code if e.code is not None else 1
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        start_total = time.time()
        logger.info("NVIDIA 3D Object Reconstruction CLI")
        
        # Validate inputs
        logger.info("Validating configuration and inputs...")
        config = validate_config_file(args.config)
        exp_path = validate_data_path(args.data_path)
        
        # Create output directory
        output_path = Path(args.output_path)
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Output directory: {output_path}")
        
        # Setup configuration paths
        config['workdir'] = output_path
        config['bundletrack']['debug_dir'] = output_path / "bundletrack"
        config['nerf']['save_dir'] = output_path

        # Extract configuration sections
        bundletrack_config = config['bundletrack']
        nerf_config = config['nerf']
        roma_config = config['roma']
        sam2_config = config['sam2']
        foundation_stereo_config = config['foundation_stereo']
        texture_config = config['texture_bake']

        logger.info(f"Starting reconstruction pipeline for: {exp_path}")

        # Copy contents of input data path to output folder
        logger.info("Copying input data to output folder...")
        for item in exp_path.iterdir():
            if item.is_dir():
                shutil.copytree(item, output_path / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, output_path)
        logger.info("Input data copied successfully")

        # Step 1: Mask extraction
        logger.info("Step 1/4: Running mask extraction...")
        try:
            start_mask = time.time()
            run_mask_extraction(sam2_config, output_path, output_path / 'left', mask_path=output_path / 'masks')
            logger.info("Mask extraction completed successfully")
            time_mask = time.time() - start_mask
        except Exception as e:
            logger.error(f"Mask extraction failed: {e}")
            raise RuntimeError(f"Mask extraction step failed: {e}")

    
        # Step 2: Depth estimation
        logger.info("Step 2/4: Running depth estimation...")
        try:
            start_depth = time.time()
            response = run_depth_estimation(foundation_stereo_config, output_path, output_path / 'left', depth_path=output_path / 'depth')
            if not response:
                raise RuntimeError("Depth estimation failed")
            logger.info("Depth estimation completed successfully")
            time_depth = time.time() - start_depth
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            raise RuntimeError(f"Depth estimation step failed: {e}")

        # Step 3: Initialize tracker and datasets
        logger.info("Step 3/4: Initializing reconstruction components...")
        try:
            start_pipeline = time.time()
            tracker = NVBundleSDF(nerf_config, bundletrack_config, roma_config, texture_config, logger=logger)

            track_dataset = ReconstructionDataLoader(
                str(output_path), 
                config, 
                downscale=bundletrack_config['downscale'],
                min_resolution=bundletrack_config['min_resolution']
            )
            nerf_dataset = ReconstructionDataLoader(
                str(output_path), 
                config, 
                downscale=nerf_config['downscale'],
                min_resolution=nerf_config['min_resolution']
            )
            texture_dataset = ReconstructionDataLoader(
                str(output_path), 
                config, 
                downscale=texture_config['downscale'],
                min_resolution=texture_config['min_resolution']
            )
            logger.info("Components initialized successfully")
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize reconstruction components: {e}")

        # Step 4: Run reconstruction pipeline
        logger.info("Step 4/4: Running reconstruction pipeline...")
        
        # Object tracking
        logger.info("  4a. Running object tracking...")
        try:
            start_track = time.time()
            tracker.run_track(track_dataset)
            logger.info("  Object tracking completed")
            time_track = time.time() - start_track
        except Exception as e:
            logger.error(f"  Object tracking failed: {e}")
            raise RuntimeError(f"Object tracking failed: {e}")

        # SDF training
        logger.info("  4b. Running SDF training...")
        try:
            start_sdf = time.time()
            tracker.run_global_sdf(nerf_dataset)
            logger.info("  SDF training completed")
            time_sdf = time.time() - start_sdf
        except Exception as e:
            logger.error(f"  SDF training failed: {e}")
            raise RuntimeError(f"SDF training failed: {e}")

        # Texture baking
        logger.info("  4c. Running texture baking...")
        try:
            start_texture = time.time()
            tracker.run_texture_bake(texture_dataset)
            logger.info("  Texture baking completed")
            time_texture = time.time() - start_texture
        except Exception as e:
            logger.error(f"  Texture baking failed: {e}")
            raise RuntimeError(f"Texture baking failed: {e}")

        logger.info(f"Reconstruction completed successfully for {output_path}")
        time_pipeline = time.time() - start_pipeline
        times = {
            "total": time.time() - start_total,
            "mask": time_mask,
            "depth": time_depth,
            "pipeline": time_pipeline,
            "track": time_track,
            "sdf": time_sdf,
            "texture": time_texture,
            "gpu_name": torch.cuda.get_device_name(0),
        }
        with open(output_path / "run_time.yaml", "w") as f:
            yaml.dump(times, f)
        return 0

    except KeyboardInterrupt:
        logger.warning("Reconstruction interrupted by user (Ctrl+C)")
        return 130  # Standard exit code for SIGINT
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 2
        
    except NotADirectoryError as e:
        logger.error(f"Invalid directory: {e}")
        return 2
        
    except yaml.YAMLError as e:
        logger.error(f"Configuration file error: {e}")
        return 3
        
    except ValueError as e:
        logger.error(f"Configuration validation error: {e}")
        return 3
        
    except RuntimeError as e:
        logger.error(f"Processing error: {e}")
        return 4
        
    except MemoryError:
        logger.error("Out of memory - try reducing batch size or image resolution")
        return 5
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 