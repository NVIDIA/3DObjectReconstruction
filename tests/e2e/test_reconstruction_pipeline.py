"""
End-to-end test for the 3D reconstruction pipeline using the retail_item sample dataset.

This test validates the complete reconstruction workflow from stereo images to 3D mesh.
"""
import os
import sys
import subprocess
import pytest
from pathlib import Path
import yaml
import time


class TestReconstructionPipeline:
    """Test suite for full reconstruction pipeline."""
    
    def test_cli_help(self):
        """Test that CLI help command works."""
        result = subprocess.run(
            ["nvidia-3d-reconstruct", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "CLI help command failed"
        assert "NVIDIA 3D Object Reconstruction" in result.stdout
    
    def test_cli_version(self):
        """Test that CLI version command works."""
        result = subprocess.run(
            ["nvidia-3d-reconstruct", "--version"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "CLI version command failed"
        assert "0.1.0" in result.stdout or "0.1.0" in result.stderr
    
    def test_config_file_validation(self, test_config_path):
        """Test that the base config file is valid."""
        assert os.path.exists(test_config_path), f"Config file not found: {test_config_path}"
        
        with open(test_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['bundletrack', 'nerf', 'roma', 'sam2', 'foundation_stereo', 'texture_bake']
        for section in required_sections:
            assert section in config, f"Config missing required section: {section}"
    
    def test_sample_data_exists(self, test_data_path):
        """Test that sample data directory structure is valid."""
        data_path = Path(test_data_path)
        
        # Check for left and right directories
        assert (data_path / "left").exists(), "Left image directory not found"
        assert (data_path / "right").exists(), "Right image directory not found"
        
        # Check that there are image files
        left_images = list((data_path / "left").glob("*.png"))
        right_images = list((data_path / "right").glob("*.png"))
        
        assert len(left_images) > 0, "No left images found"
        assert len(right_images) > 0, "No right images found"
        assert len(left_images) == len(right_images), "Mismatch in number of left/right images"
    
    @pytest.mark.slow
    @pytest.mark.e2e
    def test_full_reconstruction_pipeline(
        self, 
        test_config_path, 
        test_data_path, 
        test_output_dir,
        skip_if_no_gpu
    ):
        """
        Test the complete reconstruction pipeline on sample data.
        
        This is a comprehensive test that runs the full pipeline:
        1. Depth estimation (FoundationStereo)
        2. Mask extraction (SAM2)
        3. Pose estimation (BundleTrack)
        4. SDF training (Neural Implicit Surface)
        5. Texture baking
        
        Note: This test requires GPU and takes ~30 minutes to complete.
        """
        print("\n" + "="*60)
        print("Starting Full Reconstruction Pipeline Test")
        print("="*60)
        print(f"Config: {test_config_path}")
        print(f"Data: {test_data_path}")
        print(f"Output: {test_output_dir}")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        # Run the reconstruction CLI
        cmd = [
            "nvidia-3d-reconstruct",
            "--config", test_config_path,
            "--data-path", test_data_path,
            "--output-path", str(test_output_dir),
            "--verbose"
        ]
        
        print(f"Running command: {' '.join(cmd)}\n")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("STDOUT:")
        print("="*60)
        print(result.stdout)
        
        if result.stderr:
            print("\n" + "="*60)
            print("STDERR:")
            print("="*60)
            print(result.stderr)
        
        print("\n" + "="*60)
        print(f"Test completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print("="*60 + "\n")
        
        # Check that the command succeeded
        assert result.returncode == 0, f"Reconstruction pipeline failed with return code {result.returncode}"
        
        # Validate output artifacts
        output_path = Path(test_output_dir)
        
        # Check for expected output directories
        expected_dirs = ["masks", "depth", "bundletrack"]
        for dir_name in expected_dirs:
            dir_path = output_path / dir_name
            assert dir_path.exists(), f"Expected output directory not found: {dir_name}"
        
        # Check for mask files
        mask_files = list((output_path / "masks").glob("*.png"))
        assert len(mask_files) > 0, "No mask files generated"
        print(f"✓ Generated {len(mask_files)} mask files")
        
        # Check for depth files
        depth_files = list((output_path / "depth").glob("*.png"))
        assert len(depth_files) > 0, "No depth files generated"
        print(f"✓ Generated {len(depth_files)} depth files")
        
        # Check for final mesh output (OBJ file)
        obj_files = list(output_path.glob("**/*.obj"))
        assert len(obj_files) > 0, "No mesh OBJ file generated"
        print(f"✓ Generated {len(obj_files)} mesh file(s)")
        
        # Check for texture file
        texture_files = list(output_path.glob("**/*.png"))
        # Filter out mask and depth files
        texture_files = [f for f in texture_files 
                        if "masks" not in str(f) and "depth" not in str(f)]
        assert len(texture_files) > 0, "No texture file generated"
        print(f"✓ Generated texture files")
        
        # Check for timing info
        timing_file = output_path / "run_time.yaml"
        if timing_file.exists():
            with open(timing_file, 'r') as f:
                timing_data = yaml.safe_load(f)
            print("\nPipeline Timing:")
            for key, value in timing_data.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.2f}s")
                else:
                    print(f"  {key}: {value}")
        
        print("\n" + "="*60)
        print("✓ Full reconstruction pipeline test PASSED")
        print("="*60)
    
    @pytest.mark.e2e
    def test_cli_with_invalid_config(self, test_data_path, test_output_dir):
        """Test that CLI fails gracefully with invalid config."""
        result = subprocess.run(
            [
                "nvidia-3d-reconstruct",
                "--config", "/nonexistent/config.yaml",
                "--data-path", test_data_path,
                "--output-path", str(test_output_dir)
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0, "CLI should fail with invalid config"
        assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()
    
    @pytest.mark.e2e
    def test_cli_with_invalid_data_path(self, test_config_path, test_output_dir):
        """Test that CLI fails gracefully with invalid data path."""
        result = subprocess.run(
            [
                "nvidia-3d-reconstruct",
                "--config", test_config_path,
                "--data-path", "/nonexistent/data/path",
                "--output-path", str(test_output_dir)
            ],
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0, "CLI should fail with invalid data path"
        assert "not exist" in result.stderr.lower() or "not exist" in result.stdout.lower()


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v", "-s"])
