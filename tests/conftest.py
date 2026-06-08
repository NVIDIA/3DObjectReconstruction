"""
Pytest configuration and shared fixtures for testing.
"""
import os
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def repo_root():
    """Return the repository root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_config_path(repo_root):
    """Return path to the test configuration file."""
    config_path = repo_root / "data" / "configs" / "base.yaml"
    assert config_path.exists(), f"Test config not found at {config_path}"
    return str(config_path)


@pytest.fixture(scope="session")
def test_data_path(repo_root):
    """Return path to the test data (retail_item sample)."""
    data_path = repo_root / "data" / "samples" / "retail_item"
    assert data_path.exists(), f"Test data not found at {data_path}"
    return str(data_path)


@pytest.fixture(scope="session")
def test_output_dir(repo_root, tmp_path_factory):
    """Create and return a temporary output directory for tests."""
    output_dir = tmp_path_factory.mktemp("test_output")
    return output_dir


@pytest.fixture(scope="session")
def weights_dir(repo_root):
    """Return path to model weights directory."""
    weights_path = repo_root / "data" / "weights"
    return weights_path


@pytest.fixture
def gpu_available():
    """Check if GPU is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture
def skip_if_no_gpu(gpu_available):
    """Skip test if GPU is not available."""
    if not gpu_available:
        pytest.skip("GPU not available - test requires CUDA")
