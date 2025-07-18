"""
Utility functions for 3D reconstruction and processing.

This module provides various utilities for point cloud processing, mesh operations,
tensor type management, and pose transformations used in 3D reconstruction tasks.
"""

import copy
import random
import logging
from typing import Tuple, Optional, Dict, Any, List, Union
from pathlib import Path

import numpy as np
import joblib
import open3d as o3d
import torch
import trimesh
from sklearn.cluster import DBSCAN

from ..utils.preprocessing import toOpen3dCloud, depth2xyzmap

logger = logging.getLogger(__name__)

# Constants
GL_CAM_TO_CV_CAM = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])


class PointCloudProcessor:
    """Handles point cloud processing operations."""

    @staticmethod
    def find_biggest_cluster(
        pts: np.ndarray,
        eps: float = 0.06,
        min_samples: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the largest cluster in a point cloud using DBSCAN.

        This method uses DBSCAN clustering to identify the largest coherent
        cluster in a point cloud, which is useful for filtering out noise
        and isolating the main object of interest.

        Args:
            pts: Point cloud as numpy array of shape (N, 3) containing
                 3D coordinates of points
            eps: DBSCAN epsilon parameter - maximum distance between two
                 samples for one to be considered as in the neighborhood
                 of the other
            min_samples: Minimum number of samples in a neighborhood for
                        a point to be considered as a core point

        Returns:
            A tuple containing:
                - clustered points: numpy array of points belonging to the
                  largest cluster
                - keep mask: boolean array indicating which points were kept

        Raises:
            ValueError: If input point cloud is empty or has invalid dimensions
        """
        if pts.size == 0:
            raise ValueError("Input point cloud is empty")
        if pts.shape[1] != 3:
            raise ValueError(f"Expected 3D points, got shape {pts.shape}")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        dbscan.fit(pts)
        ids, cnts = np.unique(dbscan.labels_, return_counts=True)
        best_id = ids[cnts.argsort()[-1]]
        keep_mask = dbscan.labels_ == best_id
        pts_cluster = pts[keep_mask]
        return pts_cluster, keep_mask

    @staticmethod
    def compute_translation_scales(
        pts: np.ndarray,
        max_dim: float = 2,
        cluster: bool = True,
        eps: float = 0.06,
        min_samples: int = 1
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Compute translation and scale factors for point cloud normalization.

        This method calculates the translation and scaling parameters needed
        to normalize a point cloud to fit within a specified dimension range,
        typically [-1, 1] after scaling.

        Args:
            pts: Point cloud as numpy array of shape (N, 3)
            max_dim: Maximum dimension for normalization - the largest
                    dimension of the bounding box will be scaled to this value
            cluster: Whether to cluster points first using DBSCAN to remove
                    outliers and focus on the main object
            eps: DBSCAN epsilon parameter (only used if cluster=True)
            min_samples: Minimum samples for DBSCAN (only used if cluster=True)

        Returns:
            A tuple containing:
                - translation_cvcam: Translation vector to center the point cloud
                - sc_factor: Scale factor to normalize the point cloud
                - keep_mask: Boolean mask indicating which points were kept

        Raises:
            ValueError: If input point cloud is empty after clustering
        """
        if cluster:
            pts, keep_mask = PointCloudProcessor.find_biggest_cluster(
                pts, eps, min_samples
            )
        else:
            keep_mask = np.ones((len(pts)), dtype=bool)

        if len(pts) == 0:
            raise ValueError("No points remain after clustering")

        max_xyz = pts.max(axis=0)
        min_xyz = pts.min(axis=0)
        center = (max_xyz + min_xyz) / 2
        sc_factor = max_dim / (max_xyz - min_xyz).max()  # Normalize to [-1,1]
        sc_factor *= 0.9  # Reserve some space
        translation_cvcam = -center
        return translation_cvcam, sc_factor, keep_mask

    @staticmethod
    def compute_scene_bounds_worker(
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        glcam_in_world: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Process a single frame to extract point cloud.

        This worker function processes a single RGB-D frame with an associated
        mask to extract a cleaned point cloud in world coordinates. It applies
        statistical outlier removal and transforms the points using the provided
        camera pose.

        Args:
            rgb: RGB image as numpy array of shape (H, W, 3) or (H, W, 4)
            depth: Depth image as numpy array of shape (H, W) with depth values
                  in meters
            mask: Binary mask as numpy array of shape (H, W) where non-zero
                  values indicate valid regions
            K: Camera intrinsic matrix as numpy array of shape (3, 3)
            glcam_in_world: Camera pose matrix (OpenGL convention) as numpy
                           array of shape (4, 4)

        Returns:
            A tuple containing (points, colors) as numpy arrays, or None if
            no valid points are found after processing

        Note:
            - Uses depth threshold of 0.1m to filter out invalid depth values
            - Applies statistical outlier removal with two passes
            - Transforms from OpenGL camera convention to OpenCV convention
        """
        rgb = rgb[..., :3]
        xyz_map = depth2xyzmap(depth, K)
        valid = depth >= 0.1
        valid = valid & (mask > 0)
        pts = xyz_map[valid].reshape(-1, 3)

        if len(pts) == 0:
            return None

        colors = rgb[valid].reshape(-1, 3)
        pcd = toOpen3dCloud(pts, colors)
        pcd = pcd.voxel_down_sample(0.01)
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30,std_ratio=2.0)

        cam_in_world = glcam_in_world @ GL_CAM_TO_CV_CAM
        pcd.transform(cam_in_world)

        return np.asarray(pcd.points).copy(), np.asarray(pcd.colors).copy()

    @staticmethod
    def make_transform(
        translation_cvcam: np.ndarray,
        sc_factor: float
    ) -> np.ndarray:
        """
        Create transformation matrix from translation and scale.

        Constructs a 4x4 homogeneous transformation matrix that first
        translates then scales the input coordinates.

        Args:
            translation_cvcam: Translation vector as numpy array of shape (3,)
            sc_factor: Uniform scale factor to apply after translation

        Returns:
            4x4 transformation matrix as numpy array that applies translation
            followed by scaling
        """
        tf = np.eye(4)
        tf[:3, 3] = translation_cvcam
        tf1 = np.eye(4)
        tf1[:3, :3] *= sc_factor
        return tf1 @ tf

    @staticmethod
    def compute_scene_bounds(
        rgbs: List[np.ndarray],
        depths: List[np.ndarray],
        masks: List[np.ndarray],
        glcam_in_worlds: List[np.ndarray],
        K: np.ndarray,
        translation_cvcam: Optional[np.ndarray] = None,
        sc_factor: Optional[float] = None,
        eps: float = 0.06,
        min_samples: int = 1,
        cluster: bool = True
    ) -> Tuple[float, np.ndarray, o3d.geometry.PointCloud,
               o3d.geometry.PointCloud]:
        """
        Compute scene bounds from multiple frames.

        This method processes multiple RGB-D frames to create a unified point
        cloud representation of the scene, computes normalization parameters,
        and returns both the original and normalized point clouds.

        Args:
            rgbs: List of RGB images, each as numpy array of shape (H, W, 3)
            depths: List of depth images, each as numpy array of shape (H, W)
            masks: List of binary masks, each as numpy array of shape (H, W)
            glcam_in_worlds: List of camera pose matrices, each as numpy array
                           of shape (4, 4) in OpenGL convention
            K: Camera intrinsic matrix as numpy array of shape (3, 3)
            translation_cvcam: Optional pre-computed translation vector. If None,
                              will be computed automatically
            sc_factor: Optional pre-computed scale factor. If None, will be
                      computed automatically
            eps: DBSCAN epsilon parameter for clustering
            min_samples: Minimum samples for DBSCAN clustering
            cluster: Whether to apply clustering to remove outliers

        Returns:
            A tuple containing:
                - sc_factor: Computed or provided scale factor
                - translation_cvcam: Computed or provided translation vector
                - pcd_real_scale: Point cloud in original real-world scale
                - pcd: Normalized point cloud (scaled and translated)

        Raises:
            ValueError: If no valid frames are provided or processed
        """
        if not rgbs or len(rgbs) == 0:
            raise ValueError("No RGB frames provided")

        args = [
            (rgbs[i], depths[i], masks[i], K, glcam_in_worlds[i])
            for i in range(len(rgbs))
        ]

        ret = joblib.Parallel(n_jobs=5, prefer="threads")(
            joblib.delayed(PointCloudProcessor.compute_scene_bounds_worker)(
                *arg
            ) for arg in args
        )

        pcd_all = None
        for r in ret:
            if r is None:
                continue
            if pcd_all is None:
                pcd_all = toOpen3dCloud(r[0], r[1])
            else:
                pcd_all += toOpen3dCloud(r[0], r[1])

        if pcd_all is None:
            raise ValueError("No valid point clouds generated from frames")

        pcd = pcd_all.voxel_down_sample(eps/5)

        pts = np.asarray(pcd.points).copy()

        if translation_cvcam is None:
            translation_cvcam, sc_factor, keep_mask = (
                PointCloudProcessor.compute_translation_scales(
                    pts, cluster=cluster, eps=eps, min_samples=min_samples
                )
            )
            tf = PointCloudProcessor.make_transform(
                translation_cvcam, sc_factor
            )
        else:
            tf = PointCloudProcessor.make_transform(
                translation_cvcam, sc_factor
            )
            tmp = copy.deepcopy(pcd)
            tmp.transform(tf)
            tmp_pts = np.asarray(tmp.points)
            keep_mask = (np.abs(tmp_pts) < 1).all(axis=-1)


        pcd = toOpen3dCloud(pts[keep_mask], np.asarray(pcd.colors)[keep_mask])
        pcd_real_scale = copy.deepcopy(pcd)
        pcd.transform(tf)

        return sc_factor, translation_cvcam, pcd_real_scale, pcd


class MeshProcessor:
    """Handles mesh processing operations."""

    @staticmethod
    def smooth_mesh(
        mesh: trimesh.Trimesh,
        cfg_nerf: Dict[str, Any],
        debug_dir: Union[str, Path]
    ) -> trimesh.Trimesh:
        """
        Apply smoothing to a mesh based on configuration.

        This method applies various smoothing algorithms to a mesh to reduce
        noise and improve surface quality. It supports Laplacian smoothing
        and optionally Taubin smoothing for better preservation of geometric
        features.

        Args:
            mesh: Input mesh as a trimesh.Trimesh object
            cfg_nerf: Configuration dictionary containing smoothing parameters
                     Expected keys:
                     - mesh_smoothing.enabled: bool, whether to apply smoothing
                     - mesh_smoothing.iterations: int, number of iterations
                     - mesh_smoothing.lambda: float, smoothing parameter
                     - mesh_smoothing.use_taubin: bool, whether to use Taubin
            debug_dir: Directory path for saving debug outputs (smoothed mesh)

        Returns:
            Smoothed mesh as trimesh.Trimesh object

        Note:
            - Saves the smoothed mesh to debug_dir/mesh_smoothed.obj
            - Applies mesh.process() for basic cleanup before smoothing
            - Uses both Laplacian and Taubin smoothing if use_taubin is True
        """
        if cfg_nerf.get('mesh_smoothing', {}).get('enabled', True):
            smoothing_iterations = cfg_nerf.get('mesh_smoothing', {}).get(
                'iterations', 5
            )
            smoothing_lambda = cfg_nerf.get('mesh_smoothing', {}).get(
                'lambda', 0.3
            )

            mesh.process()

            trimesh.smoothing.filter_laplacian(
                mesh,
                iterations=smoothing_iterations,
                lamb=smoothing_lambda
            )

            if cfg_nerf.get('mesh_smoothing', {}).get('use_taubin', True):
                lambda_factor = 0.330
                mu_factor = -0.331

                trimesh.smoothing.filter_taubin(
                    mesh,
                    iterations=smoothing_iterations,
                    lamb=lambda_factor,
                    nu=mu_factor
                )

            logger.info(
                f'Applied mesh smoothing with {smoothing_iterations} '
                f'iterations'
            )

            smoothed_mesh_path = Path(debug_dir) / "mesh_smoothed.obj"
            mesh.export(str(smoothed_mesh_path))
            logger.info(f"Saved smoothed mesh to {smoothed_mesh_path}")

        return mesh

    @staticmethod
    def mesh_to_real_world(
        mesh: trimesh.Trimesh,
        pose_offset: np.ndarray,
        translation: np.ndarray,
        sc_factor: float
    ) -> trimesh.Trimesh:
        """
        Transform mesh to real world coordinates.

        This method transforms a normalized mesh back to real-world coordinates
        by reversing the normalization (scaling and translation) and applying
        a pose offset transformation.

        Args:
            mesh: Input mesh in normalized coordinates
            pose_offset: 4x4 transformation matrix to apply as final pose offset
            translation: Translation vector used during normalization
            sc_factor: Scale factor used during normalization

        Returns:
            Transformed mesh in real-world coordinates

        Note:
            - The transformation order is: scale -> translate -> pose offset
            - Modifies the mesh in-place and returns it
        """
        mesh.vertices = (
            mesh.vertices/sc_factor -
            np.array(translation).reshape(1, 3)
        )
        mesh.apply_transform(pose_offset)
        return mesh


class TensorUtils:
    """Handles tensor type management and conversions."""

    @staticmethod
    def ensure_tensor_type(
        tensor: torch.Tensor,
        target_dtype: torch.dtype = torch.float32,
        name: str = "tensor"
    ) -> torch.Tensor:
        """
        Ensure tensor is of specified type.

        This utility function checks if a tensor has the required data type
        and converts it if necessary, with optional logging for debugging.

        Args:
            tensor: Input tensor to check and potentially convert
            target_dtype: Desired PyTorch data type (default: torch.float32)
            name: Descriptive name for the tensor used in debug logging

        Returns:
            Tensor with the correct data type (either original or converted)

        Note:
            - Logs conversion operations at debug level
            - Returns original tensor if already correct type (no copy)
        """
        if tensor.dtype != target_dtype:
            logger.debug(
                f"Converting {name} from {tensor.dtype} to "
                f"{target_dtype}"
            )
            return tensor.to(target_dtype)
        return tensor

    @staticmethod
    def ensure_tensor_types(
        tensors_dict: Dict[str, torch.Tensor],
        dtypes_dict: Dict[str, torch.dtype]
    ) -> Dict[str, torch.Tensor]:
        """
        Ensure multiple tensors have correct types.

        This method processes a dictionary of tensors and ensures each one
        has the correct data type as specified in the dtypes dictionary.

        Args:
            tensors_dict: Dictionary mapping tensor names to torch.Tensor
                         objects
            dtypes_dict: Dictionary mapping tensor names to required
                        torch.dtype values

        Returns:
            Dictionary with same keys as tensors_dict but with corrected
            tensor types where necessary

        Note:
            - Only tensors specified in dtypes_dict are type-checked
            - Other tensors are passed through unchanged
        """
        result = {}
        for name, tensor in tensors_dict.items():
            if name in dtypes_dict:
                result[name] = TensorUtils.ensure_tensor_type(
                    tensor, dtypes_dict[name], name
                )
            else:
                result[name] = tensor
        return result


class PoseUtils:
    """Handles pose transformations and optimizations."""

    @staticmethod
    def get_optimized_poses_in_real_world(
        poses_normalized: np.ndarray,
        pose_array: Any,
        sc_factor: float,
        translation: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert normalized poses to real world coordinates.

        This method takes poses that have been optimized in normalized space
        and converts them back to real-world coordinates, applying appropriate
        transformations and maintaining consistency with the original pose
        relationships.

        Args:
            poses_normalized: Array of pose matrices in normalized space,
                            shape (N, 4, 4)
            pose_array: Pose array object containing optimization transforms
                       (expected to have get_matrices method)
            sc_factor: Scale factor used for normalization
            translation: Translation vector used for normalization

        Returns:
            A tuple containing:
                - optimized_poses: Array of poses in real-world coordinates
                - offset: Transformation offset matrix used for alignment

        Note:
            - Applies GL_CAM_TO_CV_CAM transformation for camera convention
            - Ensures pose consistency by computing and applying offset
            - Returns poses as float32 arrays
        """
        original_poses = poses_normalized.copy()
        original_poses[:, :3, 3] /= sc_factor
        original_poses[:, :3, 3] -= translation

        tf = (
            pose_array.get_matrices(np.arange(len(poses_normalized)))
            .reshape(-1, 4, 4)
            .data.cpu().numpy()
        )
        optimized_poses = tf @ poses_normalized

        optimized_poses = np.array(optimized_poses).astype(np.float32)
        optimized_poses[:, :3, 3] /= sc_factor
        optimized_poses[:, :3, 3] -= translation

        original_init_ob_in_cam = optimized_poses[0].copy()
        offset = np.linalg.inv(original_init_ob_in_cam) @ original_poses[0]

        for i in range(len(optimized_poses)):
            new_ob_in_cam = optimized_poses[i] @ offset
            optimized_poses[i] = new_ob_in_cam
            optimized_poses[i] = optimized_poses[i] @ GL_CAM_TO_CV_CAM

        return optimized_poses, offset


# Expose commonly used functions at module level for backward compatibility
compute_scene_bounds = PointCloudProcessor.compute_scene_bounds


def set_seed(random_seed: int) -> None:
    """
    Set random seeds for reproducible results.

    This function sets the random seed for multiple libraries to ensure
    reproducible results across different runs of the same code.

    Args:
        random_seed: Integer seed value to use for all random number generators

    Note:
        - Sets seeds for numpy, Python's random, PyTorch CPU and CUDA
        - Configures PyTorch for deterministic behavior (may impact performance)
        - Should be called early in the program before any random operations
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
