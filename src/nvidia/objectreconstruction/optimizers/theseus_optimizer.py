#!/usr/bin/env python3
"""
Theseus-based optimizer for optimizing camera poses by matching 3D-3D correspondences
"""
import warnings

import torch

# Suppress optional CHOLMOD backend warning before importing Theseus (any import order / entry point).
warnings.filterwarnings(
    "ignore",
    message=r"Couldn't import skparse\.cholmod\. Cholmod solver won't work\.$",
    category=UserWarning,
)

import torch

import theseus as th
import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from ..networks.tool import set_seed
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EntryJ:
    """Correspondence structure (corresponding to original C++ EntryJ) - adjusted for 3D-3D correspondences"""
    imgIdx_i: int
    imgIdx_j: int
    pos_i_3d: np.ndarray  # 3D position in world coordinates for frame i
    pos_j_3d: np.ndarray  # 3D position in world coordinates for frame j
    normal_i: np.ndarray  # Surface normal for frame i
    normal_j: np.ndarray  # Surface normal for frame j

class TheseusOptimizer:
    """
    Theseus-based optimizer for pose optimization by matching 3D-3D correspondences.
    """
    
    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = logger):
        self.config = config or {}
        self.debug_dir = self.config.get("debug_dir", "debug")
        self.outer_iterations = self.config.get("theseus", {}).get("num_iter_outter", 3)
        self.inner_iterations = self.config.get("theseus", {}).get("num_iter_inner", 10)
        self.num_sampled_corres = self.config.get("theseus", {}).get("num_sampled_corres", 10000)
        self.use_cpu = self.config.get("theseus", {}).get("use_cpu", False)
        self.sparse_weights = self.config.get("theseus", {}).get("w_sparse", 1)
        self.dense_weights = self.config.get("theseus", {}).get("w_dense", 1)
        self.robust_delta = self.config.get("theseus", {}).get("robust_delta", 0.005)
        self.step_size = self.config.get("theseus", {}).get("step_size", 0.1)
        self.damping = self.config.get("theseus", {}).get("damping", 0.1)

        # print(f"[TheseusOpt] initialize with self.config: {self.config}")
        self.logger = logger
        self.logger.info(f"Initialize Theseus optimizer, outer iterations: {self.outer_iterations}, inner iterations: {self.inner_iterations}, sparse weights: {self.sparse_weights}, dense weights: {self.dense_weights}, robust delta: {self.robust_delta}")
        
        # detect and set device
        self.device = torch.device("cpu") if self.use_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        self.logger.info(f"Using device: {self.device}")
    
    def optimize_frames(self, 
                       global_corres: Dict[Tuple[int, int], List[EntryJ]],
                       poses: List[np.ndarray],
                       progress_bar: Any = None) -> List[np.ndarray]:

        """Optimize camera poses by matching 3D-3D correspondences.
        
        Args:
            global_corres: Dictionary of correspondences between frames
            poses: List of initial camera poses
            progress_bar: Optional tqdm progress bar for phase-level updates
            K: Camera intrinsic matrix
        
        Returns:
            List of optimized camera poses
        
        Processing Steps:
            1. Sample correspondences for each iteration
            2. Create cost function for comuting losses of correspondences
            3. Optimize poses by minimizing the losses with Theseus optimizer
        """
        # poses: (N, 4, 4)
        # global_corres: (N, N)
        
        set_seed(1)

        # Outter/Inner Loop optimization like BundleFusion (we removed the multi-scale optimization, originally downscale the image 4x in BundleFusion's implementation)
        # Outter loop: sample correspondences for each iteration
        # Inner loop: optimize poses with step size for each iteration
        frame_pairs = list(global_corres.keys())
        self.logger.info(f"frame_pairs: {frame_pairs}")
        for iter_idx in range(self.outer_iterations):
            if progress_bar is not None:
                progress_bar.set_postfix_str(
                    f"optimizing poses: iteration {iter_idx + 1}/{self.outer_iterations}",
                    refresh=True,
                )
            self.logger.info(f"=== Iteration {iter_idx} ===")
            # Outer loop: sample correspondences for each iteration
            poses, err = self._optimize_one_iteration(
                global_corres, poses, 
                self.sparse_weights, 
                self.dense_weights, 
                self.robust_delta,
                self.inner_iterations,
                self.num_sampled_corres,
                progress_bar=progress_bar,
                iteration_label=f"{iter_idx + 1}/{self.outer_iterations}",
            )
            if progress_bar is not None:
                try:
                    err_value = err.detach().cpu().item() if hasattr(err, "detach") else float(err)
                    progress_bar.set_postfix_str(
                        f"optimizing poses: iteration {iter_idx + 1}/{self.outer_iterations}, err {err_value:.4g}",
                        refresh=True,
                    )
                except Exception:
                    progress_bar.set_postfix_str(
                        f"optimizing poses: iteration {iter_idx + 1}/{self.outer_iterations} complete",
                        refresh=True,
                    )
        
        return poses
    
    def _create_cost_function(self, pose_vars, global_corres, frame_i, frame_j, weight_sparse, weight_dense, robust_delta, max_corres=1024):
        """Create a cost function for pose optimization by matching 3D-3D correspondences.
        
        Args:
            pose_vars: List of Theseus SE3 pose variables
            global_corres: List of correspondence entries
            weight_sparse: Weight for sparse feature correspondences  
            weight_dense: Weight for point-to-plane correspondences
            robust_delta: Delta parameter for Huber robust loss
            max_corres: Maximum number of correspondences to use during optimization (default: 1024)
        Returns:
            Theseus AutoDiffCostFunction for pose optimization
        """

        # if no correspondences, return a cost function that returns 0
        if len(global_corres) == 0:
            return th.AutoDiffCostFunction(
                        pose_vars,
                        lambda optim_vars, aux_vars=None: torch.zeros(1, 1, dtype=torch.float32, device=self.device),
                        1,
                        name=f"corres_cost_{frame_i}_{frame_j}"
                    )

        # check if normal is available, if not, use sparse optimization (minimize distance of 3D correspondences only) only
        is_normal_available = global_corres[0].normal_i is not None
        if not is_normal_available:
            self.logger.info(f"Use sparse optimization only")
            weight_sparse = 1.0
            weight_dense = 0.0

        # downsample the number of correspondences to avoid OOM
        if len(global_corres) > max_corres:
            # evenly sample the correspondences from each pair
            self.logger.debug(f"Pair {frame_i} and {frame_j}: Too many correspondences ({len(global_corres)}), sampling {max_corres} to avoid OOM.")
            sampled_global_corres = [global_corres[n] for n in np.random.choice(len(global_corres), max_corres, replace=False).astype(np.int32)]
        else:
            self.logger.info(f"Pair {frame_i} and {frame_j}: Using all correspondences ({len(global_corres)})")
            sampled_global_corres = global_corres
                
        def huber_loss(residual_squared, delta):
            """Huber robust loss function"""
            delta_sq = delta * delta
            # residual_squared is already squared error
            mask = residual_squared <= delta_sq
            
            # For inliers: rho = e, rho' = 1
            rho = torch.where(mask, residual_squared, torch.zeros_like(residual_squared))
            rho_deriv = torch.where(mask, torch.ones_like(residual_squared), torch.zeros_like(residual_squared))
            
            # For outliers: rho = 2*sqrt(e)*delta - delta^2, rho' = delta/sqrt(e)
            sqrt_e = torch.sqrt(residual_squared + 1e-8)  # add small epsilon for numerical stability
            rho_outlier = 2 * sqrt_e * delta - delta_sq
            rho_deriv_outlier = delta / sqrt_e
            
            rho = torch.where(~mask, rho_outlier, rho)
            rho_deriv = torch.where(~mask, rho_deriv_outlier, rho_deriv)
            
            return rho, rho_deriv

        # cost function that uses sparse optimization (minimize distance of 3D correspondences) only
        def corres_cost_fn_sparse(optim_vars, aux_vars=None):
            N = len(sampled_global_corres)
            pos_i = torch.stack([torch.tensor(c.pos_i_3d, dtype=torch.float32, device=self.device) for c in sampled_global_corres], dim=0)  # (N, 3)
            pos_j = torch.stack([torch.tensor(c.pos_j_3d, dtype=torch.float32, device=self.device) for c in sampled_global_corres], dim=0)  # (N, 3)

            pi_w = optim_vars[0].transform_from(pos_i).tensor  # (N, 3)
            pj_w = optim_vars[1].transform_from(pos_j).tensor  # (N, 3)
            sparse_residual = pi_w - pj_w  # (N, 3)

            sparse_residual_sq = torch.sum(sparse_residual ** 2, dim=1)  # (N,)

            _, sparse_rho_deriv = huber_loss(sparse_residual_sq, robust_delta)
            residuals = weight_sparse * (sparse_rho_deriv.unsqueeze(1) * sparse_residual)      # (N, 3)
            
            return residuals.reshape(1, -1)

        # cost function that uses dense optimization (minimize point-to-plane distance) only
        def corres_cost_fn_dense(optim_vars, aux_vars=None):
            N = len(sampled_global_corres)
            pos_i = torch.stack([torch.tensor(c.pos_i_3d, dtype=torch.float32, device=self.device) for c in sampled_global_corres], dim=0)  # (N, 3)
            pos_j = torch.stack([torch.tensor(c.pos_j_3d, dtype=torch.float32, device=self.device) for c in sampled_global_corres], dim=0)  # (N, 3)

            pi_w = optim_vars[0].transform_from(pos_i).tensor  # (N, 3)
            pj_w = optim_vars[1].transform_from(pos_j).tensor  # (N, 3)

            normal_i = torch.stack([torch.tensor(c.normal_i, dtype=torch.float32, device=self.device) for c in sampled_global_corres], dim=0)  # (N, 3)
            normal_j = torch.stack([torch.tensor(c.normal_j, dtype=torch.float32, device=self.device) for c in sampled_global_corres], dim=0)  # (N, 3)
            
            pi_in_j = optim_vars[1].transform_to(pi_w).tensor # (N, 3)
            pj_in_i = optim_vars[0].transform_to(pj_w).tensor # (N, 3)
            
            # Point-to-plane residual 
            diff = torch.cat([(pos_j - pi_in_j) * normal_j, (pos_i - pj_in_i) * normal_i], dim=0)  # (2N, 3)
            
            # Point-to-plane distance: dot(diff, normal)
            dense_residual = torch.sum(diff, dim=1)  # (2N,) 

            dense_residual_sq = dense_residual ** 2  # (2N,)
            
            _, dense_rho_deriv = huber_loss(dense_residual_sq, robust_delta)

            robust_dense_residual = weight_dense * (dense_rho_deriv * dense_residual)   

            residuals = torch.cat([
                robust_dense_residual[:N].unsqueeze(1),           # (N, 1)
                robust_dense_residual[N:].unsqueeze(1),           # (N, 1)
            ], dim=1)  # (N, 2)
            
            return residuals.reshape(1, -1)

        # cost function that uses both sparse and dense optimization
        def corres_cost_fn_sparse_dense(optim_vars, aux_vars=None):
            N = len(sampled_global_corres)
            pos_i = torch.stack([torch.tensor(c.pos_i_3d, dtype=torch.float32, device=self.device) for c in sampled_global_corres], dim=0)  # (N, 3)
            pos_j = torch.stack([torch.tensor(c.pos_j_3d, dtype=torch.float32, device=self.device) for c in sampled_global_corres], dim=0)  # (N, 3)

            pi_w = optim_vars[0].transform_from(pos_i).tensor  # (N, 3)
            pj_w = optim_vars[1].transform_from(pos_j).tensor  # (N, 3)
            sparse_residual = pi_w - pj_w  # (N, 3)

            sparse_residual_sq = torch.sum(sparse_residual ** 2, dim=1)  # (N,)

            _, sparse_rho_deriv = huber_loss(sparse_residual_sq, robust_delta)
            robust_sparse_residual = weight_sparse * (sparse_rho_deriv.unsqueeze(1) * sparse_residual)      # (N, 3)

            normal_i = torch.stack([torch.tensor(c.normal_i, dtype=torch.float32, device=self.device) for c in sampled_global_corres], dim=0)  # (N, 3)
            normal_j = torch.stack([torch.tensor(c.normal_j, dtype=torch.float32, device=self.device) for c in sampled_global_corres], dim=0)  # (N, 3)

            pi_in_j = optim_vars[1].transform_to(pi_w).tensor # (N, 3)
            pj_in_i = optim_vars[0].transform_to(pj_w).tensor # (N, 3)
            
            # Point-to-plane residual 
            diff = torch.cat([(pos_j - pi_in_j) * normal_j, (pos_i - pj_in_i) * normal_i], dim=0)  # (2N, 3)
            
            # Point-to-plane distance: dot(diff, normal)
            dense_residual = torch.sum(diff, dim=1)  # (2N,) 

            dense_residual_sq = dense_residual ** 2  # (2N,)
            
            _, dense_rho_deriv = huber_loss(dense_residual_sq, robust_delta)

            robust_dense_residual = weight_dense * (dense_rho_deriv * dense_residual)   

            residuals = torch.cat([
                robust_sparse_residual,                           # (N, 3)
                robust_dense_residual[:N].unsqueeze(1),           # (N, 1)
                robust_dense_residual[N:].unsqueeze(1),           # (N, 1)
            ], dim=1)  # (N, 5)
            
            return residuals.reshape(1, -1)

        if is_normal_available and weight_sparse > 0.0:
            corres_cost_fn = corres_cost_fn_sparse_dense
            dim = 5
        elif is_normal_available and weight_sparse == 0.0:
            corres_cost_fn = corres_cost_fn_dense
            dim = 2
        else:
            corres_cost_fn = corres_cost_fn_sparse
            dim = 3

        cost_function = th.AutoDiffCostFunction(
            pose_vars,
            corres_cost_fn,
            len(sampled_global_corres) * dim,
            name=f"corres_cost_{frame_i}_{frame_j}"
        )
        
        return cost_function

    @torch.no_grad()
    def _optimize_one_iteration(
        self,
        global_corres,
        poses,
        weight_sparse,
        weight_dense,
        robust_delta,
        max_iterations,
        num_sampled_corres,
        progress_bar=None,
        iteration_label=None,
    ):
        """Single scale optimization (replaces the original optimize_frames core logic)"""
        iter_text = f"iteration {iteration_label}" if iteration_label is not None else "iteration"
        # --- Theseus pose variable preparation ---
        if progress_bar is not None:
            progress_bar.set_postfix_str(f"optimizing poses: {iter_text}, preparing variables", refresh=True)
        n_frames = len(poses)
        se3_matrices = np.stack([pose[:3, :4] for pose in poses], axis=0).astype(np.float32)
        se3_tensor = torch.from_numpy(se3_matrices).float().to(self.device)
        pose_vars = [th.SE3(tensor=se3_tensor[i:i+1], name=f"pose_{i}") for i in range(n_frames)]

        objective = th.Objective()
        frame_pairs = list(global_corres.keys())
        if progress_bar is not None:
            progress_bar.set_postfix_str(f"optimizing poses: {iter_text}, building objective", refresh=True)
        for pair_idx, (frame_i, frame_j) in enumerate(frame_pairs, start=1):
            if progress_bar is not None:
                progress_bar.set_postfix_str(
                    f"optimizing poses: {iter_text}, cost {pair_idx}/{len(frame_pairs)}",
                    refresh=True,
                )
            cost_function = self._create_cost_function([pose_vars[frame_i], pose_vars[frame_j]], global_corres[frame_i, frame_j], frame_i, frame_j, weight_sparse, weight_dense, robust_delta, num_sampled_corres // len(frame_pairs))
            objective.add(cost_function)
            
        # Inner loop: optimize poses with step size for each iteration
        def run_optimization():
            optimizer = th.LevenbergMarquardt(
                objective.to(self.device),
                max_iterations=max_iterations,
                vectorize=True,
                step_size=self.step_size
            )
            layer = th.TheseusLayer(optimizer).to(self.device)
            input_tensors = {f"pose_{i}": se3_tensor[i:i+1].float().to(self.device) for i in range(n_frames)}
            return layer.forward(input_tensors=input_tensors, optimizer_kwargs={"damping": self.damping})
        
        if self.device.type == "cuda":
            # force float32 precision to avoid BFloat16 issues.
            if progress_bar is not None:
                progress_bar.set_postfix_str(f"optimizing poses: {iter_text}, running Theseus", refresh=True)
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                solution, info = run_optimization()
        else:
            if progress_bar is not None:
                progress_bar.set_postfix_str(f"optimizing poses: {iter_text}, running Theseus", refresh=True)
            solution, info = run_optimization()


        if progress_bar is not None:
            progress_bar.set_postfix_str(f"optimizing poses: {iter_text}, collecting results", refresh=True)
        optimized_poses = []
        for i in range(n_frames):
            if f"pose_{i}" in solution:
                optimized_se3 = solution[f"pose_{i}"]
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :] = optimized_se3.squeeze(0).detach().cpu().numpy()
                optimized_poses.append(pose)
            else:
                optimized_poses.append(poses[i])
        try:        
            # Post-process: set first frame as anchor (identity matrix)
            if progress_bar is not None:
                progress_bar.set_postfix_str(f"optimizing poses: {iter_text}, anchoring poses", refresh=True)
            T0_inv = np.linalg.inv(optimized_poses[0])
            optimized_poses = [poses[0] @ T0_inv @ p for p in optimized_poses]

            self.logger.info(f"Iteration finished. Final loss: {info.last_err.item()}")
            return optimized_poses, info.last_err/self.num_sampled_corres
        except np.linalg.LinAlgError:
            self.logger.warning("Failed to invert first pose matrix, returning original poses")
            return poses, float('inf')
        
