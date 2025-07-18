"""
RoMa (Robust Matching) feature matching implementation.

This module provides dense feature matching capabilities using the RoMa model
for 3D object reconstruction. It includes custom transforms and inference
classes for processing stereo image pairs.
"""

import logging
import math
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm


class TupleResize:
    """Resize transform that applies to tuples of images."""

    def __init__(self, size, mode=InterpolationMode.BICUBIC):
        """
        Initialize the TupleResize transform.

        Args:
            size: Target size for resizing
            mode: Interpolation mode for resizing
        """
        self.size = size
        self.resize = transforms.Resize(size, mode)

    def __call__(self, im_tuple):
        """Apply resize transform to all images in the tuple."""
        return [self.resize(im) for im in im_tuple]

    def __repr__(self):
        """Return string representation."""
        return f"TupleResize(size={self.size})"


class ToTensorScaled:
    """Convert a RGB PIL Image to a CHW ordered Tensor, scale range to [0, 1]."""

    def __call__(self, im):
        """
        Convert image to tensor and scale to [0, 1].

        Args:
            im: Input image (PIL or tensor)

        Returns:
            torch.Tensor: Scaled tensor in CHW format
        """
        if not isinstance(im, torch.Tensor):
            im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
            im /= 255.0
            return torch.from_numpy(im)
        else:
            return im

    def __repr__(self):
        """Return string representation."""
        return "ToTensorScaled(./255)"


class TupleToTensorScaled:
    """Apply ToTensorScaled transform to tuples of images."""

    def __init__(self):
        """Initialize the tuple tensor transform."""
        self.to_tensor = ToTensorScaled()

    def __call__(self, im_tuple):
        """Apply ToTensorScaled to all images in the tuple."""
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        """Return string representation."""
        return "TupleToTensorScaled(./255)"


class TupleTensorNormalize:
    """Apply tensor normalization to tuples of images."""

    def __init__(self, mean, std):
        """
        Initialize the normalization transform.

        Args:
            mean: Mean values for normalization
            std: Standard deviation values for normalization
        """
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, im_tuple):
        """
        Apply normalization to all images in the tuple.

        Args:
            im_tuple: Tuple of image tensors

        Returns:
            List of normalized tensors
        """
        b, c, h, w = im_tuple[0].shape
        if c > 3:
            warnings.warn(f"Number of channels c={c} > 3, assuming first 3 are rgb")
        return [self.normalize(im) for im in im_tuple]

    def __repr__(self):
        """Return string representation."""
        return f"TupleNormalize(mean={self.mean}, std={self.std})"


class TupleCompose:
    """Compose multiple transforms for tuples of images."""

    def __init__(self, transforms):
        """
        Initialize the composition of transforms.

        Args:
            transforms: List of transforms to apply
        """
        self.transforms = transforms

    def __call__(self, im_tuple):
        """Apply all transforms sequentially to the image tuple."""
        for t in self.transforms:
            im_tuple = t(im_tuple)
        return im_tuple

    def __repr__(self):
        """Return string representation."""
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


def get_tuple_tensor_transform_ops(resize=None, normalize=True, unscale=False,
                                   clahe=False, colorjiggle_params=None):
    """
    Get a composition of tuple tensor transforms.

    Args:
        resize: Target size for resizing (optional)
        normalize: Whether to apply ImageNet normalization
        unscale: Whether to unscale (unused parameter)
        clahe: Whether to apply CLAHE (unused parameter)
        colorjiggle_params: Color jitter parameters (unused parameter)

    Returns:
        TupleCompose: Composed transform operations
    """
    ops = []
    if resize:
        ops.append(TupleResize(resize))
    ops.append(TupleToTensorScaled())
    if normalize:
        ops.append(
            TupleTensorNormalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        )  # Imagenet mean/std
    return TupleCompose(ops)


class FeatureMatchingInfer:
    """
    Feature matching inference class using RoMa model.

    This class provides dense feature matching capabilities between image pairs
    using the RoMa (Robust Matching) model for 3D object reconstruction.
    """

    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        """
        Initialize the feature matching inference.

        Args:
            config: Configuration dictionary containing model parameters
            logger: Optional logger instance
        """
        self.weights_path = config['weights']
        self.dinov2_weights_path = config['dinov2_weights']
        self.coarse_res = config['coarse_res']
        self.upsample_res = config['upsample_res']
        self.device = config.get('device', 'cuda')
        self.logger = logger if logger is not None else logging.getLogger(
            __name__
        )
        self.model = None

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the RoMa model with loaded weights."""
        try:
            import romatch
            weights = torch.load(self.weights_path)
            dinov2_weights = torch.load(self.dinov2_weights_path)

            self.model = romatch.roma_outdoor(
                weights=weights,
                dinov2_weights=dinov2_weights,
                coarse_res=self.coarse_res,
                upsample_res=self.upsample_res,
                device=self.device
            )
            self.model.eval()
            # to float32
            self.model.to(torch.float32)

            # Set target size to upsample resolution
            self.target_size = self.upsample_res
            self.max_batch_size = 1  # PyTorch model handles one pair at a time

            self.logger.info(
                f"PyTorch model loaded successfully. "
                f"Maximum batch size: {self.max_batch_size}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load PyTorch model: {str(e)}")
            raise

    def match_bundlesdf(self, im_A_tensor, im_B_tensor, *args, batched=True,
                        device=None):
        """
        Perform dense feature matching between two sets of images.

        This method uses the RoMa model to find dense correspondences between
        source and target images, returning warping fields and certainty scores.

        Args:
            im_A_tensor: Source images tensor of shape (B, H, W, C) or (H, W, C)
            im_B_tensor: Target images tensor of shape (B, H, W, C) or (H, W, C)
            *args: Additional arguments passed to the model
            batched: Whether the input is batched. Default: True
            device: Device to run the model on. Default: None (uses CUDA)

        Returns:
            tuple: A pair of (warp, certainty) where:
                - warp: Tensor of shape (B, H, W, 4) containing correspondence
                  coordinates. First two channels are source coordinates,
                  last two are target coordinates
                - certainty: Tensor of shape (B, H, W) containing confidence
                  scores for each correspondence

        Note:
            The function performs the following steps:
            1. Normalizes and resizes input images
            2. Runs the RoMa model to get dense correspondences
            3. Optionally upsamples the predictions
            4. Applies certainty attenuation if configured
            5. Returns the warping field and certainty scores
        """
        # Ensure device is set
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Force FP32 precision for feature matching - disable autocast completely
        # with torch.amp.autocast('cuda'):
        with torch.amp.autocast('cuda', enabled=False):
            # Ensure inputs are torch tensors in float32 format
            if not isinstance(im_A_tensor, torch.Tensor):
                im_A_tensor = torch.tensor(im_A_tensor, dtype=torch.float32)
            if not isinstance(im_B_tensor, torch.Tensor):
                im_B_tensor = torch.tensor(im_B_tensor, dtype=torch.float32)

            # Convert to required format and precision - explicitly set to float32
            if im_A_tensor.ndim == 3:  # (H, W, C) → add batch dim
                im_A_tensor = im_A_tensor.unsqueeze(0)
                im_B_tensor = im_B_tensor.unsqueeze(0)

            # Convert to required format and precision
            im_A_tensor = im_A_tensor.permute(0, 3, 1, 2).float().to(device)
            im_B_tensor = im_B_tensor.permute(0, 3, 1, 2).float().to(device)


            # Normalize
            im_A_tensor = im_A_tensor / 255.0
            im_B_tensor = im_B_tensor / 255.0

            # Get model parameters
            ws = self.model.w_resized
            hs = self.model.h_resized

            # Apply transformations
            test_transform = get_tuple_tensor_transform_ops(
                resize=(hs, ws), normalize=True, clahe=False
            )
            im_A, im_B = test_transform((im_A_tensor, im_B_tensor))
            
            # Explicitly ensure FP32 precision
            im_A = im_A.to(torch.float32)
            im_B = im_B.to(torch.float32)

            # Set model to evaluation mode and disable gradients
            symmetric = False
            self.model.train(False)
            with torch.no_grad():

                # Handle batched images
                b, c, h, w = im_A.shape
                b, c, h2, w2 = im_B.shape
                assert w == w2 and h == h2, \
                    "For batched images we assume same size"

                # Create batch dictionary - explicitly set to float32
                batch = {"im_A": im_A.to(device).to(torch.float32),
                         "im_B": im_B.to(device).to(torch.float32)}

                # Check for resolution mismatch
                if h != self.model.h_resized or self.model.w_resized != w:
                    warnings.warn(
                        "Model resolution and batch resolution differ, "
                        "may produce unexpected results"
                    )
                hs, ws = h, w

                # Set finest scale for output
                finest_scale = 1

                # Run the model
                if symmetric:
                    corresps = self.model.forward_symmetric(batch)
                else:
                    corresps = self.model.forward(batch, batched=True)

                # Handle upsampling if configured
                if self.model.upsample_preds:
                    hs, ws = self.model.upsample_res

                # Apply certainty attenuation if configured
                low_res_certainty = 0
                if self.model.attenuate_cert:
                    low_res_certainty = F.interpolate(
                        corresps[16]["certainty"], size=(hs, ws),
                        align_corners=False, mode="bilinear"
                    )
                    cert_clamp = 0
                    factor = 0.5
                    low_res_certainty = (
                        factor * low_res_certainty *
                        (low_res_certainty < cert_clamp)
                    )

                # Apply upsampling if configured
                if self.model.upsample_preds:
                    finest_corresps = corresps[finest_scale]
                    torch.cuda.empty_cache()
                    test_transform = get_tuple_tensor_transform_ops(
                        resize=(hs, ws), normalize=True
                    )
                    im_A, im_B = test_transform((im_A_tensor, im_B_tensor))
                    # Ensure FP32 precision
                    im_A = im_A.to(torch.float32)
                    im_B = im_B.to(torch.float32)
                    
                    scale_factor = math.sqrt(
                        self.model.upsample_res[0] * self.model.upsample_res[1]
                        / (self.model.w_resized * self.model.h_resized)
                    )
                    batch = {"im_A": im_A.to(device).to(torch.float32),
                             "im_B": im_B.to(device).to(torch.float32),
                             "corresps": finest_corresps}
                    if symmetric:
                        corresps = self.model.forward_symmetric(
                            batch, upsample=True, batched=True,
                            scale_factor=scale_factor
                        )
                    else:
                        corresps = self.model.forward(
                            batch, batched=True, upsample=True,
                            scale_factor=scale_factor
                        )

                # Extract flow and certainty from results
                im_A_to_im_B = corresps[finest_scale]["flow"].to(torch.float32)
                certainty = (
                    corresps[finest_scale]["certainty"].to(torch.float32) -
                    (low_res_certainty.to(torch.float32) if self.model.attenuate_cert else 0)
                )

                # Apply interpolation if necessary
                if finest_scale != 1:
                    im_A_to_im_B = F.interpolate(
                        im_A_to_im_B, size=(hs, ws), align_corners=False,
                        mode="bilinear"
                    ).to(torch.float32)
                    certainty = F.interpolate(
                        certainty, size=(hs, ws), align_corners=False,
                        mode="bilinear"
                    ).to(torch.float32)

                # Reshape flow
                im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)

                # Create source image coordinates grid
                im_A_coords = torch.meshgrid(
                    (
                        torch.linspace(
                            -1 + 1 / hs, 1 - 1 / hs, hs, device=device, dtype=torch.float32
                        ),
                        torch.linspace(
                            -1 + 1 / ws, 1 - 1 / ws, ws, device=device, dtype=torch.float32
                        ),
                    ),
                    indexing='ij'
                )
                im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
                im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)

                # Convert certainty from logits to probabilities
                certainty = certainty.sigmoid()

                # Reshape coordinates
                im_A_coords = im_A_coords.permute(0, 2, 3, 1)

                # Handle out-of-bounds predictions
                if (im_A_to_im_B.abs() > 1).any():
                    wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
                    certainty[wrong[:, None]] = 0

                # Clamp flow values
                im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)

                # Generate output based on symmetric flag
                if symmetric:
                    A_to_B, B_to_A = im_A_to_im_B.chunk(2)
                    q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
                    im_B_coords = im_A_coords
                    s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
                    warp = torch.cat((q_warp, s_warp), dim=2)
                    certainty = torch.cat(certainty.chunk(2), dim=3)
                else:
                    warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)

                # Return based on batched flag
                if batched:
                    return (
                        warp.to(torch.float32),
                        certainty[:, 0].to(torch.float32)
                    )
                else:
                    return (
                        warp[0].to(torch.float32),
                        certainty[0, 0].to(torch.float32),
                    )

    def _process_batch(self, rgbAs: np.ndarray, rgbBs: np.ndarray):
        """
        Find and return correspondences between source and target images.

        Args:
            rgbAs: Source images of shape (N, H, W, C)
            rgbBs: Target images of shape (N, H, W, C)

        Returns:
            List of correspondences for each image pair
        """
        # Ensure consistent precision for feature matching
        with torch.amp.autocast('cuda', enabled=False):
            H_A, W_A = rgbAs.shape[1:3]
            H_B, W_B = rgbBs.shape[1:3]

            batch_size = 1
            corres = []
            for b in tqdm(range(0, len(rgbAs), batch_size)):
                # Convert input to torch tensors if needed
                if not isinstance(rgbAs, torch.Tensor):
                    rgbAs_batch = torch.from_numpy(
                        rgbAs[b:b+batch_size]
                    ).to(torch.float32)
                else:
                    rgbAs_batch = rgbAs[b:b+batch_size].to(torch.float32)

                if not isinstance(rgbBs, torch.Tensor):
                    rgbBs_batch = torch.from_numpy(
                        rgbBs[b:b+batch_size]
                    ).to(torch.float32)
                else:
                    rgbBs_batch = rgbBs[b:b+batch_size].to(torch.float32)

                # Run matching with explicit float32 precision
                warp, certainty = self.match_bundlesdf(
                    rgbAs_batch,
                    rgbBs_batch,
                    device='cuda'
                )

                for warp_i, certainty_i in zip(warp, certainty):
                    # Ensure float32 precision for sampling
                    warp_i = warp_i.to(torch.float32)
                    certainty_i = certainty_i.to(torch.float32)

                    # Sample correspondences
                    matches, cert = self.model.sample(
                        warp_i, certainty_i, num=5000
                    )
                    kptsA, kptsB = self.model.to_pixel_coordinates(
                        matches, H_A, W_A, H_B, W_B
                    )

                    # Filter by confidence
                    score_mask = cert >= 1.0
                    kptsA = kptsA[score_mask]
                    kptsB = kptsB[score_mask]

                    # Convert to numpy
                    kptsA = kptsA.cpu().numpy()
                    kptsB = kptsB.cpu().numpy()

                    # Store correspondences
                    corres.append(
                        np.concatenate(
                            (kptsA.reshape(-1, 2), kptsB.reshape(-1, 2)),
                            axis=-1
                        ).reshape(-1, 4)
                    )

            # Clean up GPU memory
            if 'matches' in locals():
                del matches
            if 'certainty' in locals():
                del certainty
            if 'warp' in locals():
                del warp

            torch.cuda.empty_cache()

            return corres
