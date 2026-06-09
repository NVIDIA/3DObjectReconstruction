from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from omegaconf import MISSING

"""BundleTrack and NVBundleSDF config schemas"""

@dataclass
class RoMaConfig:
    """RoMa configuration."""
    coarse_res: int = 560
    upsample_res: Tuple[int, int] = (864, 864)
    device: str = "cuda"
    weights: str = "/workspace/3d-object-reconstruction/data/weights/roma/roma_outdoor.pth"
    dinov2_weights: str = "/workspace/3d-object-reconstruction/data/weights/roma/dinov2_vitl14_pretrain.pth"
    

@dataclass
class CameraConfig:
    """Camera configuration."""
    step: int = 4
    intrinsic: List[float] = field(default_factory=lambda: [3.0796e+03, 0, 2.0000e+03, 0, 3.0751e+03, 1.50001e+03, 0, 0, 1])

@dataclass
class FoundationStereoConfig:
    """Foundation Stereo configuration."""
    # inference_uri: str = MISSING
    pth_path: str = '/workspace/3d-object-reconstruction/data/weights/foundationstereo/model_best_bp2.pth'
    cfg_path: str = '/workspace/3d-object-reconstruction/data/weights/foundationstereo/cfg.yaml'
    vit_size: str = 'vitl'
    scale: float = 0.3
    hiera: int = 0
    z_far: float = 10
    remove_invisible: bool = True
    intrinsic: List[float] = field(default_factory=lambda: [3.0796e+03, 0, 2.0000e+03, 0, 3.0751e+03, 1.50001e+03, 0, 0, 1])
    baseline: float = 0.0657696127

@dataclass
class SAM2Config:
    """SAM2 configuration."""
    checkpoint_path: str = "/workspace/3d-object-reconstruction/data/weights/sam2/sam2.1_hiera_large.pt"
    model_config: str = "/workspace/3d-object-reconstruction/data/weights/sam2/sam2.1_hiera_l.yaml"
    bbox: List[int] = field(default_factory=lambda: [1144, 627, 2227, 2232])
    device: str = "cuda"

@dataclass
class TextureBakeConfig:
    """Texture baking configuration."""
    downscale: float = 1.0
    texture_res: int = 2048

@dataclass
class SegmentationConfig:
    """Segmentation configuration."""
    ob_scales: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.3])
    tolerance: float = 0.03

@dataclass
class DepthProcessingConfig:
    """Depth processing configuration."""
    zfar: float = 1.0
    
    @dataclass
    class ErodeConfig:
        radius: int = 1
        diff: float = 0.001
        ratio: float = 0.8  # If ratio larger than this, depth set to 0
    
    erode: ErodeConfig = field(default_factory=ErodeConfig)
    
    @dataclass
    class BilateralFilterConfig:
        radius: int = 2
        sigma_D: int = 2
        sigma_R: int = 100000
    
    bilateral_filter: BilateralFilterConfig = field(default_factory=BilateralFilterConfig)
    
    @dataclass
    class OutlierRemovalConfig:
        num: int = 30
        std_mul: int = 3
    
    outlier_removal: OutlierRemovalConfig = field(default_factory=OutlierRemovalConfig)
    
    edge_normal_thres: int = 10  # Deg between normal and ray
    denoise_cloud: bool = False
    percentile: int = 95

@dataclass
class BundleConfig:
    num_iter_outter: int = 7
    num_iter_inner: int = 5
    window_size: int = 5  # Exclude keyframes, include new frame
    max_BA_frames: int = 10
    subset_selection_method: str = "normal_orientation_nearest"
    depth_association_radius: int = 5  # Used for depth point association
    non_neighbor_max_rot: int = 90
    non_neighbor_min_visible: float = 0.1   # Ratio of pixel visible
    icp_pose_rot_thres: int = 60    # Rotation larger than XX deg is ignored for icp
    w_rpi: int = 0
    w_p2p: int = 1    # Used in loss.cpp
    w_fm: int = 1
    w_sdf: int = 0
    w_pm: int = 0
    robust_delta: float = 0.005
    min_fm_edges_newframe: int = 15
    image_downscale: List[int] = field(default_factory=lambda: [4])
    feature_edge_dist_thres: float = 0.01
    feature_edge_normal_thres: int = 30   # Normal angle should be within this range
    max_optimized_feature_loss: float = 0.03

@dataclass
class KeyframeConfig:
    min_interval: int = 1
    min_feat_num: int = 0
    min_trans: int = 0
    min_rot: int = 5
    min_visible: int = 1

@dataclass
class SiftConfig:
    scales: List[int] = field(default_factory=lambda: [2, 4, 8])
    max_match_per_query: int = 5
    nOctaveLayers: int = 3
    contrastThreshold: float = 0.01
    edgeThreshold: int = 50
    sigma: float = 1.6

@dataclass
class FeatureCorresConfig:
    mutual: bool = True
    map_points: bool = True
    max_dist_no_neighbor: float = 0.01
    max_normal_no_neighbor: int = 20
    max_dist_neighbor: float = 0.02
    max_normal_neighbor: int = 30
    suppression_patch_size: int = 5
    max_view_normal_angle: int = 180
    min_match_with_ref: int = 5
    resize: int = 800
    rematch_after_nerf: bool = False

@dataclass
class RansacConfig:
    max_iter: int = 2000
    num_sample: int = 3
    inlier_dist: float = 0.01
    inlier_normal_angle: int = 20
    desired_succ_rate: float = 0.99
    max_trans_neighbor: float = 0.02   # ransac model estimated pose shouldnt be too far
    max_rot_deg_neighbor: int = 30
    max_trans_no_neighbor: float = 0.01
    max_rot_no_neighbor: int = 10
    epipolar_thres: int = 1
    min_match_after_ransac: int = 5

@dataclass
class P2PConfig:
    projective: bool = False
    max_dist: float = 0.02
    max_normal_angle: int = 45

@dataclass
class SDFEdgeConfig:
    max_dist: float = 0.02

@dataclass
class ShapeConfig:
    res: float = 0.005
    xrange: Tuple[float, float] = (-0.2, 0.2)
    yrange: Tuple[float, float] = (-0.2, 0.2)
    zrange: Tuple[float, float] = (-0.2, 0.2)
    max_weight: int = 100

@dataclass
class BundleTrackConfig:
    debug_dir: str = MISSING
    SPDLOG: int = 2
    USE_GRAY: bool = False
    port: str = "5555"
    nerf_port: str = "9999"
    downscale: float = 1.0
    erode_mask: int = 3
    visible_angle: int = 70  # Angle between normal and point to camera origin within XXX is regarded as visible
    
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    depth_processing: DepthProcessingConfig = field(default_factory=DepthProcessingConfig)
    bundle: BundleConfig = field(default_factory=BundleConfig)
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)
    sift: SiftConfig = field(default_factory=SiftConfig)
    feature_corres: FeatureCorresConfig = field(default_factory=FeatureCorresConfig)
    ransac: RansacConfig = field(default_factory=RansacConfig)
    p2p: P2PConfig = field(default_factory=P2PConfig)
    sdf_edge: SDFEdgeConfig = field(default_factory=SDFEdgeConfig)
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    

@dataclass
class NeRFConfig:
    """NeRF configuration."""
    batch_size: int = 32
    downscale: float = 0.5
    n_step: int = 2000
    save_dir: str = MISSING
    
    # Network architecture
    netdepth: int = 8
    netwidth: int = 256
    netdepth_fine: int = 8
    netwidth_fine: int = 256
    
    # Training parameters
    N_rand: int = 2048
    lrate: float = 0.01
    lrate_pose: float = 0.01
    decay_rate: float = 0.1
    chunk: int = 99999999999
    netchunk: int = 6553600
    no_batching: int = 0
    amp: bool = True
    
    # Sampling parameters
    N_samples: int = 64
    N_samples_around_depth: int = 256
    N_importance: int = 0
    perturb: int = 1
    use_viewdirs: int = 1
    
    # Embedding parameters
    i_embed: int = 1
    i_embed_views: int = 2
    multires: int = 8
    multires_views: int = 3
    feature_grid_dim: int = 2
    raw_noise_std: int = 0
    
    # Logging options
    i_img: int = 99999
    i_weights: int = 999999
    i_mesh: int = 999999
    i_pose: int = 999999
    i_print: int = 999999
    
    # Hash encoding parameters
    finest_res: int = 256
    base_res: int = 16
    num_levels: int = 16
    log2_hashmap_size: int = 22
    
    # Octree parameters
    use_octree: int = 1
    first_frame_weight: int = 1
    denoise_depth_use_octree_cloud: bool = True
    octree_embed_base_voxel_size: float = 0.02
    octree_smallest_voxel_size: float = 0.02
    octree_raytracing_voxel_size: float = 0.02
    octree_dilate_size: float = 0.02
    down_scale_ratio: int = 1
    
    # Scene parameters
    bounding_box: List[List[float]] = field(default_factory=lambda: [[-1, -1, -1], [1, 1, 1]])
    use_mask: int = 1
    dilate_mask_size: int = 0
    rays_valid_depth_only: bool = True
    near: float = 0.1
    far: float = 1.0
    
    # Loss weights
    rgb_weight: int = 10
    depth_weight: int = 0
    sdf_lambda: int = 5
    trunc: float = 0.002
    trunc_start: float = 0.002
    neg_trunc_ratio: int = 1
    trunc_decay_type: str = ""
    fs_weight: int = 100
    empty_weight: int = 2
    fs_rgb_weight: int = 0
    fs_sdf: float = 0.1
    trunc_weight: int = 6000
    tv_loss_weight: int = 0
    frame_features: int = 2
    optimize_poses: int = 0
    pose_reg_weight: int = 0
    feature_reg_weight: float = 0.1
    share_coarse_fine: int = 1
    eikonal_weight: int = 0
    
    # Rendering mode and mesh extraction
    mode: str = "sdf"
    mesh_resolution: float = 0.002
    max_trans: float = 0.02
    max_rot: int = 20
    
    
    @dataclass
    class MeshSmoothingConfig:
        enabled: bool = True
        iterations: int = 2
        lambda_: float = 0.5
        use_taubin: bool = True
    
    mesh_smoothing: MeshSmoothingConfig = field(default_factory=MeshSmoothingConfig)
    save_octree_clouds: bool = True

@dataclass
class BasePathConfig:
    """Base path configuration."""
    base_folder: str = MISSING
    image_folder: str = MISSING
    save_dir: str = MISSING

@dataclass
class NVBundleSDFConfig:
    """NVBundleSDF configuration."""
    data_path: str = MISSING
    workdir: str = MISSING
    downscale: float = 1.0
    camera_config: CameraConfig = field(default_factory=CameraConfig)
    bundletrack: BundleTrackConfig = field(default_factory=BundleTrackConfig)
    foundation_stereo: FoundationStereoConfig = field(default_factory=FoundationStereoConfig)
    sam2: SAM2Config = field(default_factory=SAM2Config)
    nerf: NeRFConfig = field(default_factory=NeRFConfig)
    texture_bake: TextureBakeConfig = field(default_factory=TextureBakeConfig)
    roma: RoMaConfig = field(default_factory=RoMaConfig)
    base_path: BasePathConfig = field(default_factory=BasePathConfig)
    