import os
import cv2
import numpy as np
import json
class ReconstructionDataLoader:
    """
    Data loader for multi-view object reconstruction using Bundle-SDF approach.
    
    This class manages loading and preprocessing of data for Bundle-SDF reconstruction,
    including RGB images, depth maps, and segmentation masks.
    
    Args:
        image_dir (str): Directory containing the dataset with subdirectories:
            - left/: RGB images
            - depth/: Corresponding depth maps
            - masks/: Segmentation masks
            - poses/: Camera poses
        config (Dict): Configuration dictionary containing:
            - camera_config: Camera parameters dictionary with:
                - intrinsic: Camera intrinsic matrix (flattened)
        downscale (float, optional): Scale factor to resize inputs (1.0 = original size).
            Defaults to 1.0.
        version (int, optional): Version of the dataloader implementation:
            - 1: Original implementation, returns (color, depth, mask)
            - 2: Enhanced implementation, returns (left, right, depth, mask, pose, id_str)
                where right and pose are None if not available
            Defaults to 1.
            
    Raises:
        ValueError: If directory structure is invalid or files cannot be found
        IOError: If image files cannot be read properly
    """
    def __init__(self, image_dir, config, downscale=1, version=1, min_resolution=300):
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory not found: {image_dir}")
            
        self.image_dir = image_dir
        self.downscale = downscale
        self.version = version
        
        if self.version not in [1, 2]:
            raise ValueError(f"Invalid version {version}. Must be 1 or 2.")
            
        # Validate and process camera intrinsics
        if 'camera_config' not in config or 'intrinsic' not in config['camera_config']:
            raise ValueError("Config must contain 'camera_config' with 'intrinsic' parameter")
            
        self.K = np.array(config['camera_config']['intrinsic']).reshape(3, 3)

        # Find and sort frame files
        left_dir = os.path.join(self.image_dir, 'left/')
        if not os.path.exists(left_dir):
            raise ValueError(f"Left image directory not found: {left_dir}")
            
        # Check for optional directories
        right_dir = os.path.join(self.image_dir, 'right/')
        depth_dir = os.path.join(self.image_dir, 'depth/')
        mask_dir = os.path.join(self.image_dir, 'masks/')
        pose_dir = os.path.join(self.image_dir, 'poses/')
        
        # Track which features are available
        self.has_right_images = os.path.exists(right_dir)
        self.has_depth_maps = os.path.exists(depth_dir)
        self.has_masks = os.path.exists(mask_dir)
        self.has_poses = os.path.exists(pose_dir)
        
        # Validate required directories based on version
        if self.version == 2:
            missing_dirs = []
            if not self.has_depth_maps:
                missing_dirs.append("depth/")
            if not self.has_masks:
                missing_dirs.append("masks/")
            
            if missing_dirs:
                print(f"Warning: Required directories missing for version 2: {', '.join(missing_dirs)}")
                print("Returning None for missing data fields")
        
        frame_names = [
            p for p in os.listdir(left_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
        ]
        
        if not frame_names:
            raise ValueError(f"No valid image files found in {left_dir}")
            
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0][4:]))
        
        self.color_files = [os.path.join(left_dir, file_name) for file_name in frame_names]

        # Extract frame IDs
        self.id_strs = []
        for color_file in self.color_files:
            id_str = os.path.basename(color_file)[:-4]
            self.id_strs.append(id_str)
        
        # Get image dimensions from first frame and apply downscaling
        first_img_path = self.color_files[0]
        first_img = cv2.imread(first_img_path)
        if first_img is None:
            raise IOError(f"Could not read image: {first_img_path}")
            
        self.H, self.W = first_img.shape[:2]
        if self.H < min_resolution or self.W < min_resolution:
            self.downscale = 1.0
        else:
            scale = min_resolution / min(self.H, self.W)
            self.downscale = max(self.downscale, scale)
        self.H = int(self.H * self.downscale)
        self.W = int(self.W * self.downscale)
        
        # Scale intrinsics according to downscale factor
        self.K[:2] *= self.downscale
        self.far = config['nerf']['far']

    def __len__(self):
        """Return the number of frames in the dataset."""
        return len(self.color_files)
    
    def get_color(self, idx):
        """
        Load and preprocess RGB image for the specified index.
        
        Args:
            idx (int): Index of the frame to retrieve
            
        Returns:
            np.ndarray: RGB image as np.uint8 with shape (H, W, 3)
            
        Raises:
            IndexError: If idx is out of range
            IOError: If image file cannot be read
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
            
        color_path = self.color_files[idx]
        color = cv2.imread(color_path)
        
        if color is None:
            raise IOError(f"Failed to load image: {color_path}")
            
        color = cv2.resize(color, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return color
    
    def get_right(self, idx):
        """
        Load and preprocess right RGB image for the specified index (if available).
        
        Args:
            idx (int): Index of the frame to retrieve
            
        Returns:
            np.ndarray or None: Right RGB image as np.uint8 with shape (H, W, 3),
                                or None if not available
            
        Raises:
            IndexError: If idx is out of range
        """
        if not self.has_right_images:
            return None
            
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
            
        right_path = self.color_files[idx].replace('left/', 'right/')
        
        if not os.path.exists(right_path):
            return None
            
        right = cv2.imread(right_path)
        if right is None:
            return None
            
        right = cv2.resize(right, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return right
    
    def get_depth(self, idx):
        """
        Load and preprocess depth map for the specified index.
        
        Args:
            idx (int): Index of the frame to retrieve
            
        Returns:
            np.ndarray or None: Depth map as np.float32 with shape (H, W),
                                or None if not available
            
        Raises:
            IndexError: If idx is out of range
        """
        if not self.has_depth_maps:
            return None
            
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
            
        depth_file = self.color_files[idx].replace('left/', 'depth/')
        
        # if not os.path.exists(depth_file):
        #     # check if npy file exists
        #     if os.path.exists(depth_file.replace('.png', '.npy')):
        #         depth_file = depth_file.replace('.png', '.npy')
        #     else:
        #         return None

        if os.path.exists(depth_file.replace('.png', '.npy')):
            depth_file = depth_file.replace('.png', '.npy')
        
        # Support multiple depth formats
        try:
            if os.path.splitext(depth_file)[1].lower() == '.npy':
                depth = np.load(depth_file)
            else:
                depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                if depth is None:
                    return None
                depth = depth.astype(np.float32) / 1000.0
                
            depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            return depth
        except Exception as e:
            print(f"Warning: Failed to load depth map {depth_file}: {e}")
            return None

    def get_mask(self, idx):
        """
        Load and preprocess segmentation mask for the specified index.
        
        Args:
            idx (int): Index of the frame to retrieve
            
        Returns:
            np.ndarray or None: Binary mask as np.uint8 with shape (H, W),
                               or None if not available
            
        Raises:
            IndexError: If idx is out of range
        """
        if not self.has_masks:
            return None
            
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
            
        mask_file = self.color_files[idx].replace('left/', 'masks/')
        
        if not os.path.exists(mask_file):
            return None
            
        try:
            mask = cv2.imread(mask_file, -1)
            
            if mask is None:
                return None
                
            # Ensure binary format
            if len(mask.shape) == 3:
                mask = (mask.sum(axis=-1) > 0).astype(np.uint8)
                
            mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8)) #add denoising and smoothing to masks
            return mask
        except Exception as e:
            print(f"Warning: Failed to load mask {mask_file}: {e}")
            return None
            
    def get_pose(self, idx):
        """
        Load camera pose for the specified index (if available).
        
        Args:
            idx (int): Index of the frame to retrieve
            
        Returns:
            np.ndarray or None: Camera pose matrix, or None if not available
            
        Raises:
            IndexError: If idx is out of range
        """
        if not self.has_poses:
            return None
            
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
            
        pose_file = self.color_files[idx].replace('left/', 'poses/').replace(
            os.path.splitext(self.color_files[idx])[1], '.json')
            
        if not os.path.exists(pose_file):
            return None
            
        try:
            with open(pose_file, 'r') as f:
                pose_data = json.load(f)
                
            # Assuming pose data is a flat list or nested list that can be converted to a matrix
            pose = np.array(pose_data)
            return pose
        except Exception as e:
            print(f"Warning: Failed to load pose {pose_file}: {e}")
            return None

    def __getitem__(self, idx):
        """
        Get preprocessed data for the specified index.
        
        Args:
            idx (int): Index of the frame to retrieve
            
        Returns:
            If version=1:
                Tuple containing:
                - RGB image (H, W, 3) as np.uint8
                - Depth map (H, W) as np.float32, or None if not available
                - Binary mask (H, W) as np.uint8, or None if not available
            
            If version=2:
                Tuple containing:
                - left: RGB image (H, W, 3) as np.uint8
                - right: RGB image (H, W, 3) as np.uint8, or None if not available
                - depth: Depth map (H, W) as np.float32, or None if not available
                - mask: Binary mask (H, W) as np.uint8, or None if not available
                - pose: Camera pose matrix as np.ndarray, or None if not available
                - id_str: ID string of the frame
            
        Raises:
            IndexError: If idx is out of range
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
            
        color = self.get_color(idx)
        depth = self.get_depth(idx)
        mask = self.get_mask(idx)
        id_str = self.id_strs[idx]
        
        if self.version == 1:
            return color, depth, mask
        elif self.version == 2:
            # Version 2 returns data in a format compatible with ReconstructionDataloader
            right = self.get_right(idx)
            pose = self.get_pose(idx)
            return color, right, depth, mask, pose, id_str
        
    def get_camera_intrinsics(self):
        """
        Get the camera intrinsics matrix adjusted for current downscale factor.
        
        Returns:
            np.ndarray: 3x3 camera intrinsics matrix
        """
        return self.K.copy()
    
    def get_image_dimensions(self):
        """
        Get the current image dimensions after downscaling.
        
        Returns:
            Tuple[int, int]: Height and width of the images
        """
        return self.H, self.W
        
    def get_frame_id(self, idx):
        """
        Get the ID string for the frame at the specified index.
        
        Args:
            idx (int): Index of the frame
            
        Returns:
            str: ID string of the frame
            
        Raises:
            IndexError: If idx is out of range
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
            
        return self.id_strs[idx]