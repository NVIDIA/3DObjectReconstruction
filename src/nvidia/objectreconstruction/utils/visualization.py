import os
import cv2
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
import torch
import math
import copy
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
# Optional Kaolin import
try:
    import kaolin as kal
    KAOLIN_AVAILABLE = True
except ImportError:
    KAOLIN_AVAILABLE = False

class MinimalPreviewWidget:
    """
    Minimal preview widget for stereo, depth, mask, and combined views.
    Lazy-loads data and provides a simple slider UI for frame navigation.
    """
    def __init__(self, folder_path, mode='stereo'):
        """
        Initialize minimal preview widget that auto-loads data.

        Args:
            folder_path (str): Path to data folder
            mode (str): Display mode ('stereo', 'depth', 'mask', 'combined',
                'left', 'right')
        """
        self.folder_path = Path(folder_path)
        self.mode = mode
        self.left_image_paths = []
        self.right_image_paths = []
        self.depth_map_paths = []
        self.mask_paths = []
        self.data_loaded = False
        self.frame_slider = widgets.IntSlider(
            value=0, min=0, max=0, step=1, description='Frame:',
            continuous_update=False, layout=widgets.Layout(width='500px')
        )
        self.output = widgets.Output()
        self.frame_slider.observe(self.update_display, names='value')
        self.load_data()

    def get_file_paths_from_folder(self, subfolder):
        """Get file paths from a specific subfolder."""
        folder = self.folder_path / subfolder
        if not folder.exists():
            return []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.npy']
        files = []
        for ext in image_extensions:
            files.extend(folder.glob(f'*{ext}'))
            files.extend(folder.glob(f'*{ext.upper()}'))
        
        files.sort()
        return files

    def load_single_file(self, file_path):
        """Load a single file (image or numpy array)."""
        if not file_path or not file_path.exists():
            return None
        
        try:
            if file_path.suffix.lower() == '.npy':
                return np.load(file_path)
            
            # Try loading with OpenCV first (handles most formats)
            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Convert BGR to RGB for color images
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
            
            # Fallback: try with PIL for additional format support
            try:
                from PIL import Image
                pil_img = Image.open(file_path)
                return np.array(pil_img)
            except Exception:
                pass
            
            return None
            
        except Exception:
            return None

    def load_data(self):
        """Load file paths only (lazy loading)."""
        if not self.folder_path.exists():
            with self.output:
                print(f"Folder does not exist: {self.folder_path}")
            return
        
        self.left_image_paths = self.get_file_paths_from_folder('left')
        self.right_image_paths = self.get_file_paths_from_folder('right')
        self.depth_map_paths = self.get_file_paths_from_folder('depth')
        self.mask_paths = self.get_file_paths_from_folder('masks')
        
        if not self.left_image_paths:
            with self.output:
                print("No images found in left/ folder")
            return
        
        num_frames = len(self.left_image_paths)
        
        # Ensure all arrays have the same length
        for arr in [self.right_image_paths, self.depth_map_paths, self.mask_paths]:
            while len(arr) < num_frames:
                arr.append(None)
        
        self.frame_slider.max = num_frames - 1
        self.frame_slider.value = 0
        self.data_loaded = True
        
        with self.output:
            print(f"Found {num_frames} frames (lazy loading enabled)")
            if self.mode == 'mask':
                mask_count = sum(1 for x in self.mask_paths if x is not None)
                print(f"Mask files: {mask_count}/{num_frames}")
                if mask_count == 0:
                    print("Warning: No mask files found. Expected masks in 'masks/' subfolder")
        
        self.update_display()

    def normalize_depth(self, depth_map):
        """Normalize depth map for visualization."""
        if depth_map is None:
            return None
        valid_mask = (
            (depth_map > 0) &
            (depth_map < np.percentile(depth_map[depth_map > 0], 95))
        )
        if not np.any(valid_mask):
            return depth_map
        min_depth = np.min(depth_map[valid_mask])
        max_depth = np.max(depth_map[valid_mask])
        if max_depth > min_depth:
            normalized = (depth_map - min_depth) / (max_depth - min_depth)
            normalized[~valid_mask] = 0
            return normalized
        return depth_map

    def process_mask(self, mask):
        """Process mask for visualization with robust format handling."""
        if mask is None:
            return None
        
        try:
            # Handle multi-channel masks (convert to single channel)
            if len(mask.shape) == 3:
                if mask.shape[2] == 1:
                    # Single channel stored as (H, W, 1)
                    mask = mask.squeeze(-1)
                else:
                    # Multi-channel - use any non-zero pixel across channels
                    mask = (mask.sum(axis=-1) > 0).astype(np.uint8)
            
            # Handle different data types and value ranges
            if mask.dtype == bool:
                # Boolean mask: convert to 0-255
                processed_mask = (mask * 255).astype(np.uint8)
            elif mask.dtype in [np.float32, np.float64]:
                # Float mask: assume 0.0-1.0 range, convert to 0-255
                processed_mask = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
            elif mask.max() <= 1 and mask.min() >= 0:
                # Integer mask with 0-1 values: convert to 0-255
                processed_mask = (mask * 255).astype(np.uint8)
            else:
                # Other integer masks: ensure proper range
                if mask.max() > 255:
                    # Normalize to 0-255 range
                    processed_mask = ((mask - mask.min()) / (mask.max() - mask.min()) * 255).astype(np.uint8)
                else:
                    processed_mask = mask.astype(np.uint8)
            
            return processed_mask
            
        except Exception:
            # Fallback: try simple conversion
            try:
                return (mask > 0).astype(np.uint8) * 255
            except Exception:
                return None

    def update_display(self, change=None):
        """Update the display based on current frame."""
        if not self.data_loaded:
            return
        
        with self.output:
            clear_output(wait=True)
            frame_idx = self.frame_slider.value
            
            left_path = (
                self.left_image_paths[frame_idx]
                if frame_idx < len(self.left_image_paths) else None
            )
            right_path = (
                self.right_image_paths[frame_idx]
                if frame_idx < len(self.right_image_paths) else None
            )
            depth_path = (
                self.depth_map_paths[frame_idx]
                if frame_idx < len(self.depth_map_paths) else None
            )
            mask_path = (
                self.mask_paths[frame_idx]
                if frame_idx < len(self.mask_paths) else None
            )
            
            left_img = self.load_single_file(left_path)
            right_img = self.load_single_file(right_path)
            depth_map = self.load_single_file(depth_path)
            mask = self.load_single_file(mask_path)
            
            if self.mode == 'stereo':
                self.show_stereo(left_img, right_img, frame_idx)
            elif self.mode == 'depth':
                self.show_depth(depth_map, frame_idx)
            elif self.mode == 'mask':
                self.show_mask(mask, left_img, frame_idx)
            elif self.mode == 'combined':
                self.show_combined(left_img, right_img, depth_map, mask, frame_idx)
            elif self.mode == 'left':
                self.show_single(left_img, 'Left', frame_idx)
            elif self.mode == 'right':
                self.show_single(right_img, 'Right', frame_idx)

    def show_stereo(self, left_img, right_img, frame_idx):
        """Show stereo pair."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        if left_img is not None:
            axes[0].imshow(left_img, cmap='gray' if len(left_img.shape) == 2 else None)
            axes[0].set_title(f'Left Frame {frame_idx}')
            axes[0].axis('off')
        else:
            axes[0].text(0.5, 0.5, 'No Left Image', ha='center', va='center',
                         transform=axes[0].transAxes)
            axes[0].set_title(f'Left Frame {frame_idx} (Missing)')
        if right_img is not None:
            axes[1].imshow(right_img, cmap='gray' if len(right_img.shape) == 2 else None)
            axes[1].set_title(f'Right Frame {frame_idx}')
            axes[1].axis('off')
        else:
            axes[1].text(0.5, 0.5, 'No Right Image', ha='center', va='center',
                         transform=axes[1].transAxes)
            axes[1].set_title(f'Right Frame {frame_idx} (Missing)')
        plt.tight_layout()
        plt.show()

    def show_depth(self, depth_map, frame_idx):
        """Show depth map."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if depth_map is not None:
            normalized_depth = self.normalize_depth(depth_map)
            im = ax.imshow(normalized_depth, cmap='viridis')
            ax.set_title(f'Depth Map - Frame {frame_idx}')
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Normalized Depth')
        else:
            ax.text(0.5, 0.5, 'No Depth Map', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'Depth Map - Frame {frame_idx} (Missing)')
        plt.tight_layout()
        plt.show()

    def show_mask(self, mask, background_img, frame_idx):
        """Show segmentation mask with robust error handling."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Show background image
            if background_img is not None:
                axes[0].imshow(
                    background_img,
                    cmap='gray' if len(background_img.shape) == 2 else None
                )
                axes[0].set_title(f'Original - Frame {frame_idx}')
                axes[0].axis('off')
            else:
                axes[0].text(0.5, 0.5, 'No Background Image', ha='center',
                             va='center', transform=axes[0].transAxes)
                axes[0].set_title(f'Original - Frame {frame_idx} (Missing)')
            
            # Show mask
            if mask is not None:
                processed_mask = self.process_mask(mask)
                
                if processed_mask is not None:
                    axes[1].imshow(processed_mask, cmap='gray', vmin=0, vmax=255)
                    axes[1].set_title(f'Segmentation Mask - Frame {frame_idx}')
                    axes[1].axis('off')
                else:
                    axes[1].text(0.5, 0.5, 'Mask Processing Failed', ha='center', va='center',
                                 transform=axes[1].transAxes, color='red')
                    axes[1].set_title(f'Mask - Frame {frame_idx} (Processing Error)')
            else:
                axes[1].text(0.5, 0.5, 'No Mask Available', ha='center', va='center',
                             transform=axes[1].transAxes)
                axes[1].set_title(f'Mask - Frame {frame_idx} (Missing)')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error displaying mask: {e}")
            # Still try to show something
            try:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.text(0.5, 0.5, f'Error loading mask for frame {frame_idx}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Mask Viewer Error - Frame {frame_idx}')
                plt.show()
            except:
                pass

    def show_single(self, img, img_type, frame_idx):
        """Show single image."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if img is not None:
            ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            ax.set_title(f'{img_type} - Frame {frame_idx}')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'No {img_type} Image', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{img_type} - Frame {frame_idx} (Missing)')
        plt.tight_layout()
        plt.show()

    def show_combined(self, left_img, right_img, depth_map, mask, frame_idx):
        """Show combined view of all data."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        # Left image
        if left_img is not None:
            axes[0, 0].imshow(left_img, cmap='gray' if len(left_img.shape) == 2 else None)
            axes[0, 0].set_title('Left Image')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Left Image', ha='center', va='center',
                            transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Left Image (Missing)')
        axes[0, 0].axis('off')
        # Right image
        if right_img is not None:
            axes[0, 1].imshow(right_img, cmap='gray' if len(right_img.shape) == 2 else None)
            axes[0, 1].set_title('Right Image')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Right Image', ha='center', va='center',
                            transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Right Image (Missing)')
        axes[0, 1].axis('off')
        # Depth map
        if depth_map is not None:
            normalized_depth = self.normalize_depth(depth_map)
            im = axes[1, 0].imshow(normalized_depth, cmap='viridis')
            axes[1, 0].set_title('Depth Map')
            cbar = plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
            cbar.set_label('Depth', fontsize=8)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Depth Map', ha='center', va='center',
                            transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Depth Map (Missing)')
        axes[1, 0].axis('off')
        # Mask
        if mask is not None:
            processed_mask = self.process_mask(mask)
            if processed_mask is not None:
                axes[1, 1].imshow(processed_mask, cmap='gray')
                axes[1, 1].set_title('Segmentation Mask')
            else:
                axes[1, 1].text(0.5, 0.5, 'Mask Processing Failed', ha='center', va='center',
                                transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Mask (Processing Error)')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Mask', ha='center', va='center',
                            transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Mask (Missing)')
        axes[1, 1].axis('off')
        plt.suptitle(f'Combined View - Frame {frame_idx}', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()

    def display(self):
        """Display the minimal widget."""
        return widgets.VBox([self.frame_slider, self.output])

class BoundingBoxWidget:
    """
    Widget for interactively drawing or entering a bounding box on the first
    image in a folder. Provides UI for drawing, resetting, finalizing, and
    manual input of bounding box coordinates.
    """
    def __init__(self, folder_path, default_bbox=None, default_bbox_fraction=0.70):
        """
        Initialize bounding box widget that loads the first frame from left folder.

        Args:
            folder_path (str): Path to data folder containing 'left' subfolder
            default_bbox (tuple, optional): Default bounding box as (x, y, width, height).
                                          If None, will create centered bbox using default_bbox_fraction.
            default_bbox_fraction (float): Fraction of image to use for default bbox (0.5-1.0).
                                         Only used if default_bbox is None.
        """
        self.folder_path = Path(folder_path)
        self.default_bbox = default_bbox
        self.default_bbox_fraction = max(0.5, min(1.0, default_bbox_fraction))  # Clamp between 0.5 and 1.0
        self.bbox_coords = None  # Will be set after image is loaded
        self.image = None
        self.fig = None
        self.ax = None
        self.selector = None
        self.finalized = False
        
        # Check if SAMPLES_MODE environment variable is set
        self.samples_mode = os.getenv('SAMPLES_MODE') is not None
        self.instruction_label = widgets.HTML(
            value="<b>Instructions:</b> Click and drag to draw a bounding box "
                  "on the image below."
        )
        self.coords_label = widgets.HTML(
            value="<b>Bounding Box:</b> None selected"
        )
        self.reset_button = widgets.Button(
            description="Reset to Default",
            button_style='warning',
            tooltip="Reset bounding box to default position"
        )
        self.get_coords_button = widgets.Button(
            description="Get Coordinates",
            button_style='success',
            tooltip="Print the current bounding box coordinates"
        )
        self.finalize_button = widgets.Button(
            description="Finalize & Close",
            button_style='primary',
            tooltip="Finalize the bounding box and close the widget"
        )
        self.manual_input_button = widgets.Button(
            description="Manual Input",
            button_style='info',
            tooltip="Enter coordinates manually if drawing doesn't work"
        )
        self.output = widgets.Output()
        self.reset_button.on_click(self.reset_bbox)
        self.get_coords_button.on_click(self.print_coordinates)
        self.finalize_button.on_click(self.finalize_and_close)
        self.manual_input_button.on_click(self.manual_input)
        self.load_first_image()

    def load_first_image(self):
        """Load the first image from the left folder."""
        left_folder = self.folder_path / 'left'
        if not left_folder.exists():
            with self.output:
                print(f"Left folder does not exist: {left_folder}")
            return
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(left_folder.glob(f'*{ext}'))
            image_files.extend(left_folder.glob(f'*{ext.upper()}'))
        image_files.sort()
        if not image_files:
            with self.output:
                print("No image files found in left folder")
            return
        first_image_path = image_files[0]
        self.image = cv2.imread(str(first_image_path), cv2.IMREAD_UNCHANGED)
        if self.image is None:
            with self.output:
                print(f"Failed to load image: {first_image_path}")
            return
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Always set a default bounding box
        self._set_default_bbox()
        
        # If SAMPLES_MODE is set, keep it non-interactive, otherwise setup interactive plot
        if not self.samples_mode:
            self.setup_interactive_plot()

    def _set_default_bbox(self):
        """Set a default bounding box based on provided parameters or image-relative sizing."""
        height, width = self.image.shape[:2]
        
        if self.default_bbox is not None:
            # Use provided default bbox
            self.bbox_coords = tuple(self.default_bbox)
            bbox_source = "provided default"
        else:
            # Create a default bounding box using the specified fraction of the image
            margin_x = int(width * (1.0 - self.default_bbox_fraction) / 2)
            margin_y = int(height * (1.0 - self.default_bbox_fraction) / 2)
            
            x = margin_x
            y = margin_y
            bbox_width = width - 2 * margin_x
            bbox_height = height - 2 * margin_y
            
            self.bbox_coords = (x, y, bbox_width, bbox_height)
            bbox_source = f"centered {int(self.default_bbox_fraction * 100)}% of image"
        
        # Update the label
        x, y, w, h = self.bbox_coords
        label_text = f"<b>Default Bounding Box:</b> x={x}, y={y}, width={w}, height={h}"
        if self.samples_mode:
            label_text = f"<b>Bounding Box (SAMPLES_MODE):</b> x={x}, y={y}, width={w}, height={h}"
        
        self.coords_label.value = label_text
        
        with self.output:
            clear_output(wait=True)
            if self.samples_mode:
                print("=" * 50)
                print("SAMPLES_MODE DETECTED - DEFAULT BOUNDING BOX SET")
                print("=" * 50)
            else:
                print("=" * 50)
                print("DEFAULT BOUNDING BOX SET")
                print("=" * 50)
            print(f"Default bounding box coordinates: {self.bbox_coords}")
            print(f"Format: (x, y, width, height)")
            print(f"Source: {bbox_source}")
            print(f"Image dimensions: {width} x {height}")
            print("=" * 50)

    def setup_interactive_plot(self):
        """Setup matplotlib plot with interactive rectangle selector."""
        with self.output:
            clear_output(wait=True)
            import matplotlib
            matplotlib.use('widget')
            self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
            self.ax.imshow(
                self.image,
                cmap='gray' if len(self.image.shape) == 2 else None
            )
            self.ax.set_title('Draw Bounding Box - Click and Drag (Default box shown in red)')
            self.ax.axis('on')
            
            # Show the default bounding box
            if self.bbox_coords:
                x, y, width, height = self.bbox_coords
                rect = Rectangle((x, y), width, height, 
                               linewidth=2, edgecolor='red', facecolor='none', 
                               linestyle='--', alpha=0.7)
                self.ax.add_patch(rect)
            
            self.selector = RectangleSelector(
                self.ax,
                self.on_select,
                useblit=False,
                button=[1],
                minspanx=10, minspany=10,
                spancoords='pixels',
                interactive=True,
                drag_from_anywhere=True
            )
            plt.tight_layout()
            plt.show()

    def on_select(self, eclick, erelease):
        """Callback for rectangle selection."""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in [x1, y1, x2, y2]:
            return
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        self.bbox_coords = (int(x), int(y), int(width), int(height))
        self.coords_label.value = (
            f"<b>Bounding Box:</b> x={int(x)}, y={int(y)}, "
            f"width={int(width)}, height={int(height)}"
        )

    def reset_bbox(self, button):
        """Reset the bounding box to default."""
        self._set_default_bbox()
        if self.selector:
            self.selector.set_visible(False)
        if self.fig:
            # Clear existing rectangles and redraw with default
            for patch in self.ax.patches:
                patch.remove()
            
            # Add default bbox visualization
            if self.bbox_coords:
                x, y, width, height = self.bbox_coords
                rect = Rectangle((x, y), width, height, 
                               linewidth=2, edgecolor='red', facecolor='none', 
                               linestyle='--', alpha=0.7)
                self.ax.add_patch(rect)
            
            self.fig.canvas.draw()

    def print_coordinates(self, button):
        """Print the current coordinates."""
        if self.bbox_coords:
            print(f"Bounding box coordinates: {self.bbox_coords}")
            print(f"Format: (x, y, width, height) = {self.bbox_coords}")
        else:
            print("No bounding box selected")

    def finalize_and_close(self, button):
        """Finalize the bounding box selection and close the widget."""
        if self.bbox_coords:
            with self.output:
                clear_output(wait=True)
                print("=" * 50)
                print("BOUNDING BOX FINALIZED")
                print("=" * 50)
                print(f"Coordinates: {self.bbox_coords}")
                print(f"Format: (x, y, width, height)")
                print(f"Top-left corner: ({self.bbox_coords[0]}, "
                      f"{self.bbox_coords[1]})")
                print(f"Bottom-right corner: "
                      f"({self.bbox_coords[0] + self.bbox_coords[2]}, "
                      f"{self.bbox_coords[1] + self.bbox_coords[3]})")
                print(f"Size: {self.bbox_coords[2]} x {self.bbox_coords[3]} "
                      f"pixels")
                print("=" * 50)
                print("Widget closed. Access coordinates with: "
                      "widget_variable.get_bbox()")
                print("=" * 50)
            if hasattr(self, 'fig') and self.fig:
                plt.close(self.fig)
            self.finalized = True
        else:
            print("No bounding box selected. Please draw a bounding box first.")

    def manual_input(self, button):
        """Allow manual input of coordinates."""
        with self.output:
            print("Manual coordinate input:")
            print("Please enter coordinates in the format: x,y,width,height")
            print("Example: 100,50,200,150\n")
            coord_input = widgets.Text(
                placeholder="x,y,width,height (e.g., 100,50,200,150)",
                description="Coordinates:",
                style={'description_width': 'initial'}
            )
            submit_button = widgets.Button(
                description="Set Coordinates",
                button_style='primary'
            )
            def on_submit(b):
                try:
                    coords_str = coord_input.value.strip()
                    coords = [int(x.strip()) for x in coords_str.split(',')]
                    if len(coords) == 4:
                        self.bbox_coords = tuple(coords)
                        x, y, w, h = coords
                        self.coords_label.value = (
                            f"<b>Bounding Box:</b> x={x}, y={y}, "
                            f"width={w}, height={h}")
                        print(f"Coordinates set: {self.bbox_coords}")
                    else:
                        print("Error: Please enter exactly 4 numbers separated "
                              "by commas")
                except ValueError:
                    print("Error: Please enter valid numbers separated by "
                          "commas")
            submit_button.on_click(on_submit)
            display(widgets.VBox([coord_input, submit_button]))

    def get_bbox(self):
        """Get the current bounding box coordinates."""
        return self.bbox_coords

    def get_bbox_dict(self):
        """Get bounding box as a dictionary."""
        if self.bbox_coords:
            x, y, width, height = self.bbox_coords
            return {
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'x2': x + width,
                'y2': y + height
            }
        return None

    def is_finalized(self):
        """Check if the bounding box selection has been finalized."""
        return self.finalized

    def display(self):
        """Display the bounding box widget."""
        if self.samples_mode:
            # In SAMPLES_MODE, show minimal interface
            return widgets.VBox([
                widgets.HTML(value="<b>SAMPLES_MODE:</b> Default bounding box automatically set"),
                self.coords_label,
                widgets.HBox([self.get_coords_button]),
                self.output
            ])
        else:
            # Normal interactive mode
            controls = widgets.HBox([
                self.reset_button, self.get_coords_button,
                self.finalize_button, self.manual_input_button
            ])
            return widgets.VBox([
                self.instruction_label,
                self.coords_label,
                controls,
                self.output
            ])

# Simplified factory functions
def create_stereo_viewer(folder_path):
    """Create a minimal stereo viewer that auto-loads"""
    widget = MinimalPreviewWidget(folder_path, mode='stereo')
    return widget.display()

def create_depth_viewer(folder_path):
    """Create a minimal depth viewer that auto-loads"""
    widget = MinimalPreviewWidget(folder_path, mode='depth')
    return widget.display()

def create_mask_viewer(folder_path):
    """Create a minimal mask viewer that auto-loads"""
    widget = MinimalPreviewWidget(folder_path, mode='mask')
    return widget.display()

def create_combined_viewer(folder_path):
    """Create a minimal combined viewer that auto-loads"""
    widget = MinimalPreviewWidget(folder_path, mode='combined')
    return widget.display()

def create_left_viewer(folder_path):
    """Create a minimal left-only viewer that auto-loads"""
    widget = MinimalPreviewWidget(folder_path, mode='left')
    return widget.display()

def create_right_viewer(folder_path):
    """Create a minimal right-only viewer that auto-loads"""
    widget = MinimalPreviewWidget(folder_path, mode='right')
    return widget.display()

def create_bbox_widget(folder_path, default_bbox=None, default_bbox_fraction=0.8):
    """
    Create a bounding box drawing widget that auto-loads the first left image.
    
    Args:
        folder_path (str): Path to data folder containing 'left' subfolder
        default_bbox (tuple, optional): Default bounding box as (x, y, width, height).
                                      If None, will create centered bbox using default_bbox_fraction.
        default_bbox_fraction (float): Fraction of image to use for default bbox (0.5-1.0).
                                     Only used if default_bbox is None.
    
    If the SAMPLES_MODE environment variable is set, returns a widget with
    a default bounding box instead of requiring user interaction.
    """
    widget = BoundingBoxWidget(folder_path, default_bbox, default_bbox_fraction)
    return widget


# Legacy functions moved to Mesh3DViewer class
    
def create_3d_viewer(textured_mesh_path, texture_path=None, material_path=None):
    """Create a 3D viewer that auto-loads"""
    return Mesh3DViewer(textured_mesh_path, texture_path, material_path)



        
    
    
class Mesh3DViewer:
    """
    3D mesh viewer using Kaolin's turntable visualizer.
    """
    def __init__(self, textured_mesh_path, texture_path=None, material_path=None):
        self.textured_mesh_path = textured_mesh_path
        self.texture_path = texture_path
        self.material_path = material_path
        
        # Load and setup mesh
        self.mesh = self._load_mesh()
        
        # Setup camera
        self.camera = self._setup_camera()
        
        # Setup lighting parameters
        self.apply_lighting = True
        self.azimuth = torch.zeros((1,), dtype=torch.float32, device='cuda')
        self.elevation = torch.full((1,), math.pi / 3., dtype=torch.float32, device='cuda')
        self.amplitude = torch.full((1, 3), 3., dtype=torch.float32, device='cuda')
        self.sharpness = torch.full((1,), 5., dtype=torch.float32, device='cuda')
        
        # Final validation of all tensors
        self._validate_tensors()
        
        # Create visualizer
        self.visualizer = kal.visualize.IpyTurntableVisualizer(
            512, 512, copy.deepcopy(self.camera), self.render, 
            fast_render=self.make_lowres_render_func(self.render),
            max_fps=24, world_up_axis=1
        )
    
    def _load_mesh(self):
        """Load and process the mesh"""
        mesh = kal.io.obj.import_mesh(
            self.textured_mesh_path, 
            with_materials=True, 
            with_normals=True, 
            triangulate=True,
            raw_materials=False  # convert to PBR material
        )
        
        # Ensure all mesh data is in the correct format for nvdiffrast
        if hasattr(mesh, 'vertices') and mesh.vertices is not None:
            mesh.vertices = mesh.vertices.float()
        if hasattr(mesh, 'faces') and mesh.faces is not None:
            mesh.faces = mesh.faces.int()
        if hasattr(mesh, 'uvs') and mesh.uvs is not None:
            mesh.uvs = mesh.uvs.float()
        if hasattr(mesh, 'face_uvs_idx') and mesh.face_uvs_idx is not None:
            mesh.face_uvs_idx = mesh.face_uvs_idx.int()
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            mesh.vertex_normals = mesh.vertex_normals.float()
        if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
            mesh.face_normals = mesh.face_normals.float()
            
        return self.process_mesh(mesh)
    
    def _setup_camera(self):
        """Setup the camera"""
        return kal.render.camera.Camera.from_args(
            eye=torch.tensor([2., 1., 1.], dtype=torch.float32, device='cuda'),
            at=torch.tensor([0., 0., 0.], dtype=torch.float32, device='cuda'),
            up=torch.tensor([1., 1., 1.], dtype=torch.float32, device='cuda'),
            fov=math.pi * 45 / 180,
            width=512, height=512, device='cuda'
        )
    
    def process_mesh(self, mesh):
        """Process the mesh for rendering"""
        if mesh.materials is not None and len(mesh.materials) > 0:
            print(str(mesh.materials[0]))
            
        # Batch, move to GPU and center and normalize vertices in the range [-0.5, 0.5]
        mesh = mesh.to_batched().cuda()
        
        # Ensure all mesh tensors are in the correct dtype and device
        mesh.vertices = kal.ops.pointcloud.center_points(mesh.vertices, normalize=True).float()
        
        # Explicitly ensure faces are int32 for nvdiffrast
        if hasattr(mesh, 'faces') and mesh.faces is not None:
            mesh.faces = mesh.faces.int()
            
        # Ensure other attributes are also float32
        if hasattr(mesh, 'uvs') and mesh.uvs is not None:
            mesh.uvs = mesh.uvs.float()
        if hasattr(mesh, 'face_uvs_idx') and mesh.face_uvs_idx is not None:
            mesh.face_uvs_idx = mesh.face_uvs_idx.int()
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            mesh.vertex_normals = mesh.vertex_normals.float()
        if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
            mesh.face_normals = mesh.face_normals.float()
            
        print(f"Mesh vertices dtype: {mesh.vertices.dtype}")
        print(f"Mesh faces dtype: {mesh.faces.dtype}")
        print(mesh)
        return mesh
    
    def current_lighting(self):
        """Get current lighting parameters"""
        direction = kal.render.lighting.sg_direction_from_azimuth_elevation(
            self.azimuth, self.elevation
        )
        return kal.render.lighting.SgLightingParameters(
            amplitude=self.amplitude, 
            sharpness=self.sharpness, 
            direction=direction
        )
    
    def base_render(self, mesh, camera, lighting, clear=False, 
                   active_pass=kal.render.easy_render.RenderPass.render):
        """Base rendering function"""
        render_res = kal.render.easy_render.render_mesh(camera, mesh, lighting=lighting)
        
        img = render_res[active_pass]
        hard_mask = (render_res[kal.render.easy_render.RenderPass.face_idx] >= 0).float().unsqueeze(-1)
        
        if clear:
            img = torch.cat([img, hard_mask], dim=-1)  # Add Alpha channel
        final = (torch.clamp(img * hard_mask, 0., 1.)[0] * 255.).to(torch.uint8)

        return final
    
    def make_lowres_cam(self, in_cam, factor=8):
        """Create a lower resolution camera"""
        lowres_cam = copy.deepcopy(in_cam)
        lowres_cam.width = in_cam.width // factor
        lowres_cam.height = in_cam.height // factor
        return lowres_cam
    
    def render(self, in_cam):
        """Main render function"""
        # Debug: Check tensor dtypes before rendering
        print(f"Camera vertices dtype: {in_cam.extrinsics.t.dtype}")
        print(f"Mesh vertices dtype: {self.mesh.vertices.dtype}")
        print(f"Mesh faces dtype: {self.mesh.faces.dtype}")
        
        active_pass = (kal.render.easy_render.RenderPass.render 
                      if self.apply_lighting 
                      else kal.render.easy_render.RenderPass.albedo)
        return self.base_render(self.mesh, in_cam, self.current_lighting(), active_pass=active_pass)
    
    def make_lowres_render_func(self, render_func):
        """Create a low-resolution render function"""
        def lowres_render_func(in_cam):
            return render_func(self.make_lowres_cam(in_cam))
        return lowres_render_func
    
    def display(self):
        """Display the 3D viewer widget"""
        return self.visualizer
    
    def set_lighting(self, apply_lighting=True, azimuth=0.0, elevation=math.pi/3, 
                    amplitude=3.0, sharpness=5.0):
        """Update lighting parameters"""
        self.apply_lighting = apply_lighting
        self.azimuth = torch.tensor([azimuth], dtype=torch.float32, device='cuda')
        self.elevation = torch.tensor([elevation], dtype=torch.float32, device='cuda')
        self.amplitude = torch.full((1, 3), amplitude, dtype=torch.float32, device='cuda')
        self.sharpness = torch.tensor([sharpness], dtype=torch.float32, device='cuda')
    
    def set_camera_position(self, eye, at=None, up=None):
        """Update camera position"""
        if at is None:
            at = [0., 0., 0.]
        if up is None:
            up = [1., 1., 1.]
            
        self.camera = kal.render.camera.Camera.from_args(
            eye=torch.tensor(eye, dtype=torch.float32, device='cuda'),
            at=torch.tensor(at, dtype=torch.float32, device='cuda'),
            up=torch.tensor(up, dtype=torch.float32, device='cuda'),
            fov=self.camera.fov,
            width=self.camera.width, 
            height=self.camera.height, 
            device='cuda'
        )
        
        # Update visualizer with new camera
        self.visualizer = kal.visualize.IpyTurntableVisualizer(
            self.camera.width, self.camera.height, copy.deepcopy(self.camera), 
            self.render, fast_render=self.make_lowres_render_func(self.render),
            max_fps=24, world_up_axis=1
        )
    
    def _validate_tensors(self):
        """Validate all tensors have correct dtypes and devices"""
        print("=== TENSOR VALIDATION ===")
        
        # Check mesh tensors
        print(f"Mesh vertices: {self.mesh.vertices.dtype} on {self.mesh.vertices.device}")
        print(f"Mesh faces: {self.mesh.faces.dtype} on {self.mesh.faces.device}")
        
        # Check camera tensors
        print(f"Camera eye: {self.camera.extrinsics.t.dtype} on {self.camera.extrinsics.t.device}")
        
        # Check lighting tensors
        print(f"Azimuth: {self.azimuth.dtype} on {self.azimuth.device}")
        print(f"Elevation: {self.elevation.dtype} on {self.elevation.device}")
        print(f"Amplitude: {self.amplitude.dtype} on {self.amplitude.device}")
        print(f"Sharpness: {self.sharpness.dtype} on {self.sharpness.device}")
        
        # Force correct dtypes if needed
        if self.mesh.vertices.dtype != torch.float32:
            print("WARNING: Converting mesh vertices to float32")
            self.mesh.vertices = self.mesh.vertices.float()
            
        if self.mesh.faces.dtype not in [torch.int32, torch.int]:
            print("WARNING: Converting mesh faces to int32")
            self.mesh.faces = self.mesh.faces.int()
            
        print("=== VALIDATION COMPLETE ===\n")