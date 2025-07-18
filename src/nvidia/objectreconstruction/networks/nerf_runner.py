"""
Neural Radiance Fields (NeRF) Implementation and Runner Module.

This module implements a complete Neural Radiance Fields (NeRF) framework for 3D scene
reconstruction from 2D images.
"""

# Standard library imports
import copy
import logging
import os
import random
import tempfile 
import sys
sys.path.append('/customize_cuda/')
sys.path.append('/customize_cuda/build')

import warp
warp.config.kernel_cache_dir = os.path.join("/tmp/warpcache")

# Third-party imports
import cv2
import imageio
import igl
import kaolin
import numpy as np
import open3d as o3d
import PIL.Image as Image
import pyrender
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from pytorch3d.transforms import se3_exp_map
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from tqdm import tqdm

# Local imports
import sys
# Import CUDA common functions (now properly installed in site-packages)
import common
from ..utils.preprocessing import depth2xyzmap, toOpen3dCloud
from .tool import (
    GL_CAM_TO_CV_CAM,
    MeshProcessor,
    PoseUtils,
    set_seed
)

BAD_DEPTH = 99
BAD_COLOR = 128
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["DISPLAY"] = ":0"


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def viewmatrix(z, up, pos):
    """Create view matrix from camera parameters."""
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def get_fixed_camera(mesh):
    """Get fixed camera poses around mesh bounding box."""
    # Get Axis-Aligned Bounding Box corners
    bbox_corners = trimesh.bounds.corners(mesh.bounds)

    # Get Bounding Box Center (AABB)
    center = mesh.centroid
    assert bbox_corners.shape == (8, 3)
    # Sort corners by height (y-coordinate) first
    top_corners = bbox_corners[bbox_corners[:, 1] >= center[1]]
    bottom_corners = bbox_corners[bbox_corners[:, 1] < center[1]]

    # Sort top corners clockwise looking down from above
    top_center = top_corners.mean(axis=0)
    top_angles = np.arctan2(top_corners[:, 2] - top_center[2],
                           top_corners[:, 0] - top_center[0])
    top_corners = top_corners[np.argsort(top_angles)]

    # Sort bottom corners clockwise looking up from below
    bottom_center = bottom_corners.mean(axis=0)
    bottom_angles = np.arctan2(bottom_corners[:, 2] - bottom_center[2],
                              bottom_corners[:, 0] - bottom_center[0])
    bottom_corners = bottom_corners[np.argsort(bottom_angles)]

    # Combine sorted corners
    bbox_corners = np.vstack([top_corners, bottom_corners])
    cameras = []
    for i in range(8):
        cam_center = bbox_corners[i] + 2 * (bbox_corners[i] - center)
        lookat = center - cam_center
        up = np.array([0, 1, 0])
        pos = cam_center
        cameras.append(np.linalg.inv(viewmatrix(lookat, up, pos)))
    return cameras


class ModelRendererOffscreen:
    """Offscreen renderer for 3D models.
    
    Reference: https://colab.research.google.com/drive/1Z71mHIc-Sqval92nK290vAsHZRUkCjUx#scrollTo=fIymQapsuWGn
    If off-screen mode, set os.environ["PYOPENGL_PLATFORM"] at very top of the code
    """

    def __init__(self, model_paths, cam_K, H, W, zfar=2):
        """Initialize offscreen renderer.
        
        Args:
            model_paths: List of model file paths
            cam_K: Camera intrinsic matrix
            H: Image height
            W: Image width
            zfar: Far clipping plane
        """
        self.K = cam_K
        self.scene = pyrender.Scene(
            ambient_light=[1., 1., 1.],
            bg_color=[0, 0, 0]
        )
        self.camera = pyrender.IntrinsicsCamera(
            fx=cam_K[0, 0],
            fy=cam_K[1, 1],
            cx=cam_K[0, 2],
            cy=cam_K[1, 2],
            znear=0.1,
            zfar=zfar
        )
        self.cam_node = self.scene.add(self.camera, pose=np.eye(4))
        self.mesh_nodes = []

        for model_path in model_paths:
            print('model_path', model_path)
            obj_mesh = trimesh.load(model_path)
            mesh = pyrender.Mesh.from_trimesh(obj_mesh)
            # Object pose parent is cam
            mesh_node = self.scene.add(
                mesh, pose=np.eye(4), parent_node=self.cam_node
            )
            self.mesh_nodes.append(mesh_node)

        self.H = H
        self.W = W

        # NOTE: version>0.1.32 not work
        # https://github.com/mmatl/pyrender/issues/85
        self.r = pyrender.OffscreenRenderer(self.W, self.H)
        self.glcam_in_cvcam = GL_CAM_TO_CV_CAM
        self.cvcam_in_glcam = np.linalg.inv(self.glcam_in_cvcam)

    def add_mesh(self, mesh):
        """Add mesh to scene and compute fixed cameras."""
        self.fixe_camera = get_fixed_camera(mesh)
        mesh = pyrender.Mesh.from_trimesh(mesh)

        # Object pose parent is cam
        mesh_node = self.scene.add(
            mesh, pose=np.eye(4), parent_node=self.cam_node
        )
        self.mesh_nodes.append(mesh_node)

    def render(self, ob_in_cvcams):
        """Render scene with object poses in camera coordinates."""
        for i, ob_in_cvcam in enumerate(ob_in_cvcams):
            ob_in_glcam = self.cvcam_in_glcam.dot(ob_in_cvcam)
            self.scene.set_pose(self.mesh_nodes[i], ob_in_glcam)
        color, depth = self.r.render(self.scene)  # depth: float
        return color, depth

    def render_fixed_cameras(self):
        """Render scene from fixed camera positions."""
        colors, depths = [], []
        for i, ob_in_cvcam in enumerate(self.fixe_camera):
            ob_in_glcam = self.cvcam_in_glcam.dot(ob_in_cvcam)
            self.scene.set_pose(self.mesh_nodes[0], ob_in_glcam)
            color, depth = self.r.render(self.scene)  # depth: float
            colors.append(color)
            depths.append(depth)
        return colors, depths

def preprocess_data(rgbs, depths, masks, normal_maps, poses, sc_factor, translation):
    '''
    @rgbs: np array (N,H,W,3)
    @depths: (N,H,W)
    @masks: (N,H,W)
    @normal_maps: (N,H,W,3)
    @poses: (N,4,4)
    '''
    # Replace invalid depth values
    depths[depths < 0.1] = BAD_DEPTH
    

    if masks is not None:
        # Mask out rgbs and depths
        rgbs = np.where(masks[..., None] == 0, BAD_COLOR, rgbs)  # Apply mask without creating a new array
        depths = np.where(masks == 0, BAD_DEPTH, depths)

        if normal_maps is not None:
            # Flip y and z components of normal maps (in-place if dtype allows)
            
            normal_maps = np.where(masks[..., None] == 0, 0, normal_maps)
            

        # Expand masks for consistency (done once)
        masks = masks[..., None]

    # Normalize rgbs in-place if it's float-compatible
    rgbs = rgbs.astype(np.float32, copy=False)
    rgbs /= 255.0  # In-place division

    # Scale depths in-place and expand
    depths *= sc_factor
    depths = depths[..., None]  # Add channel dimension

    if normal_maps is not None:
        normal_maps[..., [1, 2]] *= -1

    # Adjust poses (translation and scaling)
    poses[:, :3, 3] += translation
    poses[:, :3, 3] *= sc_factor

    return rgbs, depths, masks, normal_maps, poses

def build_texture_from_images(mesh, reader, poses, Height, Width, K, sc_factor, far, tex_res,alpha=1.5,beta=0.5,adjust_brightness=True,debug_dir=None):

    frame_ids = torch.arange(len(reader)).long().cuda()
    c2w_array = torch.tensor(poses, dtype=torch.float).cuda()
    tf = c2w_array[frame_ids]
    tf = tf.data.cpu().numpy()
    # from offscreen_renderer import ModelRendererOffscreen

    tex_image = torch.zeros((tex_res,tex_res,3)).cuda().float()
    weight_tex_image = torch.zeros(tex_image.shape[:-1]).cuda().float()

    mesh.merge_vertices()
    mesh.remove_duplicate_faces()
    new_trimesh = mesh.unwrap()
    H, W = tex_image.shape[:2]
    uvs_tex = (new_trimesh.visual.uv*np.array([W-1,H-1]).reshape(1,2))    #(n_V,2)

    H,W = tex_image.shape[:2]
    uvs_tex = (new_trimesh.visual.uv*np.array([W-1,H-1]).reshape(1,2))    #(n_V,2) (63330, 2)

    renderer = ModelRendererOffscreen([], cam_K=K, H=Height, W=Width, zfar=far*sc_factor)
    renderer.add_mesh(new_trimesh)

    print(f"Texture: Initial LOOP FOR storing angles for each face")
    all_tri_list= {key: [] for key in range(new_trimesh.triangles.shape[0])}
    def preprocess(rgb,mask,adjust_brightness=True,alpha=1.5,beta=0.5):
        rgb = np.where(mask[..., None] == 0, BAD_COLOR, rgb)
        mask = mask[..., None]
        rgb = rgb[...,::-1]
        if adjust_brightness:
            rgb = cv2.convertScaleAbs(rgb, alpha=alpha, beta=beta)
        return rgb,mask
    for i in tqdm(range(len(reader))):
        cvcam_in_ob = tf[i]@np.linalg.inv(GL_CAM_TO_CV_CAM)
        color, render_depth = renderer.render([np.linalg.inv(cvcam_in_ob)])
        xyz_map = depth2xyzmap(render_depth, K)
        rgb,mask = preprocess(reader.get_color(i),reader.get_mask(i),adjust_brightness,alpha,beta)
        mask = mask.reshape(Height, Width).astype(bool)
        valid = (render_depth.reshape(Height, Width)>=0.1*sc_factor) & (mask)
        pts = xyz_map[valid].reshape(-1,3)
        pts = transform_pts(pts, cvcam_in_ob)
        ray_colors = rgb[valid].reshape(-1,3)
        # locations, distance, index_tri = trimesh.proximity.closest_point(new_trimesh, pts)
        _, index_tri, locations = igl.signed_distance(pts, new_trimesh.vertices, new_trimesh.faces)
        normals = new_trimesh.face_normals[index_tri]
        for ind_tri, each_tri in enumerate(index_tri):
            rays_o = np.zeros(3)
            pts = transform_pts(rays_o, cvcam_in_ob) #transform to world space
            rays_d = locations[ind_tri]-pts
            rays_d /= np.linalg.norm(rays_d) #unit direction vector
            dot_product = np.dot(-rays_d, normals[ind_tri])
            angle_radians = np.arccos(dot_product)
            angle_degrees = np.degrees(angle_radians)
            all_tri_list[each_tri].extend([[i,angle_degrees]])


    _CHOOSE_TOP_N = 1
    all_triangles_dict={}
    for k,v in all_tri_list.items():
        if(v):
            v.sort(key=lambda x: x[1])
            tep = [i[0] for i in v[:_CHOOSE_TOP_N]]
            all_triangles_dict[k]=set(list(tep))

    all_tri_visited = {key: 0 for key in range(new_trimesh.triangles.shape[0])}
    
    print(f"Texture: Texture map computation ")
    for i in tqdm(range(len(reader))):
        # print(f'project train_images {i}/{len(rgbs_raw)}')

        ############# Raterization
        cvcam_in_ob = tf[i]@np.linalg.inv(GL_CAM_TO_CV_CAM)
        _, render_depth = renderer.render([np.linalg.inv(cvcam_in_ob)])

        xyz_map = depth2xyzmap(render_depth, K)
        rgb,mask = preprocess(reader.get_color(i),reader.get_mask(i),adjust_brightness,alpha,beta)
        mask = mask.reshape(Height,Width).astype(bool)
        valid = (render_depth.reshape(Height,Width)>=0.1*sc_factor) & (mask)
        pts = xyz_map[valid].reshape(-1,3)
        pts = transform_pts(pts, cvcam_in_ob)
        ray_colors = rgb[valid].reshape(-1, 3)
        # locations, distance, index_tri = trimesh.proximity.closest_point(new_trimesh, pts)

        _, index_tri, locations = igl.signed_distance(pts, new_trimesh.vertices, new_trimesh.faces)

        bool_weights = torch.zeros(len(locations)).cuda()
        for jj, trtind__ in enumerate(index_tri):
            if(i in all_triangles_dict[trtind__] ):
                bool_weights[jj]=1
                all_tri_visited[trtind__]=1

        ############## CUDA
        uvs = torch.zeros((len(locations),2)).cuda().float()
        common.rayColorToTextureImageCUDA(torch.from_numpy(new_trimesh.faces).cuda().long(), 
                                            torch.from_numpy(new_trimesh.vertices).cuda().float(), 
                                            torch.from_numpy(locations).cuda().float(), 
                                            torch.from_numpy(index_tri).cuda().long(), 
                                            torch.from_numpy(uvs_tex).cuda().float(), 
                                            uvs)

        uvs = torch.round(uvs).long()
        uvs_flat = uvs[:,1]*(W-1) + uvs[:,0]

        uvs_flat_unique, inverse_ids, cnts = torch.unique(uvs_flat, return_counts=True, return_inverse=True)
        perm = torch.arange(inverse_ids.size(0)).cuda()
        inverse_ids, perm = inverse_ids.flip([0]), perm.flip([0])
        unique_ids = inverse_ids.new_empty(uvs_flat_unique.size(0)).scatter_(0, inverse_ids, perm)
        uvs_unique = torch.stack((uvs_flat_unique%(W-1), uvs_flat_unique//(W-1)), dim=-1).reshape(-1,2)
        cur_weights= bool_weights[unique_ids].cuda().float()
        
        tex_image[uvs_unique[:,1],uvs_unique[:,0]] += torch.from_numpy(ray_colors).cuda().float()[unique_ids]*cur_weights.reshape(-1,1)
        weight_tex_image[uvs_unique[:,1], uvs_unique[:,0]] += cur_weights

    tex_image = tex_image/weight_tex_image[...,None]
    tex_image = tex_image.data.cpu().numpy()
    # Handle non-finite values (NaN or inf) before clipping and conversion
    tex_image = np.nan_to_num(tex_image, nan=0.0, posinf=255.0, neginf=0.0)
    tex_image = np.clip(tex_image,0,255).astype(np.uint8)
    tex_image = tex_image[::-1].copy()      # UV origin is bottom-left

    new_texture = texture_map_interpolation(tex_image)

    new_trimesh.visual = trimesh.visual.texture.TextureVisuals(uv=new_trimesh.visual.uv, image=Image.fromarray(new_texture))
    return new_trimesh



def trimesh_split(mesh, min_edge=1000):
    '''!NOTE mesh.split takes too much memory for large mesh. That's why we have this function
    '''
    components = trimesh.graph.connected_components(mesh.edges, min_len=min_edge, nodes=None, engine=None)
    meshes = []
    for i,c in enumerate(components):
        mask = np.zeros(len(mesh.vertices),dtype=bool)
        mask[c] = 1
        cur_mesh = mesh.copy()
        cur_mesh.update_vertices(mask=mask.astype(bool))
        meshes.append(cur_mesh)
    return meshes


def mesh_to_real_world(mesh, pose_offset, translation, sc_factor):
    """
    Transform mesh to real world coordinates.
    This is a wrapper around MeshProcessor.mesh_to_real_world for backward compatibility.
    """
    return MeshProcessor.mesh_to_real_world(mesh, pose_offset, translation, sc_factor)


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def to_homo(pts):
    '''
    @pts: (N,3 or 2) will homogeneliaze the last dimension
    '''
    assert len(pts.shape)==2, f'pts.shape: {pts.shape}'
    homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
    return homo


def transform_pts(pts,tf):
    """Transform 2d or 3d points
    @pts: (...,3)
    """
    return (tf[...,:-1,:-1]@pts[...,None] + tf[...,:-1,-1:])[...,0]


def ray_box_intersection_batch(origins, dirs, bounds):
    '''
    @origins: (N,3) origin and directions. In the same coordinate frame as the bounding box
    @bounds: (2,3) xyz_min and max
    '''
    if not torch.is_tensor(origins):
        origins = torch.tensor(origins)
        dirs = torch.tensor(dirs)
    if not torch.is_tensor(bounds):
        bounds = torch.tensor(bounds)

    dirs = dirs/(torch.norm(dirs,dim=-1,keepdim=True)+1e-10)
    inv_dirs = 1/dirs
    bounds = bounds[None].expand(len(dirs),-1,-1)   #(N,2,3)

    sign = torch.zeros((len(dirs),3)).long().to(dirs.device)  #(N,3)
    sign[:,0] = (inv_dirs[:,0] < 0)
    sign[:,1] = (inv_dirs[:,1] < 0)
    sign[:,2] = (inv_dirs[:,2] < 0)

    tmin = (torch.gather(bounds[...,0],dim=1,index=sign[:,0].reshape(-1,1)).reshape(-1) - origins[:,0]) * inv_dirs[:,0]   #(N)
    tmin[tmin<0] = 0
    tmax = (torch.gather(bounds[...,0],dim=1,index=1-sign[:,0].reshape(-1,1)).reshape(-1) - origins[:,0]) * inv_dirs[:,0]
    tymin = (torch.gather(bounds[...,1],dim=1,index=sign[:,1].reshape(-1,1)).reshape(-1) - origins[:,1]) * inv_dirs[:,1]
    tymin[tymin<0] = 0
    tymax = (torch.gather(bounds[...,1],dim=1,index=1-sign[:,1].reshape(-1,1)).reshape(-1) - origins[:,1]) * inv_dirs[:,1]

    ishit = torch.ones(len(dirs)).bool().to(dirs.device)
    ishit[(tmin > tymax) | (tymin > tmax)] = 0
    tmin[tymin>tmin] = tymin[tymin>tmin]
    tmax[tymax<tmax] = tymax[tymax<tmax]

    tzmin = (torch.gather(bounds[...,2],dim=1,index=sign[:,2].reshape(-1,1)).reshape(-1) - origins[:,2]) * inv_dirs[:,2]
    tzmin[tzmin<0] = 0
    tzmax = (torch.gather(bounds[...,2],dim=1,index=1-sign[:,2].reshape(-1,1)).reshape(-1) - origins[:,2]) * inv_dirs[:,2]

    ishit[(tmin > tzmax) | (tzmin > tmax)] = 0
    tmin[tzmin>tmin] = tzmin[tzmin>tmin]   #(N)
    tmax[tzmax<tmax] = tzmax[tzmax<tmax]

    tmin[ishit==0] = -1
    tmax[ishit==0] = -1

    return tmin, tmax


def get_optimized_poses_in_real_world(poses_normalized, pose_array, sc_factor, translation):
    """
    Get optimized poses in real world coordinates.
    This is a wrapper around PoseUtils.get_optimized_poses_in_real_world for backward compatibility.
    """
    return PoseUtils.get_optimized_poses_in_real_world(poses_normalized, pose_array, sc_factor, translation)


def compute_near_far_and_filter_rays(cam_in_world,rays,cfg):
    '''
    @cam_in_world: in normalized space
    @rays: (...,D) in camera
    Return:
        (-1,D+2) with near far
    '''
    D = rays.shape[-1]
    rays = rays.reshape(-1,D)
    dirs_unit = rays[:,:3]/np.linalg.norm(rays[:,:3],axis=-1).reshape(-1,1)
    dirs = (cam_in_world[:3,:3]@rays[:,:3].T).T
    origins = (cam_in_world@to_homo(np.zeros(dirs.shape)).T).T[:,:3]
    bounds = np.array(cfg['bounding_box']).reshape(2,3)
    tmin,tmax = ray_box_intersection_batch(origins,dirs,bounds)
    tmin = tmin.data.cpu().numpy()
    tmax = tmax.data.cpu().numpy()
    ishit = tmin>=0
    near = (dirs_unit*tmin.reshape(-1,1))[:,2]
    far = (dirs_unit*tmax.reshape(-1,1))[:,2]
    good_rays = rays[ishit]
    near = near[ishit]
    far = far[ishit]
    near = np.abs(near)
    far = np.abs(far)
    good_rays = np.concatenate((good_rays,near.reshape(-1,1),far.reshape(-1,1)), axis=-1)  #(N,8+2)

    return good_rays


def get_masks(z_vals, target_d, truncation, cfg, dir_norm=None):
    valid_depth_mask = (target_d>=cfg['near']*cfg['sc_factor']) & (target_d<=cfg['far']*cfg['sc_factor'])
    front_mask = (z_vals < target_d - truncation)
    back_mask = (z_vals > target_d + truncation*cfg['neg_trunc_ratio'])

    sdf_mask = (1.0 - front_mask.float()) * (1.0 - back_mask.float()) * valid_depth_mask

    num_fs_samples = front_mask.sum()
    num_sdf_samples = sdf_mask.sum()
    num_samples = num_sdf_samples + num_fs_samples
    fs_weight = 0.5
    sdf_weight = 1.0 - fs_weight
    return front_mask.bool(), sdf_mask.bool(), fs_weight, sdf_weight


def get_sdf_loss(z_vals, target_d, predicted_sdf, truncation, cfg, return_mask=False, sample_weights=None, rays_d=None):
    dir_norm = rays_d.norm(dim=-1,keepdim=True)
    front_mask, sdf_mask, fs_weight, sdf_weight = get_masks(z_vals, target_d, truncation, cfg, dir_norm=dir_norm)
    front_mask = front_mask.bool()

    mask = (target_d>cfg['far']*cfg['sc_factor']) & (predicted_sdf<cfg['fs_sdf'])

    fs_loss = torch.mean(((predicted_sdf-cfg['fs_sdf']) * mask)**2 * sample_weights) * fs_weight

    mask = front_mask & (target_d<=cfg['far']*cfg['sc_factor']) & (predicted_sdf<1)
    empty_loss = torch.mean(torch.abs(predicted_sdf-1) * mask * sample_weights) * cfg['empty_weight']
    fs_loss += empty_loss

    sdf_loss = torch.mean(((z_vals + predicted_sdf * truncation) * sdf_mask - target_d * sdf_mask)**2 * sample_weights) * sdf_weight

    if return_mask:
        return fs_loss,sdf_loss,front_mask,sdf_mask
    return fs_loss, sdf_loss


@torch.no_grad()
def sample_rays_uniform(N_samples,near,far,lindisp=False,perturb=True):
    '''
    @near: (N_ray,1)
    '''
    N_ray = near.shape[0]
    t_vals = torch.linspace(0., 1., steps=N_samples, device=near.device).reshape(1,-1)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))   #(N_ray,N_sample)

    if perturb > 0.:
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=far.device)
        z_vals = lower + (upper - lower) * t_rand
        z_vals = torch.clip(z_vals,near,far)

    return z_vals.reshape(N_ray,N_samples)


class SHEncoder(nn.Module):
    '''Spherical encoding
    '''
    def __init__(self, input_dim=3, degree=4):

        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.out_dim = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                #result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result


class FeatureArray(nn.Module):
    """
    Per-frame corrective latent code.
    """

    def __init__(self, num_frames, num_channels):
        super().__init__()

        self.num_frames = num_frames
        self.num_channels = num_channels

        self.data = nn.parameter.Parameter(torch.normal(0,1,size=[num_frames, num_channels]).float(), requires_grad=True)
        self.register_parameter('data',self.data)

    def __call__(self, ids):
        return self.data[ids]


class PoseArray(nn.Module):
    """
    Per-frame camera pose correction in the normalized space.

    The pose correction contains 6 parameters for each pose (3 for rotation, 3 for translation).
    The rotation parameters define axis-angles which can be converted into a rotation matrix.
    """
    def __init__(self, num_frames,max_trans,max_rot):
        super().__init__()
        self.num_frames = num_frames
        self.max_trans = max_trans
        self.max_rot = max_rot
        self.data = nn.parameter.Parameter(torch.zeros([num_frames, 6]).float(), requires_grad=True)
        self.register_parameter('data',self.data)

    def get_matrices(self,ids):
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids).long()

        theta = torch.tanh(self.data)
        trans = theta[:,:3] * self.max_trans
        rot = theta[:,3:6] * self.max_rot/180.0*np.pi
        Ts_data = se3_exp_map(torch.cat((trans,rot),dim=-1)).permute(0,2,1)
        Ts = torch.eye(4, device=self.data.device).reshape(1,4,4).repeat(len(ids),1,1)
        mask = ids!=0
        Ts[mask] = Ts_data[ids[mask]]
        return Ts


class GlobalPoseArray(nn.Module):
    """
    Per-frame camera pose correction in the normalized space.

    The pose correction contains 6 parameters for each pose (3 for rotation, 3 for translation).
    The rotation parameters define axis-angles which can be converted into a rotation matrix.
    """
    def __init__(self, num_frames,max_trans,max_rot):
        super().__init__()
        self.num_frames = num_frames
        self.max_trans = max_trans
        self.max_rot = max_rot
        self.data = nn.parameter.Parameter(torch.zeros([num_frames, 6]).float(), requires_grad=False)
        self.register_parameter('data',self.data)

    def get_matrices(self,ids):
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids).long()

        theta = torch.tanh(self.data)
        trans = theta[:,:3] * self.max_trans
        rot = theta[:,3:6] * self.max_rot/180.0*np.pi
        Ts_data = se3_exp_map(torch.cat((trans,rot),dim=-1)).permute(0,2,1)
        Ts = torch.eye(4, device=self.data.device).reshape(1,4,4).repeat(len(ids),1,1)
        mask = ids!=0
        Ts[mask] = Ts_data[ids[mask]]
        return Ts


class DataLoader:
    def __init__(self,rays,batch_size):
        self.rays = rays
        self.batch_size = batch_size
        self.pos = 0
        self.ids = torch.randperm(len(self.rays)).cuda()

    def __next__(self):
        if self.pos + self.batch_size < len(self.ids):
            self.batch_ray_ids = self.ids[self.pos:self.pos+self.batch_size].cuda()
            out = self.rays[self.batch_ray_ids]
            self.pos += self.batch_size
            return out.cuda()
        elif self.pos < len(self.ids):
            self.batch_ray_ids = self.ids[self.pos:]
            out = self.rays[self.batch_ray_ids]
            self.pos = len(self.ids)
            return out.cuda()
        self.ids = torch.randperm(len(self.rays))
        self.pos = self.batch_size
        self.batch_ray_ids = self.ids[:self.batch_size].cuda()
        return self.rays[self.batch_ray_ids].cuda()
class ImageLoader:
    def __init__(self,dataset,batch_size,sc_factor=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos = 0
        self.ids = torch.randperm(len(self.dataset))
        self.sc_factor = sc_factor
    def __len__(self):
        return len(self.dataset)
    def get_items(self,indices):
        colors = np.array([self.dataset.get_color(i) for i in indices])
        depths = np.array([self.dataset.get_depth(i) for i in indices])
        masks = np.array([self.dataset.get_mask(i) for i in indices])
        depths[depths<0.1] = BAD_DEPTH
        rgbs = np.where(masks[..., None] == 0, BAD_COLOR, colors)  # Apply mask without creating a new array
        depths = np.where(masks == 0, BAD_DEPTH, depths)
        masks = masks[..., None]

        rgbs = rgbs.astype(np.float32, copy=False)
        rgbs /= 255.0
        rgbs = rgbs[..., ::-1]

        # Scale depths in-place and expand
        depths *= self.sc_factor
        depths = depths[..., None]
        return rgbs, depths, masks
    def __next__(self):
        if self.batch_size > len(self.ids):
            self.ids = torch.cat([self.ids, torch.randperm(len(self.dataset))])
        self.batch_ids = self.ids[:self.batch_size]
        ids = self.batch_ids
        self.ids = self.ids[self.batch_size:]
        return ids.cpu().numpy(),*self.get_items(self.batch_ids)

class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    

def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


def get_camera_rays_np(H, W, K):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0,2])/K[0,0], -(j - K[1,2])/K[1,1], -np.ones_like(i)], axis=-1)
    return dirs


def get_embedder(multires, cfg, i=0, octree_m=None):
    if i == -1:
        return nn.Identity(), 3
    elif i==0:
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }

        embed = Embedder(**embed_kwargs)
        out_dim = embed.out_dim
    elif i==1:
        from torch_ngp_grid_encoder.grid import GridEncoder
        embed = GridEncoder(input_dim=3, n_levels=cfg['num_levels'], log2_hashmap_size=cfg['log2_hashmap_size'], desired_resolution=cfg['finest_res'], base_resolution=cfg['base_res'], level_dim=cfg['feature_grid_dim'])
        out_dim = embed.out_dim
    elif i==2:
        embed = SHEncoder(degree=cfg['multires_views'])
        out_dim = embed.out_dim
    return embed, out_dim


def texture_map_interpolation(tex_image_numpy):
    all_channels = []
    # mask = (tex_image_numpy[:, :, 0] == 0) & (tex_image_numpy[:, :, 1] == 0) & (tex_image_numpy[:, :, 2] == 0)
    mask = np.all(tex_image_numpy == 0, axis=2)
    x = np.arange(0, tex_image_numpy.shape[1])
    y = np.arange(0, tex_image_numpy.shape[0])
    xx, yy = np.meshgrid(x, y)
    for each_channel in range(tex_image_numpy.shape[2]):
        curr_channel = tex_image_numpy[:,:,each_channel]
        x1 = xx[~mask]
        y1 = yy[~mask]
        newarr = curr_channel[~mask]

        GD1 = griddata((x1, y1), newarr.ravel(),
                                    (xx, yy),
                                method='nearest')
        # arr=(GD1[:,:,np.newaxis]/GD1.max())*255
        all_channels.append(GD1[:,:,np.newaxis].round().astype(np.uint8))
    final_image = np.concatenate(all_channels, axis =-1)
    return final_image


class OctreeManager:
    def __init__(self,pts=None,max_level=None,octree=None):
        if octree is None:
            pts_quantized = kaolin.ops.spc.quantize_points(pts.contiguous(), level=max_level)
            self.octree = kaolin.ops.spc.unbatched_points_to_octree(pts_quantized, max_level, sorted=False)
        else:
            self.octree = octree
        lengths = torch.tensor([len(self.octree)], dtype=torch.int32).cpu()
        self.max_level, self.pyramids, self.exsum = kaolin.ops.spc.scan_octrees(self.octree,lengths)
        self.n_level = self.max_level+1
        self.point_hierarchies = kaolin.ops.spc.generate_points(self.octree, self.pyramids, self.exsum)
        self.point_hierarchy_dual, self.pyramid_dual = kaolin.ops.spc.unbatched_make_dual(self.point_hierarchies, self.pyramids[0])
        self.trinkets, self.pointers_to_parent = kaolin.ops.spc.unbatched_make_trinkets(self.point_hierarchies, self.pyramids[0], self.point_hierarchy_dual, self.pyramid_dual)
        self.n_vox = len(self.point_hierarchies)
        self.n_corners = len(self.point_hierarchy_dual)

    def get_level_corner_quantized_points(self,level):
        start = self.pyramid_dual[...,1,level]
        num = self.pyramid_dual[...,0,level]
        return self.point_hierarchy_dual[start:start+num]

    def get_level_quantized_points(self,level):
        start = self.pyramids[...,1,level]
        num = self.pyramids[...,0,level]
        return self.pyramids[start:start+num]

    def get_trilinear_coeffs(self,x,level):
        quantized = kaolin.ops.spc.quantize_points(x, level)
        coeffs = kaolin.ops.spc.coords_to_trilinear_coeffs(x,quantized,level)   #(N,8)
        return coeffs

    def get_center_ids(self,x,level):
        pidx = kaolin.ops.spc.unbatched_query(self.octree, self.exsum, x, level, with_parents=False)
        return pidx


    def get_corners_ids(self,x,level):
        pidx = kaolin.ops.spc.unbatched_query(self.octree, self.exsum, x, level, with_parents=False)
        corner_ids = self.trinkets[pidx]
        is_valid = torch.ones(len(x)).bool().to(x.device)
        bad_ids = (pidx<0).nonzero()[:,0]
        is_valid[bad_ids] = 0

        return corner_ids, is_valid

    def trilinear_interpolate(self,x,level,feat):
        '''
        @feat: (N_feature of current level, D)
        '''
        ############!NOTE direct API call cannot back prop well
        # pidx = kaolin.ops.spc.unbatched_query(self.octree, self.exsum, x, level, with_parents=False)
        # x = x.unsqueeze(0)
        # interpolated = kaolin.ops.spc.unbatched_interpolate_trilinear(coords=x,pidx=pidx.int(),point_hierarchy=self.point_hierarchies,trinkets=self.trinkets, feats=feat, level=level)[0]
        ##################

        coeffs = self.get_trilinear_coeffs(x,level)  #(N,8)
        corner_ids,is_valid = self.get_corners_ids(x,level)
        # if corner_ids.max()>=feat.shape[0]:
        #     pdb.set_trace()

        corner_feat = feat[corner_ids[is_valid].long()]   #(N,8,D)
        out = torch.zeros((len(x),feat.shape[-1]),device=x.device).float()
        out[is_valid] = torch.sum(coeffs[...,None][is_valid]*corner_feat, dim=1)   #(N,D)

        # corner_feat = feat[corner_ids.long()]   #(N,8,D)
        # out = torch.sum(coeffs[...,None]*corner_feat, dim=1)   #(N,D)

        return out,is_valid

    def draw_boxes(self,level,outfile):
        centers = kaolin.ops.spc.unbatched_get_level_points(self.point_hierarchies.reshape(-1,3), self.pyramids.reshape(2,-1), level)
        pts = (centers.float()+0.5)/(2**level)*2-1   #Normalize to [-1,1]
        pcd = toOpen3dCloud(pts.data.cpu().numpy())
        o3d.io.write_point_cloud(outfile.replace("corners","centers"),pcd)

        corners = kaolin.ops.spc.unbatched_get_level_points(self.point_hierarchy_dual, self.pyramid_dual, level)
        pts = corners.float()/(2**level)*2-1   #Normalize to [-1,1]
        pcd = toOpen3dCloud(pts.data.cpu().numpy())
        o3d.io.write_point_cloud(outfile,pcd)


    def ray_trace(self,rays_o,rays_d,level,debug=False):
        """Octree is in normalized [-1,1] world coordinate frame
        'rays_o': ray origin in normalized world coordinate system
        'rays_d': (N,3) unit length ray direction in normalized world coordinate system
        'octree': spc
        @voxel_size: in the scale of [-1,1] space
        Return:
            ray_depths_in_out: traveling times, NOT the Z value
        """
        

        # Avoid corner cases. issuse in kaolin: https://github.com/NVIDIAGameWorks/kaolin/issues/490 and https://github.com/NVIDIAGameWorks/kaolin/pull/634
        # rays_o = rays_o.clone() + 1e-7

        # Convert input tensors to appropriate types
        rays_o = rays_o.to(torch.float32)
        rays_d = rays_d.to(torch.float32)
        
        # Kaolin expects specific types for each parameter
        # octree should be a Byte type (uint8)
        octree = self.octree.to(torch.uint8)
        
        # point_hierarchies should be short/int16
        point_hierarchies = self.point_hierarchies
        if point_hierarchies.dtype != torch.int16 and point_hierarchies.dtype != torch.short:
            point_hierarchies = point_hierarchies.to(torch.int16)
            
        # pyramids likely needs to be integer type as well
        pyramids_0 = self.pyramids[0].to(torch.int32)
        
        # exsum should be int32 instead of float32
        exsum = self.exsum.to(torch.int32)

        ray_index, rays_pid, depth_in_out = kaolin.render.spc.unbatched_raytrace(octree, point_hierarchies, pyramids_0, exsum, rays_o, rays_d, level=level, return_depth=True, with_exit=True)
        if ray_index.size()[0] == 0:
            print("[WARNING] batch has 0 intersections!!")
            ray_depths_in_out = torch.zeros((rays_o.shape[0],1,2))
            rays_pid = -torch.ones_like(rays_o[:, :1])
            rays_near = torch.zeros_like(rays_o[:, :1])
            rays_far = torch.zeros_like(rays_o[:, :1])
            return rays_near, rays_far, rays_pid, ray_depths_in_out

        intersected_ray_ids,counts = torch.unique_consecutive(ray_index,return_counts=True)
        max_intersections = counts.max().item()
        start_poss = torch.cat([torch.tensor([0], device=counts.device),torch.cumsum(counts[:-1],dim=0)],dim=0)

        # Ensure depth_in_out is float32 for CUDA kernel
        depth_in_out = depth_in_out.to(torch.float32)
        ray_depths_in_out = common.postprocessOctreeRayTracing(ray_index.long().contiguous(),depth_in_out.contiguous(),intersected_ray_ids.long().contiguous(),start_poss.long().contiguous(), max_intersections, rays_o.shape[0])

        rays_far = ray_depths_in_out[:,:,1].max(dim=-1)[0].reshape(-1,1)
        rays_near = ray_depths_in_out[:,0,0].reshape(-1,1)

        return rays_near, rays_far, rays_pid, ray_depths_in_out


class NeRFSmall(nn.Module):
    def __init__(self,num_layers=3,hidden_dim=64,geo_feat_dim=15,num_layers_color=4,hidden_dim_color=64,input_ch=3, input_ch_views=3):
        super(NeRFSmall, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=True))
            if l!=num_layers-1:
                sigma_net.append(nn.ReLU(inplace=True))

        self.sigma_net = nn.Sequential(*sigma_net)
        torch.nn.init.constant_(self.sigma_net[-1].bias, 0.1)     # Encourage last layer predict positive SDF

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim

            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim

            color_net.append(nn.Linear(in_dim, out_dim, bias=True))
            if l!=num_layers_color-1:
                color_net.append(nn.ReLU(inplace=True))

        self.color_net = nn.Sequential(*color_net)

    def forward_sdf(self,x):
        '''
        @x: embedded positions
        '''
        h = self.sigma_net(x)
        sigma, geo_feat = h[..., 0], h[..., 1:]
        return sigma


    def forward(self, x):
        x = x.float()
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        # sigma
        h = input_pts
        h = self.sigma_net(h)

        sigma, geo_feat = h[..., 0], h[..., 1:]

        # color
        h = torch.cat([input_views, geo_feat], dim=-1)
        color = self.color_net(h)

        outputs = torch.cat([color, sigma.unsqueeze(dim=-1)], -1)

        return outputs

class NerfRunner:
    def __init__(self,cfg,reader,poses,K,normal_maps=None,_run=None,occ_masks=None,build_octree_pcd=None):
        set_seed(0)
        self.cfg = cfg
        self.cfg['save_dir'] = os.path.expandvars(self.cfg['save_dir'])
        self.cfg['tv_loss_weight'] = eval(str(self.cfg['tv_loss_weight']))
        self._run = _run
        self.reader = reader
        self.image_loader = ImageLoader(reader, batch_size=cfg['batch_size'],sc_factor=cfg['sc_factor'])
        self.all_poses = poses
        self.all_occ_masks = occ_masks
        self.K = K.copy()
        self.mesh = None
        self.train_pose = False
        self.N_iters = self.cfg['n_step']+1
        self.build_octree_pts = np.asarray(build_octree_pcd.points).copy()   # Make it pickable
        self.all_normal_maps = normal_maps

        down_scale_ratio = cfg['down_scale_ratio']
        self.down_scale_ratio = down_scale_ratio
        self.down_scale = np.ones((2),dtype=np.float32)
        
        if down_scale_ratio!=1:
            H,W = reader.H, reader.W
            self.H, self.W = H,W
            self.cfg['dilate_mask_size'] = int(self.cfg['dilate_mask_size']//down_scale_ratio)

            self.K[0] *= float(self.W)/W
            self.K[1] *= float(self.H)/H
            self.down_scale = np.array([float(self.W)/W, float(self.H)/H])

        self.H, self.W = reader.H, reader.W

        self.octree_m = None
        if self.cfg['use_octree']:
            self.build_octree()

        self.create_nerf()
        self.create_optimizer()

        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=self.cfg['amp'])

        self.global_step = 0

        print("sc_factor",self.cfg['sc_factor'])
        print("translation",self.cfg['translation'])

        self.c2w_array = torch.tensor(poses).float().cuda()

        self.best_models = None
        self.best_loss = np.inf

        
    def get_ray_loader(self,frame_ids,images,depths,masks):
        if self.down_scale_ratio!=1:
            H,W = images[0].shape[:2]

            ############## No interpolatio nto keep consistency
            down_scale_ratio = int(down_scale_ratio)
            self.images = images[:, ::down_scale_ratio, ::down_scale_ratio]
            self.depths = depths[:, ::down_scale_ratio, ::down_scale_ratio]
            self.masks = masks[:, ::down_scale_ratio, ::down_scale_ratio]
            if self.all_normal_maps is not None:
                self.normal_maps = self.all_normal_maps[frame_ids, ::down_scale_ratio, ::down_scale_ratio]
            else:
                self.normal_maps = None
            if self.all_occ_masks is not None:
                self.occ_masks = self.all_occ_masks[frame_ids, ::down_scale_ratio, ::down_scale_ratio]
            else:
                self.occ_masks = None
            self.poses = self.all_poses[frame_ids,...]
            self.frame_ids = frame_ids
        else:
            self.images = images
            self.H, self.W = images[0].shape[:2]
            self.depths = depths
            self.masks = masks
            if self.all_normal_maps is not None:
                self.normal_maps = self.all_normal_maps[frame_ids,...]
            else:
                self.normal_maps = None
            if self.all_occ_masks is not None:
                self.occ_masks = self.all_occ_masks[frame_ids,...]
            else:
                self.occ_masks = None
            self.poses = self.all_poses[frame_ids,...]
            self.frame_ids = frame_ids
        use_batching = not self.cfg['no_batching']
        if use_batching:
            if self.cfg['use_mask']:
                print("Using mask")
                rays_ = []
                for i_mask in range(len(self.masks)):
                    rays = self.make_frame_rays(i_mask)
                    rays_.append(rays)
                rays = np.concatenate(rays_, axis=0)

            else:
                rays = np.stack([get_camera_rays_np(self.H, self.W, self.K) for _ in range(self.poses.shape[0])], 0)  # [N, self.H, self.W, 3]
                rays = np.concatenate([rays, self.images], -1)  # [N, H, W, 6]
                rays = np.concatenate([rays, self.depths], -1)  # [N, H, W, 7]
                rays = np.concatenate([rays, self.masks>0], -1)  # [N, H, W, 8]
                if self.normal_maps is not None:
                    rays = np.concatenate([rays, self.normal_maps[i_mask]], -1)  # [H, W, 11]
                ##########!TODO add normal, index changed
                pdb.set_trace()
                ids = frame_ids[:, np.newaxis, np.newaxis, np.newaxis]
                ids = ids[:, np.newaxis, np.newaxis, np.newaxis]
                ids = np.tile(ids, [1, rays.shape[1], rays.shape[2], 1])
                rays = np.concatenate([rays, ids], -1)  # [N, H, W, 9]
                print('rays',rays.shape)
                rays_ = []
                D = rays.shape[-1]
                for i_pose in range(len(self.poses)):
                    cur_rays = rays[i_pose].reshape(-1,D)
                    cur_rays = compute_near_far_and_filter_rays(self.poses[i_pose],cur_rays,self.cfg)
                    rays_.append(cur_rays)
                rays = np.concatenate(rays_,axis=0)

        if self.cfg['denoise_depth_use_octree_cloud']:
            logging.info("denoise cloud")
            mask = (rays[:,self.ray_mask_slice]>0) & (rays[:,self.ray_depth_slice]<=self.cfg['far']*self.cfg['sc_factor'])
            rays_dir = rays[mask][:,self.ray_dir_slice]
            rays_depth = rays[mask][:,self.ray_depth_slice]
            pts3d = rays_dir*rays_depth.reshape(-1,1)
            frame_ids = rays[mask][:,self.ray_frame_id_slice].astype(int)
            pts3d_w = (self.all_poses[frame_ids]@to_homo(pts3d)[...,None])[:,:3,0]
            logging.info(f"Denoising rays based on octree cloud")

            kdtree = cKDTree(self.build_octree_pts)
            dists,indices = kdtree.query(pts3d_w,k=1,workers=-1)
            bad_mask = dists>0.02*self.cfg['sc_factor']
            bad_ids = np.arange(len(rays))[mask][bad_mask]
            rays[bad_ids,self.ray_depth_slice] = BAD_DEPTH*self.cfg['sc_factor']
            rays[bad_ids, self.ray_type_slice] = 1
            rays = rays[rays[:,self.ray_type_slice]==0]
            logging.info(f"bad_mask#={bad_mask.sum()}")

        if use_batching:
            rays = torch.tensor(rays, dtype=torch.float).cuda()

        self.rays = rays
        print("rays", rays.shape, self.rays.device)
        
        self.data_loader = DataLoader(rays=self.rays, batch_size=self.cfg['N_rand'])
        #delete images
        self.images = None
        self.depths = None
        self.masks = None
        self.normal_maps = None
        self.occ_masks = None

    def create_nerf(self,device=torch.device("cuda")):
        """Instantiate NeRF's MLP model.
        """
        models = {}
        embed_fn, input_ch = get_embedder(self.cfg['multires'], self.cfg, i=self.cfg['i_embed'], octree_m=self.octree_m)
        embed_fn = embed_fn.to(device)
        models['embed_fn'] = embed_fn

        input_ch_views = 0
        embeddirs_fn = None
        if self.cfg['use_viewdirs']:
            embeddirs_fn, input_ch_views = get_embedder(self.cfg['multires_views'], self.cfg, i=self.cfg['i_embed_views'], octree_m=self.octree_m)
        models['embeddirs_fn'] = embeddirs_fn

        output_ch = 4
        skips = [4]

        model = NeRFSmall(num_layers=2,hidden_dim=64,geo_feat_dim=15,num_layers_color=3,hidden_dim_color=64,input_ch=input_ch, input_ch_views=input_ch_views+self.cfg['frame_features']).to(device)
        model = model.to(device)
        models['model'] = model

        model_fine = None
        if self.cfg['N_importance'] > 0:
            if not self.cfg['share_coarse_fine']:
                model_fine = NeRFSmall(num_layers=2,hidden_dim=64,geo_feat_dim=15,num_layers_color=3,hidden_dim_color=64,input_ch=input_ch, input_ch_views=input_ch_views).to(device)
        models['model_fine'] = model_fine

        # Create feature array
        num_training_frames = len(self.reader)
        feature_array = None
        if self.cfg['frame_features'] > 0:
            feature_array = FeatureArray(num_training_frames, self.cfg['frame_features']).to(device)
        models['feature_array'] = feature_array
        # Create pose array
        pose_array = None
        if not self.cfg['optimize_poses']:
            logging.info("Not optimizing poses")
            pose_array = GlobalPoseArray(num_training_frames,max_trans=self.cfg['max_trans']*self.cfg['sc_factor'],max_rot=self.cfg['max_rot']).to(device)
        else:
            logging.info("Optimizing poses")
            pose_array = PoseArray(num_training_frames,max_trans=self.cfg['max_trans']*self.cfg['sc_factor'],max_rot=self.cfg['max_rot']).to(device)
        models['pose_array'] = pose_array
        self.models = models



    def make_frame_rays(self,frame_id):
        mask = self.masks[frame_id,...,0].copy()
        rays = get_camera_rays_np(self.H, self.W, self.K)   # [self.H, self.W, 3]  We create rays frame-by-frame to save memory
        rays = np.concatenate([rays, self.images[frame_id]], -1)  # [H, W, 6]
        rays = np.concatenate([rays, self.depths[frame_id]], -1)  # [H, W, 7]
        rays = np.concatenate([rays, self.masks[frame_id]>0], -1)  # [H, W, 8]
        if self.normal_maps is not None:
            rays = np.concatenate([rays, self.normal_maps[frame_id]], -1)  # [H, W, 11]
        rays = np.concatenate([rays, self.frame_ids[frame_id]*np.ones(self.depths[frame_id].shape)], -1)  # [H, W, 12]
        ray_types = np.zeros((self.H,self.W,1))    # 0 is good; 1 is invalid depth (uncertain)
        invalid_depth = ((self.depths[frame_id,...,0]<self.cfg['near']*self.cfg['sc_factor']) | (self.depths[frame_id,...,0]>self.cfg['far']*self.cfg['sc_factor'])) & (mask>0)
        ray_types[invalid_depth] = 1
        rays = np.concatenate((rays,ray_types), axis=-1)
        self.ray_dir_slice = [0,1,2]
        self.ray_rgb_slice = [3,4,5]
        self.ray_depth_slice = 6
        self.ray_mask_slice = 7
        if self.normal_maps is not None:
            self.ray_normal_slice = [8,9,10]
            self.ray_frame_id_slice = 11
            self.ray_type_slice = 12
        else:
            self.ray_frame_id_slice = 8
            self.ray_type_slice = 9

        n = rays.shape[-1]

        ########## Option2: dilate
        down_scale_ratio = int(self.cfg['down_scale_ratio'])
        if self.frame_ids[frame_id]==0:   #!NOTE first frame ob mask is assumed perfect
            kernel = np.ones((100, 100), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            if self.occ_masks is not None:
                mask[self.occ_masks[frame_id]>0] = 0
        else:
            dilate = 60//down_scale_ratio
            # dilate = 100
            kernel = np.ones((dilate, dilate), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            if self.occ_masks is not None:
                mask[self.occ_masks[frame_id]>0] = 0


        if self.cfg['rays_valid_depth_only']:
            mask[invalid_depth] = 0

        vs,us = np.where(mask>0)
        cur_rays = rays[vs,us].reshape(-1,n)
        cur_rays = cur_rays[cur_rays[:,self.ray_type_slice]==0]
        cur_rays = compute_near_far_and_filter_rays(self.poses[frame_id],cur_rays,self.cfg)
        if self.normal_maps is not None:
            self.ray_near_slice = 13
            self.ray_far_slice = 14
        else:
            self.ray_near_slice = 10
            self.ray_far_slice = 11

        if self.cfg['use_octree']:
            rays_o_world = (self.poses[frame_id]@to_homo(np.zeros((len(cur_rays),3))).T).T[:,:3]
            rays_o_world = torch.from_numpy(rays_o_world).cuda().float()
            rays_unit_d_cam = cur_rays[:,:3]/np.linalg.norm(cur_rays[:,:3],axis=-1).reshape(-1,1)
            rays_d_world = (self.poses[frame_id][:3,:3]@rays_unit_d_cam.T).T
            rays_d_world = torch.from_numpy(rays_d_world).cuda().float()

            vox_size = self.cfg['octree_raytracing_voxel_size']*self.cfg['sc_factor']
            level = int(np.floor(np.log2(2.0/vox_size)))
            near,far,_,ray_depths_in_out = self.octree_m.ray_trace(rays_o_world,rays_d_world,level=level)
            near = near.cpu().numpy()
            valid = (near>0).reshape(-1)
            cur_rays = cur_rays[valid]

        return cur_rays


    def compute_rays_z_in_out(self):
        N_rays = len(self.rays)
        rays_o = torch.zeros((N_rays,3), dtype=torch.float, device=self.rays.device)
        rays_d = self.rays[:,self.ray_dir_slice]
        viewdirs = rays_d/rays_d.norm(dim=-1,keepdim=True)

        frame_ids = self.rays[:,self.ray_frame_id_slice].long()
        tf = self.c2w_array[frame_ids.view(-1)]
        if self.models['pose_array'] is not None:
            tf = self.models['pose_array'].get_matrices(frame_ids)@tf

        rays_o_w = transform_pts(rays_o,tf)
        viewdirs_w = (tf[:,:3,:3]@viewdirs[:,None].permute(0,2,1))[:,:3,0]
        voxel_size = self.cfg['octree_raytracing_voxel_size']*self.cfg['sc_factor']
        level = int(np.floor(np.log2(2.0/voxel_size)))
        near,far,_,depths_in_out = self.octree_m.ray_trace(rays_o_w,viewdirs_w,level=level,debug=0)
        N_intersect = depths_in_out.shape[1]

        ########### Convert the time to Z
        z_in_out = (depths_in_out.cuda()*torch.abs(viewdirs[...,2].view(N_rays,1,1))).cuda()
        z_in_out = z_in_out.float()

        depths = self.rays[:,self.ray_depth_slice].view(-1,1)
        trunc = self.get_truncation()
        valid = (depths>=self.cfg['near']*self.cfg['sc_factor']) & (depths<=self.cfg['far']*self.cfg['sc_factor']).expand(-1,N_intersect)
        valid = valid & (z_in_out>0).all(dim=-1)      #(N_ray, N_intersect)
        z_in_out[valid] = torch.clip(z_in_out[valid],
        min=torch.zeros_like(z_in_out[valid]),
        max=torch.ones_like(z_in_out[valid])*(depths.reshape(-1,1,1).expand(-1,N_intersect,2)[valid]+trunc))

        self.z_in_out = z_in_out


    def add_new_frames(self,images,depths,masks,normal_maps,poses,occ_masks=None, new_pcd=None, reuse_weights=False):
        '''Add new frames and continue training
        @images: (N,H,W,3) new images
        @poses: All frames, they need to reset
        '''
        prev_n_image = len(self.images)
        down_scale_ratio = int(self.cfg['down_scale_ratio'])
        images = images[:, ::down_scale_ratio, ::down_scale_ratio]
        depths = depths[:, ::down_scale_ratio, ::down_scale_ratio]
        masks = masks[:, ::down_scale_ratio, ::down_scale_ratio]
        if normal_maps is not None:
            normal_maps = normal_maps[:, ::down_scale_ratio, ::down_scale_ratio]
            self.normal_maps = np.concatenate((self.normal_maps, normal_maps), axis=0)
        if occ_masks is not None:
            occ_masks = occ_masks[:, ::down_scale_ratio, ::down_scale_ratio]
            self.occ_masks = np.concatenate((self.occ_masks, occ_masks), axis=0)
        self.images = np.concatenate((self.images, images), axis=0)
        self.depths = np.concatenate((self.depths, depths), axis=0)
        self.masks = np.concatenate((self.masks, masks), axis=0)

        self.poses = poses.copy()
        self.c2w_array = torch.tensor(poses, dtype=torch.float).cuda()

        if self.cfg['use_octree']:
            pcd = new_pcd.voxel_down_sample(0.005)
            self.build_octree_pts = np.asarray(pcd.points).copy()
            self.build_octree()


        if not reuse_weights:
            self.create_nerf()

        else:
            ########### Add new frame weights
            if self.cfg['frame_features'] > 0:
                feature_array = FeatureArray(len(self.images), self.cfg['frame_features']).to(self.rays.device)
                if reuse_weights:
                    for i in range(prev_n_image):
                        feature_array.data.data[i] = self.models['feature_array'].data.data[i].detach().clone()
                self.models['feature_array'] = feature_array

            ########!NOTE Dont need to copy delta poses, they are new
            if self.cfg['optimize_poses']:
                pose_array = PoseArray(len(self.images),max_trans=self.cfg['max_trans']*self.cfg['sc_factor'],max_rot=self.cfg['max_rot']).to(self.rays.device)
                self.models['pose_array'] = pose_array

        self.create_optimizer()
        self.global_step = 0
        self.best_models = None
        self.best_loss = np.inf

        if not self.cfg['no_batching']:
            print("Using mask")
            rays_ = []
            for i_mask in range(prev_n_image, len(self.masks)):
                rays = self.make_frame_rays(i_mask)
                rays_.append(rays)
            rays = np.concatenate(rays_, axis=0)

            if self.cfg['denoise_depth_use_octree_cloud']:
                logging.info("denoise cloud")
                mask = (rays[:,self.ray_mask_slice]>0) & (rays[:,self.ray_depth_slice]<=self.cfg['far']*self.cfg['sc_factor'])
                rays_dir = rays[mask][:,self.ray_dir_slice]
                rays_depth = rays[mask][:,self.ray_depth_slice]
                pts3d = rays_dir*rays_depth.reshape(-1,1)
                frame_ids = rays[mask][:,self.ray_frame_id_slice].astype(int)
                pts3d_w = (self.poses[frame_ids]@to_homo(pts3d)[...,None])[:,:3,0]
                logging.info(f"Denoising rays based on octree cloud")

                kdtree = cKDTree(self.build_octree_pts)
                dists,indices = kdtree.query(pts3d_w,k=1,workers=-1)
                bad_mask = dists>0.02*self.cfg['sc_factor']
                bad_ids = np.arange(len(rays))[mask][bad_mask]
                rays[bad_ids,self.ray_depth_slice] = BAD_DEPTH*self.cfg['sc_factor']
                rays[bad_ids, self.ray_type_slice] = 1
                rays = rays[rays[:,self.ray_type_slice]==0]
                logging.info(f"bad_mask#={bad_mask.sum()}")

            rays = torch.tensor(rays, device=self.rays.device, dtype=torch.float)
            self.rays = torch.cat((self.rays,rays), dim=0)

        self.data_loader = DataLoader(rays=self.rays, batch_size=self.cfg['N_rand'])


    def build_octree(self):
        if self.cfg['save_octree_clouds']:
            dir = f"{self.cfg['save_dir']}/build_octree_cloud.ply"
            pcd = toOpen3dCloud(self.build_octree_pts)
            o3d.io.write_point_cloud(dir,pcd)
            if self._run is not None:
                self._run.add_artifact(dir)
        pts = torch.tensor(self.build_octree_pts).cuda().float()                   # Must be within [-1,1]
        octree_smallest_voxel_size = self.cfg['octree_smallest_voxel_size']*self.cfg['sc_factor']
        finest_n_voxels = 2.0/octree_smallest_voxel_size
        max_level = int(np.ceil(np.log2(finest_n_voxels)))
        octree_smallest_voxel_size = 2.0/(2**max_level)

        #################### Dilate
        dilate_radius = int(np.ceil(self.cfg['octree_dilate_size']/self.cfg['octree_smallest_voxel_size']))
        dilate_radius = max(1, dilate_radius)
        logging.info(f"Octree voxel dilate_radius:{dilate_radius}")
        shifts = []
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for dz in [-1,0,1]:
                    shifts.append([dx,dy,dz])
        shifts = torch.tensor(shifts).cuda().long()    # (27,3)
        coords = torch.floor((pts+1)/octree_smallest_voxel_size).long()  #(N,3)
        dilated_coords = coords.detach().clone()
        for iter in range(dilate_radius):
            dilated_coords = (dilated_coords[None].expand(shifts.shape[0],-1,-1) + shifts[:,None]).reshape(-1,3)
            dilated_coords = torch.unique(dilated_coords,dim=0)
        pts = (dilated_coords+0.5) * octree_smallest_voxel_size - 1
        pts = torch.clip(pts,-1,1)

        if self.cfg['save_octree_clouds']:
            pcd = toOpen3dCloud(pts.data.cpu().numpy())
            dir = f"{self.cfg['save_dir']}/build_octree_cloud_dilated.ply"
            o3d.io.write_point_cloud(dir,pcd)
            if self._run is not None:
                self._run.add_artifact(dir)
        ####################

        assert pts.min()>=-1 and pts.max()<=1
        self.octree_m = OctreeManager(pts, max_level)

        if self.cfg['save_octree_clouds']:
            dir = f"{self.cfg['save_dir']}/octree_boxes_max_level.ply"
            self.octree_m.draw_boxes(level=max_level,outfile=dir)
            if self._run is not None:
                self._run.add_artifact(dir)
        vox_size = self.cfg['octree_raytracing_voxel_size']*self.cfg['sc_factor']
        level = int(np.floor(np.log2(2.0/vox_size)))
        if self.cfg['save_octree_clouds']:
            dir = f"{self.cfg['save_dir']}/octree_boxes_ray_tracing_level.ply"
            self.octree_m.draw_boxes(level=level,outfile=dir)
            if self._run is not None:
                self._run.add_artifact(dir)


    def create_optimizer(self):
        params = []
        for k in self.models:
            if self.models[k] is not None and k!='pose_array':
                params += list(self.models[k].parameters())

        param_groups = [{'name':'basic', 'params':params, 'lr':self.cfg['lrate']}]
        if self.models['pose_array'] is not None:
            print('optimize poses')
            param_groups.append({'name':'pose_array', 'params':self.models['pose_array'].parameters(), 'lr':self.cfg['lrate_pose']})

        self.optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999),weight_decay=0,eps=1e-15)

        self.param_groups_init = copy.deepcopy(self.optimizer.param_groups)


    def load_weights(self,ckpt_path):
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        print("ckpt keys: ",ckpt.keys())
        self.models['model'].load_state_dict(ckpt['model'])
        if self.models['model_fine'] is not None:
            self.models['model_fine'].load_state_dict(ckpt['model_fine'])
        if self.models['embed_fn'] is not None:
            self.models['embed_fn'].load_state_dict(ckpt['embed_fn'])
        if self.models['embeddirs_fn'] is not None:
            self.models['embeddirs_fn'].load_state_dict(ckpt['embeddirs_fn'])
        if self.models['feature_array'] is not None:
            self.models['feature_array'].load_state_dict(ckpt['feature_array'])
        if self.models['pose_array'] is not None:
          self.models['pose_array'].load_state_dict(ckpt['pose_array'])
        if 'octree' in ckpt:
            self.octree_m = OctreeManager(octree=ckpt['octree'])
        self.optimizer.load_state_dict(ckpt['optimizer'])


    def save_weights(self,out_file, models):
        data = {
        'global_step': self.global_step,
        'model': models['model'].state_dict(),
        'optimizer': self.optimizer.state_dict(),
        }
        if 'model_fine' in models and models['model_fine'] is not None:
            data['model_fine'] = models['model_fine'].state_dict()
        if models['embed_fn'] is not None:
            data['embed_fn'] = models['embed_fn'].state_dict()
        if models['embeddirs_fn'] is not None:
            data['embeddirs_fn'] = models['embeddirs_fn'].state_dict()
        if models['pose_array'] is not None:
            data['pose_array'] = models['pose_array'].state_dict()
        if models['feature_array'] is not None:
            data['feature_array'] = models['feature_array'].state_dict()
        if self.octree_m is not None:
            data['octree'] = self.octree_m.octree
        dir = out_file
        torch.save(data,dir)
        print('Saved checkpoints at', dir)
        if self._run is not None:
            self._run.add_artifact(dir)
        dir1 = copy.deepcopy(dir)
        dir = f'{os.path.dirname(out_file)}/model_latest.pth'
        if dir1!=dir:
            os.system(f'cp {dir1} {dir}')
        if self._run is not None:
            self._run.add_artifact(dir)


    def schedule_lr(self):
        for i,param_group in enumerate(self.optimizer.param_groups):
            init_lr = self.param_groups_init[i]['lr']
            new_lrate = init_lr * (self.cfg['decay_rate'] ** (float(self.global_step) / self.N_iters))
            param_group['lr'] = new_lrate

    @torch.no_grad()
    def render_images(self,img_i,cur_rays=None):
        if cur_rays is None:
            frame_ids = self.rays[:, self.ray_frame_id_slice].cuda()
            cur_rays = self.rays[frame_ids==img_i].cuda()
        gt_depth = cur_rays[:,self.ray_depth_slice]
        gt_rgb = cur_rays[:,self.ray_rgb_slice].cpu()
        ray_type = cur_rays[:,self.ray_type_slice].data.cpu().numpy()
        ray_ids = torch.arange(len(self.rays), device=cur_rays.device)[frame_ids==img_i].long()
        ids = img_i * torch.ones([len(cur_rays), 1], device=cur_rays.device).long()

        ori_chunk = self.cfg['chunk']
        self.cfg['chunk'] = copy.deepcopy(self.cfg['N_rand'])
        with torch.no_grad():
            rgb, extras = self.render(rays=cur_rays, ray_ids=ray_ids, frame_ids=ids,lindisp=False,perturb=False,raw_noise_std=0, depth=gt_depth)
        self.cfg['chunk'] = ori_chunk

        sdf = extras['raw'][...,-1]
        z_vals = extras['z_vals']
        signs = sdf[:, 1:] * sdf[:, :-1]
        empty_rays = (signs>0).all(dim=-1)
        mask = signs<0
        inds = torch.argmax(mask.float(), axis=1)
        inds = inds[..., None]
        depth = torch.gather(z_vals,dim=1,index=inds)
        depth[empty_rays] = self.cfg['far']*self.cfg['sc_factor']
        depth = depth[..., None].data.cpu().numpy()

        rgb = rgb.data.cpu().numpy()

        rgb_full = np.zeros((self.H,self.W,3),dtype=float)
        depth_full = np.zeros((self.H,self.W),dtype=float)
        ray_mask_full = np.zeros((self.H,self.W,3),dtype=np.uint8)
        X = cur_rays[:,self.ray_dir_slice].data.cpu().numpy()
        X[:,[1,2]] = -X[:,[1,2]]
        projected = (self.K@X.T).T
        uvs = projected/projected[:,2].reshape(-1,1)
        uvs = uvs.round().astype(int)
        uvs_good = uvs[ray_type==0]
        ray_mask_full[uvs_good[:,1],uvs_good[:,0]] = [255,0,0]
        uvs_uncertain = uvs[ray_type==1]
        ray_mask_full[uvs_uncertain[:,1],uvs_uncertain[:,0]] = [0,255,0]
        rgb_full[uvs[:,1],uvs[:,0]] = rgb.reshape(-1,3)
        depth_full[uvs[:,1],uvs[:,0]] = depth.reshape(-1)
        gt_rgb_full = np.zeros((self.H,self.W,3),dtype=float)
        gt_rgb_full[uvs[:,1],uvs[:,0]] = gt_rgb.reshape(-1,3).data.cpu().numpy()
        gt_depth_full = np.zeros((self.H,self.W),dtype=float)
        gt_depth_full[uvs[:,1],uvs[:,0]] = gt_depth.reshape(-1).data.cpu().numpy()

        return rgb_full, depth_full, ray_mask_full, gt_rgb_full, gt_depth_full, extras


    def get_truncation(self):
        '''Annearl truncation over training
        '''
        if self.cfg['trunc_decay_type']=='linear':
            truncation = self.cfg['trunc_start'] - (self.cfg['trunc_start']-self.cfg['trunc']) * float(self.global_step)/self.cfg['n_step']
        elif self.cfg['trunc_decay_type']=='exp':
            lamb = np.log(self.cfg['trunc']/self.cfg['trunc_start']) / (self.cfg['n_step']/4)
            truncation = self.cfg['trunc_start']*np.exp(self.global_step*lamb)
            truncation = max(truncation,self.cfg['trunc'])
        else:
            truncation = self.cfg['trunc']

        truncation *= self.cfg['sc_factor']
        return truncation


    def train_loop(self, batch):
        target_s = batch[:, self.ray_rgb_slice]    # Color (N,3)
        target_d = batch[:, self.ray_depth_slice]    # Normalized scale (N)

        target_mask = batch[:,self.ray_mask_slice].bool().reshape(-1)
        frame_ids = batch[:,self.ray_frame_id_slice]

        rgb, extras = self.render(rays=batch, ray_ids=self.data_loader.batch_ray_ids, frame_ids=frame_ids,depth=target_d,lindisp=False,perturb=True,raw_noise_std=self.cfg['raw_noise_std'], near=batch[:,self.ray_near_slice], far=batch[:,self.ray_far_slice], get_normals=False)

        valid_samples = extras['valid_samples']   #(N_ray,N_samples)
        z_vals = extras['z_vals']  # [N_rand, N_samples + N_importance]
        sdf = extras['raw'][..., -1]

        N_rays,N_samples = sdf.shape[:2]
        valid_rays = (valid_samples>0).any(dim=-1).bool().reshape(N_rays) & (batch[:,self.ray_type_slice]==0)
        # valid_rays = (valid_samples>0).any(dim=-1).bool().reshape(N_rays)

        ray_type = batch[:,self.ray_type_slice].reshape(-1)
        ray_weights = torch.ones((N_rays), device=rgb.device, dtype=torch.float32)
        ray_weights[(frame_ids==0).view(-1)] = self.cfg['first_frame_weight']
        ray_weights = ray_weights*valid_rays.view(-1)
        sample_weights = ray_weights.view(N_rays,1).expand(-1,N_samples) * valid_samples
        img_loss = (((rgb-target_s)**2 * ray_weights.view(-1,1))).mean()
        rgb_loss = self.cfg['rgb_weight'] * img_loss
        loss = rgb_loss

        rgb0_loss = torch.tensor(0)
        if 'rgb0' in extras:
            img_loss0 = (((extras['rgb0']-target_s)**2 * ray_weights.view(-1,1))).mean()
            rgb0_loss = img_loss0*self.cfg['rgb_weight']
            loss += rgb0_loss

        depth_loss = torch.tensor(0)
        depth_loss0 = torch.tensor(0)
        if self.cfg['depth_weight']>0:
            signs = sdf[:, 1:] * sdf[:, :-1]
            mask = signs<0
            inds = torch.argmax(mask.float(), axis=1)
            inds = inds[..., None]
            z_min = torch.gather(z_vals,dim=1,index=inds)
            weights = ray_weights * (depth<=self.cfg['far']*self.cfg['sc_factor']) * (mask.any(dim=-1))
            depth_loss = ((z_min*weights-depth.view(-1,1)*weights)**2).mean() * self.cfg['depth_weight']
            loss = loss+depth_loss

        truncation = self.get_truncation()
        sample_weights[ray_type==1] = 0
        fs_loss, sdf_loss,front_mask,sdf_mask = get_sdf_loss(z_vals, target_d.reshape(-1,1).expand(-1,N_samples), sdf, truncation, self.cfg,return_mask=True, sample_weights=sample_weights, rays_d=batch[:,self.ray_dir_slice])
        fs_loss = fs_loss*self.cfg['fs_weight']
        sdf_loss = sdf_loss*self.cfg['trunc_weight']
        loss = loss + fs_loss + sdf_loss

        fs_rgb_loss = torch.tensor(0)
        if self.cfg['fs_rgb_weight']>0:
            fs_rgb_loss = ((((torch.sigmoid(extras['raw'][...,:3])-1)*front_mask[...,None])**2) * sample_weights[...,None]).mean()
            loss += fs_rgb_loss*self.cfg['fs_rgb_weight']

        eikonal_loss = torch.tensor(0)
        if self.cfg['eikonal_weight']>0:
            nerf_normals = extras['normals']
            eikonal_loss = ((torch.norm(nerf_normals[sdf<1], dim=-1)-1)**2).mean() * self.cfg['eikonal_weight']
            loss += eikonal_loss

        point_cloud_loss = torch.tensor(0)
        point_cloud_normal_loss = torch.tensor(0)


        reg_features = torch.tensor(0)
        if self.models['feature_array'] is not None:
            reg_features = self.cfg['feature_reg_weight'] * (self.models['feature_array'].data**2).mean()
            loss += reg_features
        if self.models['pose_array'] is not None:
            pose_array = self.models['pose_array']
            pose_reg = self.cfg['pose_reg_weight']*pose_array.data[1:].norm()
            loss += pose_reg

        variation_loss = torch.tensor(0)

        self.optimizer.zero_grad()

        self.amp_scaler.scale(loss).backward()

        self.amp_scaler.step(self.optimizer)
        self.amp_scaler.update()
        if self.global_step%10==0 and self.global_step>0:
            self.schedule_lr()

        if self.global_step%self.cfg['i_weights']==0 and self.global_step>0:
            self.save_weights(out_file=os.path.join(self.cfg['save_dir'], f'model_latest.pth'), models=self.models)

        if self.global_step % self.cfg['i_img'] == 0 and self.global_step>0:
            ids = torch.unique(self.rays[:, self.ray_frame_id_slice]).data.cpu().numpy().astype(int).tolist()
            ids.sort()
            last = ids[-1]
            ids = ids[::max(1,len(ids)//5)]
            if last not in ids:
                ids.append(last)
            canvas = []
            for frame_idx in ids:
                rgb, depth, ray_mask, gt_rgb, gt_depth, _ = self.render_images(frame_idx)
                mask_vis = (rgb*255*0.2 + ray_mask*0.8).astype(np.uint8)
                mask_vis = np.clip(mask_vis,0,255)
                rgb = np.concatenate((rgb,gt_rgb),axis=1)
                far = self.cfg['far']*self.cfg['sc_factor']
                gt_depth = np.clip(gt_depth, self.cfg['near']*self.cfg['sc_factor'], far)
                depth_vis = np.concatenate((to8b(depth / far), to8b(gt_depth / far)), axis=1)
                depth_vis = np.tile(depth_vis[...,None],(1,1,3))
                row = np.concatenate((to8b(rgb),depth_vis,mask_vis),axis=1)
                canvas.append(row)
            canvas = np.concatenate(canvas,axis=0).astype(np.uint8)
            dir = f"{self.cfg['save_dir']}/image_step_{self.global_step:07d}.png"
            imageio.imwrite(dir,canvas)
            if self._run is not None:
                self._run.add_artifact(dir)


        if self.global_step%self.cfg['i_print']==0:
            msg = f"Iter: {self.global_step}, valid_samples: {valid_samples.sum()}/{torch.numel(valid_samples)}, valid_rays: {valid_rays.sum()}/{torch.numel(valid_rays)}, "
            metrics = {
                'loss':loss.item(),
                'rgb_loss':rgb_loss.item(),
                'rgb0_loss':rgb0_loss.item(),
                'fs_rgb_loss': fs_rgb_loss.item(),
                'depth_loss':depth_loss.item(),
                'depth_loss0':depth_loss0.item(),
                'fs_loss':fs_loss.item(),
                'point_cloud_loss': point_cloud_loss.item(),
                'point_cloud_normal_loss':point_cloud_normal_loss.item(),
                'sdf_loss':sdf_loss.item(),
                'eikonal_loss': eikonal_loss.item(),
                "variation_loss": variation_loss.item(),
                'truncation(meter)': self.get_truncation()/self.cfg['sc_factor'],
                }
            if self.models['pose_array'] is not None:
                metrics['pose_reg'] = pose_reg.item()
            if 'feature_array' in self.models:
                metrics['reg_features'] = reg_features.item()
            for k in metrics.keys():
                msg += f"{k}: {metrics[k]:.7f}, "
            msg += "\n"
            logging.info(msg)

        if self._run is not None:
            for k in metrics.keys():
                self._run.log_scalar(k,metrics[k],self.global_step)

        if self.global_step % self.cfg['i_mesh'] == 0 and self.global_step > 0:
            with torch.no_grad():
                model = self.models['model_fine'] if self.models['model_fine'] is not None else self.models['model']
                isolevel = 0.0 if self.cfg['mode'] == 'sdf' else 20.0
                mesh = self.extract_mesh(isolevel=isolevel,voxel_size=self.cfg['mesh_resolution'])
                self.mesh = copy.deepcopy(mesh)
                if mesh is not None:
                    dir = os.path.join(self.cfg['save_dir'], f'step_{self.global_step:07d}_mesh_normalized_space.obj')
                    mesh.export(dir)
                    if self._run is not None:
                        self._run.add_artifact(dir)
                    dir = os.path.join(self.cfg['save_dir'], f'step_{self.global_step:07d}_mesh_real_world.obj')
                    if self.models['pose_array'] is not None:
                        _,offset = get_optimized_poses_in_real_world(self.poses,self.models['pose_array'],translation=self.cfg['translation'],sc_factor=self.cfg['sc_factor'])
                    else:
                        offset = np.eye(4)
                    mesh = mesh_to_real_world(mesh,offset,translation=self.cfg['translation'],sc_factor=self.cfg['sc_factor'])
                    mesh.export(dir)
                    if self._run is not None:
                        self._run.add_artifact(dir)

        if self.global_step % self.cfg['i_pose'] == 0 and self.global_step > 0:
            if self.models['pose_array'] is not None:
                optimized_poses,offset = get_optimized_poses_in_real_world(self.poses,self.models['pose_array'],translation=self.cfg['translation'],sc_factor=self.cfg['sc_factor'])
            else:
                optimized_poses = self.poses
            dir = os.path.join(self.cfg['save_dir'], f'step_{self.global_step:07d}_optimized_poses.txt')
            np.savetxt(dir,optimized_poses.reshape(-1,4))
            if self._run is not None:
                self._run.add_artifact(dir)


    def train(self):
        set_seed(0)
        iter = 0
        item = next(self.image_loader)
        self.get_ray_loader(*item)
        for iter in range(self.N_iters):
            batch = next(self.data_loader)
            self.train_loop(batch.cuda())
            self.global_step += 1
            if iter%(self.N_iters//10)==0:
                logging.info(f'train progress {iter}/{self.N_iters}')
            if (self.data_loader.pos >= len(self.data_loader.ids)) and (len(self.reader)>self.cfg['batch_size']):
                print("Loading next batch images")
                item = next(self.image_loader)
                self.get_ray_loader(*item)
        self.save_weights(out_file=os.path.join(self.cfg['save_dir'], f'model_latest.pth'), models=self.models)

    @torch.no_grad()
    def sample_rays_uniform_occupied_voxels(self,ray_ids,rays_d,depths_in_out,lindisp=False,perturb=False, depths=None, N_samples=None):
        '''We first connect the discontinuous boxes for each ray and treat it as uniform sample, then we disconnect into correct boxes
        @rays_d: (N_ray,3)
        @depths_in_out: Padded tensor each has (N_ray,N_intersect,2) tensor, the time travel of each ray
        '''
        N_rays = rays_d.shape[0]
        N_intersect = depths_in_out.shape[1]
        dirs = rays_d/rays_d.norm(dim=-1,keepdim=True)

        ########### Convert the time to Z
        z_in_out = depths_in_out.cuda()*torch.abs(dirs[...,2]).reshape(N_rays,1,1).cuda()

        if depths is not None:
            depths = depths.reshape(-1,1)
            trunc = self.get_truncation()
            valid = (depths>=self.cfg['near']*self.cfg['sc_factor']) & (depths<=self.cfg['far']*self.cfg['sc_factor']).expand(-1,N_intersect)
            valid = valid & (z_in_out>0).all(dim=-1)      #(N_ray, N_intersect)
            z_in_out[valid] = torch.clip(z_in_out[valid],
                min=torch.zeros_like(z_in_out[valid]),
                max=torch.ones_like(z_in_out[valid])*(depths.reshape(-1,1,1).expand(-1,N_intersect,2)[valid]+trunc))


        depths_lens = z_in_out[:,:,1]-z_in_out[:,:,0]   #(N_ray,N_intersect)
        z_vals_continous = sample_rays_uniform(N_samples,torch.zeros((N_rays,1),device=z_in_out.device).reshape(-1,1),depths_lens.sum(dim=-1).reshape(-1,1),lindisp=lindisp,perturb=perturb)     #(N_ray,N_sample)

        ############# Option2 mycuda extension
        N_samples = z_vals_continous.shape[1]
        z_vals = torch.zeros((N_rays,N_samples), dtype=torch.float, device=rays_d.device)
        z_vals = common.sampleRaysUniformOccupiedVoxels(z_in_out.contiguous(),z_vals_continous.contiguous(), z_vals)
        z_vals = z_vals.float().to(rays_d.device)    #(N_ray,N_sample)

        return z_vals,z_vals_continous


    def render_rays(self,ray_batch,retraw=True,lindisp=False,perturb=False,raw_noise_std=0.,depth=None, get_normals=False, ray_ids=None):
        """Volumetric rendering.
        Args:
        ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction, frame_ids.
        model: function. Model for predicting RGB and density at each point
            in space.
        N_samples: int. Number of different times to sample along each ray.
        retraw: bool. If True, include model's raw, unprocessed predictions.
        lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
        perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
        N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to model_fine.
        model_fine: "fine" network with same spec as model.
        raw_noise_std: ...
        verbose: bool. If True, print more debugging info.
        @depth: depth (from depth image) values of the ray (N_ray,1)
        Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
        disp_map: [num_rays]. Disparity map. 1 / depth.
        acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
        raw: [num_rays, num_samples, 4]. Raw predictions from model.
        rgb0: See rgb_map. Output for coarse model.
        disp0: See disp_map. Output for coarse model.
        acc0: See acc_map. Output for coarse model.
        z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """
        N_rays = ray_batch.shape[0]
        rays_d = ray_batch[:,self.ray_dir_slice]
        rays_o = torch.zeros_like(rays_d)
        viewdirs = rays_d/rays_d.norm(dim=-1,keepdim=True)

        frame_ids = ray_batch[:,self.ray_frame_id_slice].long()
        tf = self.c2w_array[frame_ids]
        if self.models['pose_array'] is not None:
            tf = self.models['pose_array'].get_matrices(frame_ids)@tf

        rays_o_w = transform_pts(rays_o,tf)
        viewdirs_w = (tf[:,:3,:3]@viewdirs[:,None].permute(0,2,1))[:,:3,0]
        voxel_size = self.cfg['octree_raytracing_voxel_size']*self.cfg['sc_factor']
        level = int(np.floor(np.log2(2.0/voxel_size)))
        near,far,_,depths_in_out = self.octree_m.ray_trace(rays_o_w,viewdirs_w,level=level,debug=0)
        z_vals,_ = self.sample_rays_uniform_occupied_voxels(ray_ids=ray_ids,rays_d=viewdirs,depths_in_out=depths_in_out,lindisp=lindisp,perturb=perturb, depths=depth, N_samples=self.cfg['N_samples'])

        if self.cfg['N_samples_around_depth']>0 and depth is not None:      #!NOTE only fine when depths are all valid
            valid_depth_mask = (depth>=self.cfg['near']*self.cfg['sc_factor']) & (depth<=self.cfg['far']*self.cfg['sc_factor'])
            valid_depth_mask = valid_depth_mask.reshape(-1).cuda()
            trunc = self.get_truncation()
            near_depth = depth[valid_depth_mask]-trunc
            far_depth = depth[valid_depth_mask]+trunc*self.cfg['neg_trunc_ratio']
            z_vals_around_depth = torch.zeros((N_rays,self.cfg['N_samples_around_depth']), device=ray_batch.device).float()
            # if torch.sum(inside_mask)>0:
            z_vals_around_depth[valid_depth_mask] = sample_rays_uniform(self.cfg['N_samples_around_depth'],near_depth.reshape(-1,1),far_depth.reshape(-1,1),lindisp=lindisp,perturb=perturb)
            invalid_depth_mask = valid_depth_mask==0

            if invalid_depth_mask.any() and self.cfg['use_octree']:
                z_vals_invalid,_ = self.sample_rays_uniform_occupied_voxels(ray_ids=ray_ids[invalid_depth_mask],rays_d=viewdirs[invalid_depth_mask],depths_in_out=depths_in_out[invalid_depth_mask],lindisp=lindisp,perturb=perturb, depths=None, N_samples=self.cfg['N_samples_around_depth'])
                z_vals_around_depth[invalid_depth_mask] = z_vals_invalid
            else:
                z_vals_around_depth[invalid_depth_mask] = sample_rays_uniform(self.cfg['N_samples_around_depth'],near[invalid_depth_mask].reshape(-1,1),far[invalid_depth_mask].reshape(-1,1),lindisp=lindisp,perturb=perturb)

            z_vals = torch.cat((z_vals,z_vals_around_depth), dim=-1)
            valid_samples = torch.ones(z_vals.shape, dtype=torch.bool, device=ray_batch.device)   # During pose update if ray out of box, it becomes invalid

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

        deformation = None
        raw,normals,valid_samples = self.run_network(pts, viewdirs, frame_ids, tf=tf, valid_samples=valid_samples, get_normals=get_normals)  # [N_rays, N_samples, 4]

        rgb_map, weights = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std=raw_noise_std, valid_samples=valid_samples, depth=depth)

        if self.cfg['N_importance'] > 0:
            rgb_map_0 = rgb_map

            for iter in range(self.cfg['N_importance_iter']):
                z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
                z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.cfg['N_importance'], det=(perturb==0.))
                z_samples = z_samples.detach()
                valid_samples_importance = torch.ones(z_samples.shape, dtype=torch.bool).to(z_vals.device)
                valid_samples_importance[torch.all(valid_samples==0, dim=-1).reshape(-1)] = 0

                if self.models['model_fine'] is not None and self.models['model_fine']!=self.models['model']:
                    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
                    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + self.cfg['N_importance'], 3]
                    raw, normals,valid_samples = self.run_network(pts, viewdirs, frame_ids, tf=tf, valid_samples=valid_samples, get_normals=False)
                else:
                    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_samples[..., :, None]  # [N_rays, N_samples + self.cfg['N_importance'], 3]
                    raw_fine,valid_samples_importance = self.run_network(pts, viewdirs, frame_ids, tf=tf, valid_samples=valid_samples_importance, get_normals=False)
                    z_vals = torch.cat([z_vals, z_samples], -1)  #(N_ray, N_sample)
                    indices = torch.argsort(z_vals, dim=-1)
                    z_vals = torch.gather(z_vals,dim=1,index=indices)
                    raw = torch.gather(torch.cat([raw, raw_fine], dim=1), dim=1, index=indices[...,None].expand(-1,-1,raw.shape[-1]))
                    valid_samples = torch.cat((valid_samples,valid_samples_importance), dim=-1)

                rgb_map, weights = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std=raw_noise_std,valid_samples=valid_samples)

        ret = {'rgb_map' : rgb_map, 'valid_samples':valid_samples, 'weights':weights, 'z_vals':z_vals}

        if retraw:
            ret['raw'] = raw

        if normals is not None:
            ret['normals'] = normals

        if deformation is not None:
            ret['deformation'] = deformation

        if self.cfg['N_importance'] > 0:
            ret['rgb0'] = rgb_map_0

        return ret


    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, valid_samples=None, depth=None):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        truncation = self.get_truncation()
        if depth is not None:
            depth = depth.view(-1,1)

        if valid_samples is None:
            valid_samples = torch.ones(z_vals.shape, dtype=torch.bool).to(z_vals.device)

        def raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw) * dists)

        def sdf2weights(sdf):
            sdf_from_depth = (depth.view(-1,1)-z_vals)/truncation
            weights = torch.sigmoid(sdf_from_depth*self.cfg['sdf_lambda']) * torch.sigmoid(-sdf_from_depth*self.cfg['sdf_lambda'])  # This not work well

            invalid = (depth>self.cfg['far']*self.cfg['sc_factor']).reshape(-1)
            # mask = ((z_vals-depth)*dir_norm<=truncation*self.cfg['neg_trunc_ratio']) & ((z_vals-depth)*dir_norm>=-truncation)
            mask = (z_vals-depth<=truncation*self.cfg['neg_trunc_ratio']) & (z_vals-depth>=-truncation)
            # mask = (z_vals<=depth.reshape(-1,1) + truncation*self.cfg['neg_trunc_ratio']) & (z_vals>=depth.reshape(-1,1) - truncation)
            weights[~invalid] = weights[~invalid] * mask[~invalid]
            weights[invalid] = 0

            return weights / (weights.sum(dim=-1,keepdim=True) + 1e-10)

        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]

        if self.cfg['mode'] == 'density':
            # Predict density of each sample along each ray. Higher values imply higher likelihood of being absorbed at this point. A cumprod() is used to express the idea of the ray not having reflected up to this sample yet.
            noise = 0.
            if raw_noise_std > 0.:
                noise = torch.randn(raw[...,3].shape) * raw_noise_std
            dists = z_vals[...,1:] - z_vals[...,:-1]
            dists = torch.cat([dists, torch.tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
            dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
            alpha = raw2alpha(raw[..., 3] + noise, dists)      # [N_rays, N_samples]
            weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=z_vals.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        elif self.cfg['mode'] == 'sdf':
            weights = sdf2weights(raw[..., 3])
        else:
            raise Exception('Unknown color integration mode' + self.cfg['mode'])

        weights[valid_samples==0] = 0
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        return rgb_map, weights


    def render(self, rays, ray_ids=None, frame_ids=None,depth=None,lindisp=False,perturb=False,raw_noise_std=0.0, get_normals=False, near=None, far=None):
        """Render rays
        Args:
        H: int. Height of image in pixels.
        W: int. Width of image in pixels.
        K: float. Focal length of pinhole camera.
        chunk: int. Maximum number of rays to process simultaneously. Used to
            control maximum memory usage. Does not affect final results.
        rays: array of shape [batch_size, 6]. Ray origin and direction for
            each example in batch.
        ndc: bool. If True, represent ray origin, direction in NDC coordinates. Only true for llff data
        near: float or array of shape [batch_size]. Nearest distance for a ray.
        far: float or array of shape [batch_size]. Farthest distance for a ray.
        @depth: depth values (N_ray,1)
        Returns:
        rgb_map: [batch_size, 3]. Predicted RGB values for rays.
        disp_map: [batch_size]. Disparity map. Inverse of depth.
        acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
        extras: dict with everything.
        """
        all_ret = self.batchify_rays(rays,depth=depth,lindisp=lindisp,perturb=perturb,raw_noise_std=raw_noise_std, get_normals=get_normals, ray_ids=ray_ids)

        k_extract = ['rgb_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]



    def batchify_rays(self,rays_flat, depth=None,lindisp=False,perturb=False,raw_noise_std=0.0, get_normals=False, ray_ids=None):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        chunk = self.cfg['chunk']
        for i in range(0, rays_flat.shape[0], chunk):
            if depth is not None:
                cur_depth = depth[i:i+chunk]
            else:
                cur_depth = None
            if ray_ids is not None:
                cur_ray_ids = ray_ids[i:i+chunk]
            else:
                cur_ray_ids = None
            ret = self.render_rays(rays_flat[i:i+chunk],depth=cur_depth,lindisp=lindisp,perturb=perturb,raw_noise_std=raw_noise_std, get_normals=get_normals, ray_ids=cur_ray_ids)
            for k in ret:
                if ret[k] is None:
                    continue
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret


    def run_network(self, inputs, viewdirs, frame_ids, tf, latent_code=None, valid_samples=None, get_normals=False):
        """Prepares inputs and applies network 'fn'.
        @inputs: (N_ray,N_sample,3) sampled points on rays in GL camera's frame
        @viewdirs: (N_ray,3) unit length vector in camera frame, z-axis backward
        @frame_ids: (N_ray)
        @tf: (N_ray,4,4)
        @latent_code: (N_ray, D)
        """
        N_ray,N_sample = inputs.shape[:2]

        if valid_samples is None:
            valid_samples = torch.ones((N_ray,N_sample), dtype=torch.bool, device=inputs.device)

        # Ensure inputs is a floating-point tensor
        inputs = inputs.to(torch.float32)
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

        tf_flat = tf[:,None].expand(-1,N_sample,-1,-1).reshape(-1,4,4)
        inputs_flat = transform_pts(inputs_flat, tf_flat)
        
        # Ensure inputs_flat is still a floating-point tensor after transformation
        inputs_flat = inputs_flat.to(torch.float32)

        valid_samples = valid_samples.bool() & (torch.abs(inputs_flat)<=1).all(dim=-1).view(N_ray,N_sample).bool()

        embedded = torch.zeros((inputs_flat.shape[0],self.models['embed_fn'].out_dim), device=inputs_flat.device)
        if valid_samples is None:
            valid_samples = torch.ones((N_ray,N_sample), dtype=torch.bool, device=inputs_flat.device)

        if get_normals:
            if inputs_flat.requires_grad==False:
                inputs_flat.requires_grad = True

        with torch.amp.autocast('cuda', enabled=self.cfg['amp']):
            if self.cfg['i_embed'] in [3]:
                embedded[valid_samples.reshape(-1)], valid_samples_embed = self.models['embed_fn'](inputs_flat[valid_samples.reshape(-1)])
                valid_samples = valid_samples.reshape(-1)
                prev_valid_ids = valid_samples.nonzero().reshape(-1)
                bad_ids = prev_valid_ids[valid_samples_embed==0]
                new_valid_ids = torch.ones((N_ray*N_sample),device=inputs.device).bool()
                new_valid_ids[bad_ids] = 0
                valid_samples = valid_samples & new_valid_ids
                valid_samples = valid_samples.reshape(N_ray,N_sample).bool()
            else:
                # Ensure we're passing a floating-point tensor to embed_fn
                valid_inputs = inputs_flat[valid_samples.reshape(-1)].to(torch.float32)
                embedded[valid_samples.reshape(-1)] = self.models['embed_fn'](valid_inputs).to(embedded.dtype)
        embedded = embedded.float()

        # Add latent code
        if self.models['feature_array'] is not None:
            if latent_code is None:
                frame_features = self.models['feature_array'](frame_ids)
                D = frame_features.shape[-1]
                frame_features = frame_features[:,None].expand(-1,N_sample,-1).reshape(-1,D)
            else:
                D = latent_code.shape[-1]
                frame_features = latent_code[:,None].expand(N_ray,N_sample,latent_code.shape[-1]).reshape(-1,D)
            embedded = torch.cat([embedded, frame_features], -1)

        # Add view directions
        if self.models['embeddirs_fn'] is not None:
            input_dirs = (tf[..., :3, :3]@viewdirs[...,None])[...,0]  #(N_ray,3)
            embedded_dirs = self.models['embeddirs_fn'](input_dirs)
            tmp = embedded_dirs.shape[1:]
            embedded_dirs_flat = embedded_dirs[:,None].expand(-1,N_sample,*tmp).reshape(-1,*tmp)
            embedded = torch.cat([embedded, embedded_dirs_flat], -1)

        outputs_flat = []
        with torch.amp.autocast('cuda', enabled=self.cfg['amp']):
            chunk = self.cfg['netchunk']
            for i in range(0,embedded.shape[0],chunk):
                out = self.models['model'](embedded[i:i+chunk])
                outputs_flat.append(out)
        outputs_flat = torch.cat(outputs_flat, dim=0).float()
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]).float()

        normals = None
        if get_normals:
            sdf = outputs[...,-1]
            d_output = torch.zeros(sdf.shape, device=sdf.device)
            normals = torch.autograd.grad(outputs=sdf,inputs=inputs_flat,grad_outputs=d_output,create_graph=False,retain_graph=True,only_inputs=True,allow_unused=True)[0]
            normals = normals.reshape(N_ray,N_sample,3)

        return outputs,normals,valid_samples


    def run_network_density(self, inputs, get_normals=False):
        """Directly query the network w/o pose transformations or deformations (inputs are already in normalized [-1,1]); Particularly used for mesh extraction
        @inputs: (N,3) sampled points on rays in GL camera's frame
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

        inputs_flat = torch.clip(inputs_flat,-1,1)
        valid_samples = torch.ones((len(inputs_flat)),device=inputs.device).bool()

        if not inputs_flat.requires_grad:
            inputs_flat.requires_grad = True

        with torch.amp.autocast('cuda', enabled=self.cfg['amp']):
            if self.cfg['i_embed'] in [3]:
                embedded, valid_samples_embed = self.models['embed_fn'](inputs_flat)
                valid_samples = valid_samples.reshape(-1)
                prev_valid_ids = valid_samples.nonzero().reshape(-1)
                bad_ids = prev_valid_ids[valid_samples_embed==0]
                new_valid_ids = torch.ones((len(inputs_flat)),device=inputs.device).bool()
                new_valid_ids[bad_ids] = 0
                valid_samples = valid_samples & new_valid_ids
            else:
                embedded = self.models['embed_fn'](inputs_flat)
        embedded = embedded.float()
        input_ch = embedded.shape[-1]

        ######## Add dummy view_dirs related stuff, not actually used
        # input_ch_views = self.models['model'].input_ch_views
        # embedded = torch.cat((embedded, torch.zeros((len(embedded),input_ch_views),device=inputs.device).float()),dim=-1)

        outputs_flat = []
        with torch.amp.autocast('cuda', enabled=self.cfg['amp']):
            chunk = self.cfg['netchunk']
            for i in range(0,embedded.shape[0],chunk):
                alpha = self.models['model'].forward_sdf(embedded[i:i+chunk])   #(N,1)
                outputs_flat.append(alpha.reshape(-1,1))
        outputs_flat = torch.cat(outputs_flat,dim=0).float()
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])

        if get_normals:
            d_output = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
            normal = torch.autograd.grad(outputs=outputs,inputs=inputs_flat,grad_outputs=d_output,create_graph=False,retain_graph=True,only_inputs=True,allow_unused=True)[0]
            outputs = torch.cat((outputs, normal), dim=-1)

        return outputs,valid_samples


    @torch.no_grad()
    def extract_mesh(self, level=None, voxel_size=0.003, isolevel=0.0, mesh_savepath='',return_sigma=False):
        # Query network on dense 3d grid of points
        voxel_size *= self.cfg['sc_factor']  # in "network space"

        bounds = np.array(self.cfg['bounding_box']).reshape(2,3)
        x_min, x_max = bounds[0,0], bounds[1,0]
        y_min, y_max = bounds[0,1], bounds[1,1]
        z_min, z_max = bounds[0,2], bounds[1,2]
        tx = np.arange(x_min+0.5*voxel_size, x_max, voxel_size)
        ty = np.arange(y_min+0.5*voxel_size, y_max, voxel_size)
        tz = np.arange(z_min+0.5*voxel_size, z_max, voxel_size)
        N = len(tx)
        query_pts = torch.tensor(np.stack(np.meshgrid(tx, ty, tz, indexing='ij'), -1).astype(np.float32).reshape(-1,3)).float().cuda()

        if self.octree_m is not None:
            vox_size = self.cfg['octree_raytracing_voxel_size']*self.cfg['sc_factor']
            level = int(np.floor(np.log2(2.0/vox_size)))
            center_ids = self.octree_m.get_center_ids(query_pts, level)
            valid = center_ids>=0
        else:
            valid = torch.ones(len(query_pts), dtype=bool).cuda()

        logging.info(f'query_pts:{query_pts.shape}, valid:{valid.sum()}')
        flat = query_pts[valid]

        sigma = []
        chunk = self.cfg['netchunk']
        for i in range(0,flat.shape[0],chunk):
            inputs = flat[i:i+chunk]
            with torch.no_grad():
                outputs,valid_samples = self.run_network_density(inputs=inputs)
            sigma.append(outputs)
        sigma = torch.cat(sigma, dim=0)
        sigma_ = torch.ones((N**3)).float().cuda()
        sigma_[valid] = sigma.reshape(-1)
        sigma = sigma_.reshape(N,N,N).data.cpu().numpy()

        logging.info('Running Marching Cubes')
        from skimage import measure
        try:
            vertices, triangles, normals, values = measure.marching_cubes(sigma, isolevel)
        except Exception as e:
            logging.info(f"ERROR Marching Cubes {e}")
            return None

        logging.info(f'done V:{vertices.shape}, F:{triangles.shape}')

        # Rescale and translate
        voxel_size_ndc = np.array([tx[-1] - tx[0], ty[-1] - ty[0], tz[-1] - tz[0]]) / np.array([[tx.shape[0] - 1, ty.shape[0] - 1, tz.shape[0] - 1]])
        offset = np.array([tx[0], ty[0], tz[0]])
        vertices[:, :3] = voxel_size_ndc.reshape(1,3) * vertices[:, :3] + offset.reshape(1,3)

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles, process=False)

        if return_sigma:
            return mesh,sigma,query_pts

        return mesh


