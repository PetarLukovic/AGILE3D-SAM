import torch
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def raytrace_3d_to_pixel(scene_data, idx, point_3d, config=None):

    if config is None:
        config = {}

    pcd_points = np.asarray(scene_data.point_cloud.points)
    tree = cKDTree(pcd_points)

    pose = scene_data.__get_camera_pose__(idx)
    fx, fy, cx, cy = scene_data.__get_camera_intrinsics__(idx)
    h, w = scene_data.__get_camera_resolution__(idx)

    if point_3d.shape[0] == 3: point_h = torch.cat([point_3d, point_3d.new_tensor([1.0])])
    else: point_h = point_3d

    cam_pose_inv = torch.inverse(pose)
    point_h = point_h.to(cam_pose_inv.device)
    point_cam = cam_pose_inv @ point_h
    z = point_cam[2]

    if z <= 0 or torch.isnan(z): return False, None
    if z < config.get('projection_near_m', 0.01) or z > config.get('projection_far_m', 100.0): return False, None

    x_pixel = (fx * point_cam[0].item() / z.item()) + cx
    y_pixel = (fy * point_cam[1].item() / z.item()) + cy
    
    pad = config.get('object_click_padding', 0)
    if not (pad <= x_pixel < w - pad and pad <= y_pixel < h - pad): return False, None

    cam_pos = pose[:3, 3].cpu().numpy()
    point_np = point_3d.cpu().numpy()
    ray_dir = point_np - cam_pos
    ray_len = np.linalg.norm(ray_dir)
    ray_dir /= ray_len

    num_samples = config.get('ray_samples', 50)
    max_depth = config.get('projection_far_m', 100.0)
    t_vals = np.linspace(config.get('projection_near_m', 0.1), ray_len, num_samples)
    sample_points = cam_pos[None, :] + t_vals[:, None] * ray_dir[None, :]
    distances, _ = tree.query(sample_points)
    occlusion_threshold = config.get('occlusion_threshold_m', 0.01)
    valid_samples = distances[:-5]

    if np.any(valid_samples < occlusion_threshold): return False, None

    return True, torch.tensor([x_pixel, y_pixel], dtype=torch.float32, device=point_3d.device)


def raytrace_pixel_to_3d(scene_data, idx, pixel_coord, config=None):
    """
    Casts a ray from a pixel into the 3D point cloud and returns first visible hit.
    """
    if config is None:
        config = {}

    if False: visualize_camera_ray(scene_data, idx=idx, pixel_coord=pixel_coord, length=20.0)

    pcd_points = np.asarray(scene_data.point_cloud.points)
    tree = cKDTree(pcd_points)

    # --- Camera intrinsics and pose ---
    x, y = pixel_coord
    pose = scene_data.__get_camera_pose__(idx)
    fx, fy, cx, cy = scene_data.__get_camera_intrinsics__(idx)

    # Build ray in camera space
    z_default = config.get('default_projection_depth_m', 10.0)
    x_cam = (x - cx) * z_default / fx
    y_cam = (y - cy) * z_default / fy
    ray_dir_cam = torch.tensor([x_cam, y_cam, z_default],
                               dtype=pose.dtype, device=pose.device)
    ray_dir_cam /= torch.norm(ray_dir_cam)

    # Transform ray to world
    cam_pos = pose[:3, 3].cpu().numpy()
    R = pose[:3, :3]
    ray_dir_world = (R @ ray_dir_cam).cpu().numpy()

    # Sample points along the ray
    num_samples = config.get('ray_samples', 50)
    max_depth = config.get('projection_far_m', 100.0)
    t_vals = np.linspace(config.get('projection_near_m', 0.1), max_depth, num_samples)
    sample_points = cam_pos[None, :] + t_vals[:, None] * ray_dir_world[None, :]

    # Nearest neighbor distance to point cloud
    distances, indices = tree.query(sample_points)

    # Occlusion threshold: consider ray "hit" when close enough to point cloud
    occlusion_threshold = config.get('occlusion_threshold_m', 0.2)

    # Find first hit (closest point along the ray within threshold)
    hit_idx = np.where(distances < occlusion_threshold)[0]
    if len(hit_idx) == 0:
        return False, None

    point_3d = torch.tensor(sample_points[hit_idx[0]], dtype=pose.dtype, device=pose.device)

    # Depth range check
    distance = torch.norm(point_3d - pose[:3, 3])
    if distance < config.get('projection_near_m', 0.1) or distance > config.get('projection_far_m', 100.0):
        return False, None

    return True, point_3d


def z_filter(scene_data, cameras, point_3d):
    poses = torch.from_numpy(scene_data.poses[cameras.cpu().numpy()]).to(point_3d.device)

    R_all = poses[:, :3, :3]
    t_all = poses[:, :3, 3]

    point_rel = point_3d.unsqueeze(0) - t_all 
    R_transpose = R_all.transpose(1, 2)       
    point_cam_all = torch.bmm(R_transpose, point_rel.unsqueeze(2)).squeeze(2)

    visible_mask = point_cam_all[:, 2] > 0
    visible_cameras = cameras[visible_mask]

    return visible_cameras

def angle_sort(scene_data, cameras, point_3d):
    poses = torch.from_numpy(scene_data.poses[cameras.cpu().numpy()]).to(point_3d.device)

    R_all = poses[:, :3, :3] 
    t_all = poses[:, :3, 3]   

    # Camera forward direction (negative z-axis in world)
    view_dirs = -R_all[:, :, 2] 
    cam_to_point = point_3d.unsqueeze(0) - t_all 
    cam_to_point = cam_to_point / cam_to_point.norm(dim=1, keepdim=True) 

    cosines_vals = (view_dirs * cam_to_point).sum(dim=1)

    # Sort with best-aligned (highest cosine) first
    sorted_indices = torch.argsort(cosines_vals, descending=True)
    sorted_cameras = cameras[sorted_indices]

    return sorted_cameras

def visualize_camera_ray(scene_data, idx, pixel_coord, length=20.0):
    """
    Visualize a ray from the camera through a pixel, clipped at a fixed length.
    
    Parameters:
        scene_data: object containing `point_cloud` (Open3D) and camera methods
        idx: camera index
        pixel_coord: (x, y) pixel coordinate
        length: length to clip the ray (meters)
    """
    # Camera intrinsics and pose
    x, y = pixel_coord
    pose = scene_data.__get_camera_pose__(idx)
    fx, fy, cx, cy = scene_data.__get_camera_intrinsics__(idx)

    # Ray direction in camera space
    z_default = 1.0  # arbitrary, only direction matters
    x_cam = (x - cx) * z_default / fx
    y_cam = (y - cy) * z_default / fy
    ray_dir_cam = torch.tensor([x_cam, y_cam, z_default],
                               dtype=pose.dtype, device=pose.device)
    ray_dir_cam /= torch.norm(ray_dir_cam)

    # Transform to world coordinates
    cam_pos = pose[:3, 3].cpu().numpy()
    R = pose[:3, :3]
    ray_dir_world = (R @ ray_dir_cam).cpu().numpy()

    # Clip the ray to the specified length
    end_point = cam_pos + ray_dir_world * length

    # Create Open3D geometries
    cam_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    cam_sphere.translate(cam_pos)
    cam_sphere.paint_uniform_color([0, 0, 1])  # blue for camera

    ray_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([cam_pos, end_point]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    ray_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # red for ray

    # Draw the scene: point cloud + camera + ray
    o3d.visualization.draw_geometries([scene_data.point_cloud, cam_sphere, ray_line])