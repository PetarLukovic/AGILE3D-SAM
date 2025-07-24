import torch

def check_camera_visibility(scene_data, idx, click_coordinate, config):

    try:
        fx, fy, cx, cy = scene_data.__get_camera_intrinsics__(idx)
        pose = scene_data.__get_camera_pose__(idx)
        depth = scene_data.__get_camera_depth__(idx)
    except Exception as e:
        print(e)
        return False, None

    if torch.isinf(pose).any() or torch.isnan(pose).any():
        return False, None

    try:
        cam_pose_inv = torch.inverse(pose)
    except RuntimeError:
        return False, None

    if click_coordinate.shape[0] == 3:
        click_coordinate = torch.cat([click_coordinate, click_coordinate.new_tensor([1.0])])

    point_cam = cam_pose_inv @ click_coordinate
    z = point_cam[2]

    if torch.isnan(z) or z <= 0:
        return False, None

    x_pixel = (fx * point_cam[0].item() / z.item()) + cx
    y_pixel = (fy * point_cam[1].item() / z.item()) + cy

    x_pix_int = int(x_pixel)
    y_pix_int = int(y_pixel)

    h, w = scene_data.__get_depth_resolution__(idx)

    if not (0 <= x_pix_int < w and 0 <= y_pix_int < h):
        return False, None

    if not (config['projection_near_m'] <= z <= config['projection_far_m']):
        return False, None

    if x_pix_int < config['object_click_padding'] or y_pix_int < config['object_click_padding'] or x_pix_int >= (w - config['object_click_padding']) or y_pix_int >= (h - config['object_click_padding']):
        return False, None

    depth_val_mm = depth[y_pix_int, x_pix_int]
    if depth_val_mm == 0:
        return False, None

    z_depth_map_m = depth_val_mm / 1000.0
    depth_diff_mm = abs(z_depth_map_m - z.item()) * 1000.0

    if depth_diff_mm > config['depth_threshold_mm']:
        return False, None

    # coordinates are in depth map pixel space
    return True, (x_pixel, y_pixel)

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

    view_dirs = -R_all[:, :, 2] 
    cam_to_point = point_3d.unsqueeze(0) - t_all 
    cam_to_point = cam_to_point / cam_to_point.norm(dim=1, keepdim=True) 

    cosines_vals = (view_dirs * cam_to_point).sum(dim=1) 

    sorted_indices = torch.argsort(cosines_vals)
    sorted_cameras = cameras[sorted_indices]

    return sorted_cameras
