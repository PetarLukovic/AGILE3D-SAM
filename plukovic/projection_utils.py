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