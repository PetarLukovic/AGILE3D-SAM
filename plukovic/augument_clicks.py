import cv2
import torch
import random
import numpy as np

from tqdm import tqdm

from plukovic.visualisation import (
    visualize_scene_with_trajectory,
    visualize_camera_with_point,
    visualize_camera_with_mask_with_point
)

from plukovic.sam_utils import (
    extract_sam_masks,
)

from plukovic.projection_utils import (
    check_camera_visibility
)

def subsample_cameras(scene_data, cameras):
    return cameras[::2]

def z_filter(scene_data, cameras, point_3d):
    point_3d = point_3d.cpu().numpy() if hasattr(point_3d, 'cpu') else np.array(point_3d)
    visible_cameras = []
    poses = scene_data.poses
    poses = poses[cameras]

    R_all = poses[:, :3, :3]
    t_all = poses[:, :3, 3]

    point_rel = point_3d - t_all
    point_cam_all = np.einsum('nij,nj->ni', R_all.transpose(0, 2, 1), point_rel)
    visible_cameras = np.where(point_cam_all[:, 2] > 0)[0].tolist()

    '''
    for idx, pose in enumerate(poses):
        R = pose[:3, :3]
        t = pose[:3, 3]
        point_cam = R.T @ (point_3d - t)

        if point_cam[2] > 0:
            visible_cameras.append(idx)
    '''

    return visible_cameras

def angle_sort(scene_data, cameras, point_3d):
    point_3d = point_3d.cpu().numpy() if hasattr(point_3d, 'cpu') else np.array(point_3d)
    poses = scene_data.poses
    poses = poses[cameras]

    R_all = poses[:, :3, :3] 
    t_all = poses[:, :3, 3]

    view_dirs = -R_all[:, :, 2]
    cam_to_point = point_3d - t_all
    cam_to_point /= np.linalg.norm(cam_to_point, axis=1, keepdims=True) 
    cosines_vals = np.einsum('ij,ij->i', view_dirs, cam_to_point)
    cosines = list(zip(cosines_vals, cameras))
    cosines.sort(key=lambda x: x[0])
    sorted_cameras = [idx for _, idx in cosines]

    '''
    for idx, pose in enumerate(poses):
        R = pose[:3, :3]
        t = pose[:3, 3]

        cam_pos = t

        view_dir = -R[:, 2]

        cam_to_point = point_3d - cam_pos
        cam_to_point /= np.linalg.norm(cam_to_point)

        cosine = np.dot(view_dir, cam_to_point)

        cosines.append((cosine, cameras[idx]))
    '''

    return sorted_cameras

def find_visible_cameras(scene_data, click_coordinate, config):

    camera_indices = np.arange(len(scene_data.poses))

    if config['visualize']:
        visualize_scene_with_trajectory(scene_data, camera_indices, [click_coordinate.tolist()], subsample_frustrums=True)

    print(f"    Original number of cameras: {len(camera_indices)}")
    camera_indices = z_filter(scene_data, camera_indices, click_coordinate)
    print(f"    Number of camera after z-visibility filter: {len(camera_indices)}")
    camera_indices = angle_sort(scene_data, camera_indices, click_coordinate)
    camera_indices = subsample_cameras(scene_data, camera_indices)
    print(f"    Number of cameras after subsampling: {len(camera_indices)}")

    if config['visualize']:
        visualize_scene_with_trajectory(scene_data, camera_indices, [click_coordinate.tolist()], subsample_frustrums=True)

    visible_cameras = []
    pixel_coords = []
    i = 0

    for _ in tqdm(range(min(config['max_attempts_camera_selection'], len(camera_indices) - i)), desc="    Finding visible cameras"):
        if len(visible_cameras) >= config['num_new_clicks']:
            break

        idx = camera_indices[i]
        i += 1

        is_visible, pixel = check_camera_visibility(scene_data, idx, click_coordinate, config)
        if is_visible:
            visible_cameras.append(idx)
            pixel_coords.append(pixel)

    if len(visible_cameras) < config['num_new_clicks']:
        print(f"    Warning: Only found {len(visible_cameras)} visible cameras out of {config['num_new_clicks']} requested.")
    
    if config['visualize']:
        for cam_id, pixel in zip(visible_cameras, pixel_coords):
            visualize_camera_with_point(scene_data, cam_id, pixel)

    return visible_cameras, pixel_coords

def augment_click(scene_data, cameras, sam_masks, config):
    if cameras is None or len(cameras) == 0:
        print("    No cameras selected. Skipping augmentation mask extraction.")
        return None

    sampled_clicks = []
    attempt = 1

    while len(sampled_clicks) < config['num_new_clicks']:
        print(f"    Attempting click augumentation, attempt: {attempt}")
        attempt += 1
        for id, cam_id in enumerate(tqdm(cameras, desc="        Processing cameras", unit="cam")):
            if len(sampled_clicks) >= config['num_new_clicks']:
                break
            try:
                pose = scene_data.__get_camera_pose__(cam_id)
                fx, fy, cx, cy = scene_data.__get_camera_intrinsics__(cam_id)
                depth_raw = scene_data.__get_camera_depth__(cam_id)
                mask = sam_masks[id % len(sam_masks)].cpu()

                if mask.shape != depth_raw.shape:
                    mask = cv2.resize(mask.numpy(), (depth_raw.shape[1], depth_raw.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask = torch.tensor(mask, dtype=torch.uint8)

                foreground_pixels = torch.nonzero(mask > 0)
                if foreground_pixels.size(0) == 0:
                    continue

                d = -1
                sampling_attempt = 0
                while d<= 0 and sampling_attempt < config['max_attemps_pixel_sampling']:
                    y, x = foreground_pixels[random.randint(0, foreground_pixels.size(0) - 1)]
                    d = depth_raw[y.item(), x.item()]
                    if d <= 0:
                        print(f"        Trial: {sampling_attempt}. Sampled pixel ({y.item()}, {x.item()}) on camera {cam_id} is occluded. Trying again ...")
                    sampling_attempt += 1

                x_cam = (x.item() - cx) * d / fx
                y_cam = (y.item() - cy) * d / fy
                z_cam = d
                point_cam = torch.tensor([x_cam, y_cam, z_cam, 1.0])
                point_world = torch.matmul(pose, point_cam)
                sampled_clicks.append(point_world[:3].cpu().tolist())

                if config['visualize']:
                    point = (x.item(), y.item())
                    visualize_camera_with_mask_with_point(scene_data, cam_id, mask, point)

            except Exception as e:
                print(f"    Error processing camera {cam_id}: {e}")
                continue

    if len(sampled_clicks) < config['num_new_clicks']:
        print(f"    Only {len(sampled_clicks)} clicks collected (requested {config['num_new_clicks']}).")

    return sampled_clicks

def process_click(scene_data, click_coordinate, config):


    if not torch.is_tensor(click_coordinate):
        click_coordinate = torch.tensor(click_coordinate, dtype=torch.float32)

    new_clicks = []
    clicks = []
    clicks.append(click_coordinate.cpu().tolist())

    if config['verbose']: print(f"Processing click: {click_coordinate} (translated), on scene: {scene_data.scene_name}, for {config['num_new_clicks']} new clicks.")

    if config['verbose']: print(f"    Finding visible cameras ...")
    selected_cameras, pixels = find_visible_cameras(scene_data, click_coordinate, config)
    if config['verbose']: 
        print(f"    Found {len(selected_cameras)}/{config['num_new_clicks']} visible cameras:")
        for cam_id, pixel in zip(selected_cameras, pixels):
            print(f"        Camera ID: {cam_id}, Pixel: {pixel}")
    if config['verbose']: print(f"    Done finding visible cameras.")

    sam_masks = extract_sam_masks(scene_data, selected_cameras, pixels)
    if sam_masks is not None:
        if config['verbose']: print(f"    Extracted {len(sam_masks)} SAM masks.")

    new_clicks = augment_click(scene_data, selected_cameras, sam_masks, config)
    if new_clicks is not None:
        if config['verbose']:
            print(f"    Augmented clicks generated:")
            for i, click in enumerate(new_clicks):
                print(f"        New click {i}: {click}")

        clicks.extend(new_clicks)

    if config['visualize']:
        visualize_scene_with_trajectory(scene_data, selected_cameras, clicks)

    if config['verbose']: print(f"Done processing click: {click_coordinate}, on scene: {scene_data.scene_name}")

    return clicks, selected_cameras

def main():
    return

if __name__ == "__main__":
    main()