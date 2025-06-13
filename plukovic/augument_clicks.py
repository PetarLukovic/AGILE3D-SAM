import os
import sys
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm

from plukovic.visualisation import visualize_scene_with_trajectory
from plukovic.scannet_scene import SensorData
from segment_anything import sam_model_registry, SamPredictor
from sklearn.cluster import KMeans

def check_camera_visibility(scene_data, idx, click_coordinate, near=0.01, far=10000.0, depth_threshold_mm=50):

    try:
        intrinsics = scene_data.__get_camera_intrinsics__(idx)
        pose = scene_data.__get_camera_pose__(idx)
        depth = scene_data.__get_camera_depth__(idx)
    except Exception:
        return False, None

    if torch.isinf(pose).any() or torch.isnan(pose).any():
        return False, None

    try:
        cam_pose_inv = torch.inverse(pose)
    except RuntimeError:
        return False, None

    # Ensure homogeneous coordinate
    if click_coordinate.shape[0] == 3:
        click_coordinate = torch.cat([click_coordinate, click_coordinate.new_tensor([1.0])])

    point_cam = cam_pose_inv @ click_coordinate
    z = point_cam[2]

    if torch.isnan(z) or z <= 0:
        return False, None

    fx, fy = intrinsics[0, 0].item(), intrinsics[1, 1].item()
    cx, cy = intrinsics[0, 2].item(), intrinsics[1, 2].item()

    x_pixel = (fx * point_cam[0].item() / z.item()) + cx
    y_pixel = (fy * point_cam[1].item() / z.item()) + cy

    h, w = depth.shape
    if not (0 <= x_pixel < w and 0 <= y_pixel < h):
        return False, None

    if not (near <= z <= far):
        return False, None

    x_pix_int = int(round(x_pixel))
    y_pix_int = int(round(y_pixel))

    margin = 5
    if x_pix_int < margin or y_pix_int < margin or x_pix_int >= (w - margin) or y_pix_int >= (h - margin):
        return False, None

    depth_val_mm = depth[y_pix_int, x_pix_int]
    if depth_val_mm == 0:
        return False, None

    z_depth_map_m = depth_val_mm / 1000.0
    depth_diff_mm = abs(z_depth_map_m - z.item()) * 1000.0

    if depth_diff_mm > depth_threshold_mm:
        return False, None

    return True, (x_pix_int, y_pix_int)

def subsample_cameras(scene_data, cameras):
    return cameras[::2]

def z_filter(scene_data, cameras, point_3d):

    point_3d = point_3d.cpu().numpy() if hasattr(point_3d, 'cpu') else np.array(point_3d)
    visible_cameras = []
    poses = scene_data.poses
    poses = poses[cameras]

    for idx, pose in enumerate(poses):
        R = pose[:3, :3]
        t = pose[:3, 3]
        point_cam = R.T @ (point_3d - t)

        if point_cam[2] > 0:
            visible_cameras.append(idx)

    return visible_cameras

def angle_sort(scene_data, cameras, point_3d):
    point_3d = point_3d.cpu().numpy() if hasattr(point_3d, 'cpu') else np.array(point_3d)
    poses = scene_data.poses
    poses = poses[cameras]

    cosines = []
    for idx, pose in enumerate(poses):
        R = pose[:3, :3]
        t = pose[:3, 3]

        cam_pos = t

        view_dir = -R[:, 2]

        cam_to_point = point_3d - cam_pos
        cam_to_point /= np.linalg.norm(cam_to_point)

        cosine = np.dot(view_dir, cam_to_point)

        cosines.append((cosine, cameras[idx]))

    cosines.sort(key=lambda x: x[0])
    sorted_cameras = [idx for _, idx in cosines]

    return sorted_cameras

def find_visible_cameras(scene_data, click_coordinate, num_cameras=5, max_attempts=50, visualize=False):

    camera_indices = np.arange(len(scene_data.poses))
    #visualize_scene_with_trajectory(scene_data, camera_indices, [click_coordinate.tolist()], subsample_frustrums=True)
    print(f"    Original number of cameras: {len(camera_indices)}")
    camera_indices = z_filter(scene_data, camera_indices, click_coordinate)
    print(f"    Number of camera after z-visibility filter: {len(camera_indices)}")
    camera_indices = angle_sort(scene_data, camera_indices, click_coordinate)
    camera_indices = subsample_cameras(scene_data, camera_indices)
    print(f"    Number of cameras after subsampling: {len(camera_indices)}")
    #visualize_scene_with_trajectory(scene_data, camera_indices, [click_coordinate.tolist()], subsample_frustrums=True)

    visible_cameras = []
    pixel_coords = []

    attempts = 0
    i = 0

    for _ in tqdm(range(min(max_attempts, len(camera_indices) - i)), desc="    Finding visible cameras"):
        if len(visible_cameras) >= num_cameras:
            break

        idx = camera_indices[i]
        i += 1

        is_visible, pixel = check_camera_visibility(scene_data, idx, click_coordinate)
        if is_visible:
            visible_cameras.append(idx)
            pixel_coords.append(pixel)

        attempts += 1

    if len(visible_cameras) < num_cameras:
        print(f"    Warning: Only found {len(visible_cameras)} visible cameras out of {num_cameras} requested.")
    
    if visualize:
        for cam_id, pixel in zip(visible_cameras, pixel_coords):
            rgb = scene_data.__get_camera_rgb__(cam_id)
            rgb_vis = rgb.permute(1, 2, 0).cpu().numpy()

            # Ensure the pixel is within image bounds
            scale_x = 1296 / 640
            scale_y = 968 / 480
            x , y = pixel
            x = int(x * scale_x)
            y = int(y * scale_y)

            if x < 0 or x >= rgb_vis.shape[1] or y < 0 or y >= rgb_vis.shape[0]:
                print(f"    Warning: Pixel coordinates ({x}, {y}) are out of bounds for camera {cam_id}.")
                continue

            # Draw the selected pixel
            cv2.circle(rgb_vis, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

            # Show the image
            cv2.imshow(f'Camera {cam_id} - RGB with Selected Pixel', rgb_vis)
            cv2.waitKey(0)
            cv2.destroyWindow(f'Camera {cam_id} - RGB with Selected Pixel')

    return visible_cameras, pixel_coords

def extract_sam_masks(scene_data, cameras, pixels):

    if cameras is None or len(cameras) == 0:
        print("    No cameras selected. Skipping SAM mask extraction.")
        return None
    
    sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    all_masks = []

    for cam_id, pixel in tqdm(zip(cameras, pixels), total=len(cameras), desc="    Extracting SAM masks"):

        # Get the RGB image and depth map for the camera
        img = scene_data.__get_camera_rgb__(cam_id)
        img = img.permute(1, 2, 0).cpu().numpy()
        
        # If the image is in float32, scale to uint8
        if img.dtype == np.float32:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        # Get the corresponding depth map for the camera
        depth_raw = scene_data.__get_camera_depth__(cam_id)
        depth_shape = depth_raw.shape

        # Rescale pixel coordinates from depth map to RGB image size
        rgb_height, rgb_width = img.shape[:2]
        depth_height, depth_width = depth_shape

        # Calculate the scaling factors for both x and y axes
        scale_x = rgb_width / depth_width
        scale_y = rgb_height / depth_height

        # Rescale the pixel coordinates
        x_scaled = int(pixel[0] * scale_x)
        y_scaled = int(pixel[1] * scale_y)

        # Ensure the scaled coordinates are within bounds
        x_scaled = min(max(x_scaled, 0), rgb_width - 1)
        y_scaled = min(max(y_scaled, 0), rgb_height - 1)

        # Set the image for the SAM predictor
        predictor.set_image(img)

        # Prepare the point coordinates in the correct format
        input_point = np.array([[x_scaled, y_scaled]], dtype=np.float32)
        input_label = np.array([1], dtype=np.int32).reshape(-1)

        # Run SAM mask prediction
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Get the best mask based on the highest score
        best_mask = masks[np.argmax(scores)]
        best_mask = (best_mask * 255).astype(np.uint8)

        # Append the mask as a tensor to the list
        all_masks.append(torch.tensor(best_mask, dtype=torch.uint8))

    # Stack all masks into a tensor
    all_masks_tensor = torch.stack(all_masks)

    return all_masks_tensor
'''
def augment_click(scene_data, cameras, sam_masks, num_clicks, visualize=False):

    if cameras is None or len(cameras) == 0:
        print("    No cameras selected. Skipping augumentation mask extraction.")
        return None

    sampled_clicks = []

    for id, cam_id in tqdm(enumerate(cameras), total=len(cameras), desc="    Augumenting cameras", unit="camera"):
        # Retrieve the camera pose, intrinsics, and other data
        pose = scene_data.__get_camera_pose__(cam_id)
        intrinsics = scene_data.__get_camera_intrinsics__(cam_id)
        depth_raw = scene_data.__get_camera_depth__(cam_id)
        rgb = scene_data.__get_camera_rgb__(cam_id)
        mask = sam_masks[id].cpu()

        # Resize depth map to match the mask size if necessary
        if mask.shape != depth_raw.shape:
            mask = cv2.resize(mask.numpy(), (depth_raw.shape[1], depth_raw.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = torch.tensor(mask, dtype=torch.uint8)

        # Ensure depth is in meters (ScanNet depth is in mm by default)
        depth = depth_raw.float() / 1000.0

        # Find foreground pixels (non-zero mask values)
        foreground_pixels = torch.nonzero(mask > 0)
        if foreground_pixels.size(0) == 0:
            print(f"    No foreground pixels found in camera {cam_id}.")
            continue

        # Randomly sample a foreground pixel
        y, x = foreground_pixels[random.randint(0, foreground_pixels.size(0) - 1)]
        d = depth[y.item(), x.item()]

        if d <= 0:
            print(f"    Invalid depth value at pixel ({x.item()}, {y.item()}) in camera {cam_id}.")
            continue

        # Camera intrinsics for transforming to 3D camera space
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Transform 2D pixel (x, y) to camera space (x_cam, y_cam, z_cam)
        x_cam = (x.item() - cx) * d / fx
        y_cam = (y.item() - cy) * d / fy
        z_cam = d
        point_cam = torch.tensor([x_cam, y_cam, z_cam, 1.0])

        # Transform from camera space to world space
        point_world = torch.matmul(pose, point_cam)
        sampled_clicks.append(point_world[:3].cpu().tolist())

        # Visualization
        if visualize:
            # Prepare RGB and mask for visualization
            rgb_vis = rgb.permute(1, 2, 0).cpu().numpy()
            mask_vis = mask.numpy().astype(np.uint8) * 255  # Binary mask to 0-255 image
            mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel mask

            # Resize both RGB and mask to the same target size
            target_size = (812, 512)
            rgb_vis = cv2.resize(rgb_vis, target_size, interpolation=cv2.INTER_AREA)
            mask_vis = cv2.resize(mask_vis, target_size, interpolation=cv2.INTER_NEAREST)

            # Rescale the pixel coordinates to match the resized images
            scale_x = target_size[0] / mask.shape[1]
            scale_y = target_size[1] / mask.shape[0]
            x_scaled = int(x.item() * scale_x)
            y_scaled = int(y.item() * scale_y)

            # Draw selected point on both images (RGB and Mask)
            cv2.circle(rgb_vis, (x_scaled, y_scaled), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.circle(mask_vis, (x_scaled, y_scaled), radius=5, color=(0, 255, 0), thickness=-1)

            # Stack both images vertically and display
            combined = np.hstack((rgb_vis, mask_vis))
            cv2.imshow(f'Camera {cam_id} - RGB and SAM Mask', combined)
            cv2.waitKey(0)  # Wait for any key press
            cv2.destroyWindow(f'Camera {cam_id} - RGB and SAM Mask')

    return sampled_clicks
'''

def augment_click(scene_data, cameras, sam_masks, num_clicks, visualize=False):
    if cameras is None or len(cameras) == 0:
        print("    No cameras selected. Skipping augmentation mask extraction.")
        return None

    sampled_clicks = []
    attempt = 1

    while len(sampled_clicks) < num_clicks:
        print(f"    Attempting click augumentation, attempt: {attempt}")
        attempt += 1
        for id, cam_id in enumerate(tqdm(cameras, desc="        Processing cameras", unit="cam")):
            if len(sampled_clicks) >= num_clicks:
                break
            try:
                pose = scene_data.__get_camera_pose__(cam_id)
                intrinsics = scene_data.__get_camera_intrinsics__(cam_id)
                depth_raw = scene_data.__get_camera_depth__(cam_id)
                rgb = scene_data.__get_camera_rgb__(cam_id)
                mask = sam_masks[id % len(sam_masks)].cpu()

                if mask.shape != depth_raw.shape:
                    mask = cv2.resize(mask.numpy(), (depth_raw.shape[1], depth_raw.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask = torch.tensor(mask, dtype=torch.uint8)

                depth = depth_raw.float() / 1000.0
                foreground_pixels = torch.nonzero(mask > 0)
                if foreground_pixels.size(0) == 0:
                    continue

                d = -1
                sampling_attempt = 0
                while d<= 0 and sampling_attempt < 5:
                    y, x = foreground_pixels[random.randint(0, foreground_pixels.size(0) - 1)]
                    d = depth[y.item(), x.item()]
                    if d <= 0:
                        print(f"        Trial: {sampling_attempt}. Sampled pixel ({y.item()}, {x.item()}) on camera {cam_id} is occluded. Trying again ...")
                    sampling_attempt += 1

                fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                cx, cy = intrinsics[0, 2], intrinsics[1, 2]
                x_cam = (x.item() - cx) * d / fx
                y_cam = (y.item() - cy) * d / fy
                z_cam = d
                point_cam = torch.tensor([x_cam, y_cam, z_cam, 1.0])
                point_world = torch.matmul(pose, point_cam)
                sampled_clicks.append(point_world[:3].cpu().tolist())

                if visualize:
                    rgb_vis = rgb.permute(1, 2, 0).cpu().numpy()
                    mask_vis = mask.numpy().astype(np.uint8) * 255
                    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
                    target_size = (812, 512)
                    rgb_vis = cv2.resize(rgb_vis, target_size)
                    mask_vis = cv2.resize(mask_vis, target_size)
                    scale_x = target_size[0] / mask.shape[1]
                    scale_y = target_size[1] / mask.shape[0]
                    x_scaled = int(x.item() * scale_x)
                    y_scaled = int(y.item() * scale_y)
                    cv2.circle(rgb_vis, (x_scaled, y_scaled), radius=5, color=(0, 255, 0), thickness=-1)
                    cv2.circle(mask_vis, (x_scaled, y_scaled), radius=5, color=(0, 255, 0), thickness=-1)
                    combined = np.hstack((rgb_vis, mask_vis))
                    cv2.imshow(f'Camera {cam_id} - RGB and SAM Mask', combined)
                    cv2.waitKey(0)
                    cv2.destroyWindow(f'Camera {cam_id} - RGB and SAM Mask')

            except Exception as e:
                print(f"    Error processing camera {cam_id}: {e}")
                continue

    if len(sampled_clicks) < num_clicks:
        print(f"    Only {len(sampled_clicks)} clicks collected (requested {num_clicks}).")

    return sampled_clicks

def process_click(scene_data, click_coordinate, verbose=False, visualize=False, num_new_clicks=5):


    if not torch.is_tensor(click_coordinate):
        click_coordinate = torch.tensor(click_coordinate, dtype=torch.float32)

    new_clicks = []
    clicks = []
    clicks.append(click_coordinate.cpu().tolist())

    if verbose: print(f"Processing click: {click_coordinate} (translated), on scene: {scene_data.scene_name}, for {num_new_clicks} new clicks.")

    if verbose: print(f"    Finding visible cameras ...")
    selected_cameras, pixels = find_visible_cameras(scene_data, click_coordinate, num_cameras=num_new_clicks, visualize=visualize)
    if verbose: 
        print(f"    Found {len(selected_cameras)}/{num_new_clicks} visible cameras:")
        for cam_id, pixel in zip(selected_cameras, pixels):
            print(f"        Camera ID: {cam_id}, Pixel: {pixel}")
    if verbose: print(f"    Done finding visible cameras.")

    sam_masks = extract_sam_masks(scene_data, selected_cameras, pixels)
    if sam_masks is not None:
        if verbose: print(f"    Extracted {len(sam_masks)} SAM masks.")

    new_clicks = augment_click(scene_data, selected_cameras, sam_masks, num_new_clicks, visualize=visualize)
    if new_clicks is not None:
        if verbose:
            print(f"    Augmented clicks generated:")
            for i, click in enumerate(new_clicks):
                print(f"        New click {i}: {click}")

        clicks.extend(new_clicks)

    if visualize:
        visualize_scene_with_trajectory(scene_data, selected_cameras, clicks)

    if verbose: print(f"Done processing click: {click_coordinate}, on scene: {scene_data.scene_name}")

    return clicks, selected_cameras

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.isdir(folder_path):
        print(f"The path {folder_path} is not a valid directory.")
        sys.exit(1)

    scenes = os.listdir(folder_path)

    click_coordinates = []
    click_coordinates.append(np.array([5.341, 2.683, 0.955]))  # stove
    click_coordinates.append(np.array([8.137, 2.630, 0.550]))  # couch    
    click_coordinates.append(np.array([3.635, 0.909, 0.995]))  # stove
    click_coordinates.append(np.array([2.635, 1.909, 0.995]))  # couch

    for scene, click in zip(scenes, click_coordinates):
        scene_data = SensorData(os.path.join(folder_path, scene, scene + ".hdf5"))
        process_click(scene_data, click, verbose=True) 

if __name__ == "__main__":
    main()

