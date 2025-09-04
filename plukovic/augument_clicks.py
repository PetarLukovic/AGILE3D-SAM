import torch
import random
from tqdm import tqdm

from plukovic.visualisation import (
    visualize_scene_with_trajectory,
    visualize_camera_with_point,
    visualize_camera_with_mask_with_points
)

from plukovic.sam_extraction import (
    extract_sam_masks,
)

from plukovic.projection import (
    raytrace_3d_to_pixel,
    raytrace_pixel_to_3d,
    z_filter,
    angle_sort
)

from plukovic.sampler import (
    sample_cameras,
    sample_masks,
)

def find_visible_cameras(scene_data, click_coordinate, config):

    camera_indices = torch.arange(len(scene_data.poses)).to(click_coordinate.device)

    if config['visualize']:
        visualize_scene_with_trajectory(scene_data, camera_indices.cpu().numpy(), [click_coordinate.cpu().numpy()], [], subsample_frustrums=False)

    if config['verbose']: print(f"    Original number of cameras: {len(camera_indices)}")
    camera_indices = z_filter(scene_data, camera_indices, click_coordinate)
    if config['verbose']: print(f"    Number of camera after z-visibility filter: {len(camera_indices)}")
    camera_indices = angle_sort(scene_data, camera_indices, click_coordinate)
    camera_indices = sample_cameras(scene_data, camera_indices)
    if config['verbose']: print(f"    Number of cameras after subsampling: {len(camera_indices)}")

    if config['visualize']:
        visualize_scene_with_trajectory(scene_data, camera_indices.cpu().numpy(), [click_coordinate.cpu().numpy()], [], subsample_frustrums=False)

    visible_cameras = []
    pixel_coords = []
    i = 0

    num_new_clicks = config['num_cameras']
    camera_indices = camera_indices[:config['max_attempts_camera_selection']]
    random.shuffle(camera_indices)

    for _ in tqdm(range(min(config['max_attempts_camera_selection'], len(camera_indices) - i)), desc=f"    Finding visible cameras ({num_new_clicks})", disable=not config['verbose']):
        
        if len(visible_cameras) >= num_new_clicks: break

        if i >= len(camera_indices):
            if config['verbose']: print(f"    Warning: Reached the end of camera indices while trying to find {num_new_clicks} cameras.")
            break

        idx = camera_indices[i]
        i += 1

        is_visible, pixel = raytrace_3d_to_pixel(scene_data, idx, click_coordinate, config)

        if is_visible:
            visible_cameras.append(idx)
            pixel_coords.append(pixel)

    if len(visible_cameras) < num_new_clicks and config['verbose']:
        print(f"        Warning: Only found {len(visible_cameras)} visible cameras out of {num_new_clicks} requested.")
    
    if config['visualize']:
        for cam_id, pixel in zip(visible_cameras, pixel_coords):
            visualize_camera_with_point(scene_data, cam_id, pixel)

    return visible_cameras, pixel_coords

def augment_click(scene_data, cameras, sam_masks, config, foreground=True): 

    if cameras is None or len(cameras) == 0 and config['verbose']:
        print("    No cameras selected. Skipping augmentation mask extraction.")
        return None

    sampled_clicks = []

    if foreground: num_new_clicks = config['num_new_clicks_fg']
    else: num_new_clicks = config['num_new_clicks_bg']

    description = f"    Augmenting clicks {'(foreground)' if foreground else '(background)'}"
    for _ in tqdm(range(len(cameras) * config['num_samples']), desc=description, disable=not config['verbose']):
        if len(sampled_clicks) >= num_new_clicks: break
        for cam_id in sam_masks.keys():
            if len(sampled_clicks) >= num_new_clicks: break
            try:
                cam_id = int(cam_id)

                mask = sam_masks[str(cam_id)]['foreground'].to(config['device']) if foreground else sam_masks[str(cam_id)]['background'].to(config['device'])

                key = 'foreground_samples' if foreground else 'background_samples'
                if len(sam_masks[str(cam_id)][key]) == 0: continue
                x, y = sam_masks[str(cam_id)][key].pop()

                visible, point_world = raytrace_pixel_to_3d(scene_data, cam_id, (x, y), config=config)

                if not visible: continue
                if foreground: sam_masks[str(cam_id)]['selected_foreground_samples'].append((x, y))
                else: sam_masks[str(cam_id)]['selected_background_samples'].append((x, y))
                sampled_clicks.append(point_world.cpu().tolist())

            except Exception as e:
                print(f"        Error processing camera {cam_id}: {e}")
                continue

    if len(sampled_clicks) < num_new_clicks and config['verbose']: print(f"        Only {len(sampled_clicks)} clicks collected (requested {num_new_clicks}).")
    elif config['verbose']: print(f"        Successfully collected {len(sampled_clicks)} clicks (requested {num_new_clicks}).")

    if config['visualize']:
        for cam_id in sam_masks.keys():
            mask = sam_masks[cam_id]['foreground'].to(config['device']) if foreground else sam_masks[cam_id]['background'].to(config['device'])
            sampled_points = sam_masks[cam_id]['selected_foreground_samples'] if foreground else sam_masks[cam_id]['selected_background_samples']
            title = f"Camera {cam_id} - {'Foreground' if foreground else 'Background'} selected point"
            visualize_camera_with_mask_with_points(scene_data, int(cam_id), mask, sampled_points, title)

    return sampled_clicks

def process_click(scene_data, click_coordinate, config):

    if not config['verbose']:
        print("Verbose parameter is False. Supressing all output ...")

    click_coordinate = torch.from_numpy(click_coordinate).to(config['device'])
    scene_data.DEVICE = click_coordinate.device

    clicks_fg = []
    clicks_bg = []

    if config['verbose']: print(f"Processing click: {click_coordinate} (translated), on scene: {scene_data.scene_name}.")
    if config['verbose']: print(f"Generating {config['num_new_clicks_fg']} new clicks on foreground and {config['num_new_clicks_bg']} new clicks on background.")
    if config['verbose']: print(f"    Finding visible cameras ...")
    selected_cameras, pixels = find_visible_cameras(scene_data, click_coordinate, config)
    if config['verbose']:
        if len(selected_cameras) > 0:
            print(f"    Found {len(selected_cameras)}/{config['num_cameras']} visible cameras:")
            for cam_id, pixel in zip(selected_cameras, pixels):
                pixel = (int(pixel[0].item()), int(pixel[1].item()))
                print(f"        Camera ID: {cam_id}, Pixel: {pixel}")

    if config['verbose']: print(f"    Done finding visible cameras.")

    sam_masks = extract_sam_masks(scene_data, selected_cameras, pixels, config)
    if sam_masks is not None:
        if config['verbose']: print(f"    Extracted {len(sam_masks)} SAM masks.")

    sam_masks = sample_masks(scene_data, sam_masks, config)
    if config['verbose']:
        print(f"    Sampling done for {len(sam_masks)} SAM masks.")
        for camera_id in sam_masks.keys():
            sam_mask = sam_masks[camera_id]
            if 'foreground_samples' in sam_mask:
                print(f"        Extracted {len(sam_mask['foreground_samples'])}/{config['num_samples']} foreground samples for camera {camera_id}.")
            if 'background_samples' in sam_mask:
                print(f"        Extracted {len(sam_mask['background_samples'])}/{config['num_samples']} background samples for camera {camera_id}.")

    new_clicks = augment_click(scene_data, selected_cameras, sam_masks, config, True)
    if new_clicks is not None and new_clicks != []:
        if config['verbose']:
            print(f"    Augmented clicks generated (foreground):")
            for i, click in enumerate(new_clicks):
                print(f"        New click {i}: {click}")

        clicks_fg.extend(new_clicks)

    new_clicks = augment_click(scene_data, selected_cameras, sam_masks, config, False)
    if new_clicks is not None and new_clicks != []:
        if config['verbose']:
            print(f"    Augmented clicks generated (background):")
            for i, click in enumerate(new_clicks):
                print(f"        New click {i}: {click}")

        clicks_bg.extend(new_clicks)
    
    if config['visualize']: visualize_scene_with_trajectory(scene_data, selected_cameras, [click_coordinate.cpu().tolist()] + clicks_fg, clicks_bg, subsample_frustrums=False)

    if config['verbose']: print(f"Done processing click: {click_coordinate}, on scene: {scene_data.scene_name}")

    return clicks_fg, clicks_bg, selected_cameras

def main():
    return

if __name__ == "__main__":
    main()