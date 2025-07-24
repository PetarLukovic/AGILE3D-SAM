import torch
import random
import torch.nn.functional as F

from plukovic.visualisation import (
    visualize_camera_with_mask_with_points,
)

def random_sampling(sam_mask, num_samples):
    mask_pixels = torch.nonzero(sam_mask >= 0.5)
    num_samples = min(num_samples, mask_pixels.size(0))
    indices = random.sample(range(mask_pixels.size(0)), num_samples)
    sampled_pixels = mask_pixels[indices].to(sam_mask.device)
    return [tuple(reversed(p.cpu().tolist())) for p in sampled_pixels]

def stratified_sampling(sam_mask, num_samples):
    
    H, W = sam_mask.shape
    ys, xs = torch.nonzero(sam_mask >= 0.5).unbind(1)

    y_min, y_max = ys.min().item(), ys.max().item()
    x_min, x_max = xs.min().item(), xs.max().item()

    cropped_mask = sam_mask[y_min:y_max+1, x_min:x_max+1]
    ch, cw = cropped_mask.shape

    grid_size = int(num_samples)
    grid_h = max(ch // grid_size, 1)
    grid_w = max(cw // grid_size, 1)

    sampled_pixels = []

    for i in range(grid_size):
        for j in range(grid_size):
            y0 = i * grid_h
            x0 = j * grid_w
            y1 = min((i + 1) * grid_h, ch)
            x1 = min((j + 1) * grid_w, cw)

            sub_mask = cropped_mask[y0:y1, x0:x1]
            valid_pixels = torch.nonzero(sub_mask >= 0.5)

            if valid_pixels.numel() > 0:
                idx = random.randint(0, valid_pixels.size(0) - 1)
                yx = valid_pixels[idx] + torch.tensor([y0, x0], device=sam_mask.device)
                orig_y, orig_x = yx[0].item() + y_min, yx[1].item() + x_min
                sampled_pixels.append((orig_x, orig_y))

    random.shuffle(sampled_pixels)
    return sampled_pixels[:num_samples]

def sample_masks(scene_data, sam_masks, config):

    for camera_id in sam_masks.keys():  

        foreground_mask = sam_masks[camera_id]['foreground']
        background_mask = sam_masks[camera_id]['background']

        if config['sampling'] == 'random':
            foreground_samples = random_sampling(foreground_mask, config['num_samples'])
            background_samples = random_sampling(background_mask, config['num_samples'])
        elif config['sampling'] == 'stratified':
            foreground_samples = stratified_sampling(foreground_mask, config['num_samples'])
            background_samples = stratified_sampling(background_mask, config['num_samples'])
        else:
            raise ValueError(f"Unknown sampling strategy: {config['sampling']}")
        
        sam_masks[camera_id]['foreground_samples'] = foreground_samples
        sam_masks[camera_id]['background_samples'] = background_samples
        sam_masks[camera_id]['selected_foreground_samples'] = []
        sam_masks[camera_id]['selected_background_samples'] = []

        if config['visualize']:
            title = f"Camera {camera_id} - Foreground sampled points suing " + config['sampling'] + " sampling"
            visualize_camera_with_mask_with_points(scene_data, int(camera_id), foreground_mask, foreground_samples, title)
            title = f"Camera {camera_id} - Background sampled points"
            visualize_camera_with_mask_with_points(scene_data, int(camera_id), background_mask, background_samples, title)

    return sam_masks

def sample_cameras(scene_data, cameras):
    return cameras[::2]