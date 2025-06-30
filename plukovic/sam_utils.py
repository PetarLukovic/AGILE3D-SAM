import cv2
import torch
import random
import numpy as np

from tqdm import tqdm
import torch.nn.functional as F

from segment_anything import (
    sam_model_registry, 
    SamPredictor,
)

from plukovic.visualisation import (
    visualize_camera_with_mask_with_point,
)

def sample_foreground(mask):
    foreground_pixels = torch.nonzero(mask >= 0.5)
    selected_index = random.randint(0, foreground_pixels.size(0) - 1)
    coords = foreground_pixels[selected_index]
    y, x = coords[0], coords[1]
    return y, x, mask

def sample_background(mask, border_width=50):
    bin_mask = (mask > 0.5).float()
    
    pad = border_width
    bin_mask_padded = F.pad(bin_mask.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode='constant', value=0)
    
    kernel = torch.ones((1, 1, 3, 3), device=mask.device)
    
    # Dilate mask by border_width pixels
    dilated = bin_mask_padded
    for _ in range(border_width):
        dilated = F.conv2d(dilated, kernel, padding=1)
        dilated = (dilated > 0).float()
    dilated = dilated.squeeze(0).squeeze(0)[pad:-pad, pad:-pad]
    
    # Border outside = dilated mask minus original mask
    outside_border_mask = (dilated - bin_mask) > 0
    
    border_pixels = torch.nonzero(outside_border_mask)
    if border_pixels.size(0) == 0:
        raise ValueError("No border pixels found outside the object within the given border width.")
    
    selected_index = random.randint(0, border_pixels.size(0) - 1)
    y, x = border_pixels[selected_index]
    
    return y, x, outside_border_mask

"""
def sample_background(mask):
    background_pixels = torch.nonzero(mask <= 0.5)
    selected_index = random.randint(0, background_pixels.size(0) - 1)
    coords = background_pixels[selected_index].to(mask.device)
    y, x = coords[0], coords[1]
    return y, x
"""

def extract_sam_masks_v1(scene_data, cameras, pixels, config):
    if cameras is None or len(cameras) == 0:
        print("    No cameras selected. Skipping SAM mask extraction.")
        return {}
    
    sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(config['device'])
    predictor = SamPredictor(sam)

    masks_list = []
    sizes = []
    cam_ids = []

    for cam_id, pixel in tqdm(zip(cameras, pixels), total=len(cameras), desc="    Extracting SAM masks"):
        img = scene_data.__get_camera_rgb__(cam_id)

        x = pixel[0].to(config['device'])
        y = pixel[1].to(config['device'])

        img.to(config['device'])
        predictor.set_image(img.cpu().numpy())

        input_point = np.array([[x.cpu().item(), y.cpu().item()]], dtype=np.int32)
        input_label = np.array([1], dtype=np.int32).reshape(-1)

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        best_mask = masks[np.argmax(scores)]
        best_mask_tensor = torch.tensor(best_mask, dtype=torch.uint8)
        masks_list.append(best_mask_tensor)
        sizes.append(best_mask_tensor.sum().item())
        cam_ids.append(cam_id)

        if config['visualize']:
            visualize_camera_with_mask_with_point(scene_data, cam_id, best_mask_tensor, (x.cpu().item(), y.cpu().item()))

    if not masks_list:
        return {}

    median_size = int(np.median(sizes))
    closest_idx = min(range(len(sizes)), key=lambda i: abs(sizes[i] - median_size))

    return {str(cam_ids[closest_idx].item()): masks_list[closest_idx]}


"""
def extract_sam_masks_v1(scene_data, cameras, pixels, config):

    if cameras is None or len(cameras) == 0:
        print("    No cameras selected. Skipping SAM mask extraction.")
        return None
    
    sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(config['device'])
    predictor = SamPredictor(sam)
    all_masks = {}

    for cam_id, pixel in tqdm(zip(cameras, pixels), total=len(cameras), desc="    Extracting SAM masks"):
        img = scene_data.__get_camera_rgb__(cam_id)

        x = pixel[0].to(config['device'])
        y = pixel[1].to(config['device'])

        img.to(config['device'])
        predictor.set_image(img.cpu().numpy())

        input_point = np.array([[x, y]], dtype=np.int32)
        input_label = np.array([1], dtype=np.int32).reshape(-1)

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        best_mask = masks[np.argmax(scores)]
        best_mask = torch.tensor(best_mask, dtype=torch.uint8)

        if config['visualize']:
            visualize_camera_with_mask_with_point(scene_data, cam_id, best_mask, (x, y))
        
        all_masks[str(cam_id)] = best_mask

    return all_masks
"""