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

def sample_background(mask, border_width=50, erosion_iters=10):
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

    # --- Erode the outside_border_mask ---
    eroded = outside_border_mask.float().unsqueeze(0).unsqueeze(0)
    for _ in range(erosion_iters):
        eroded = F.conv2d(eroded, kernel, padding=1)
        eroded = (eroded == kernel.sum()).float()
    eroded = eroded.squeeze(0).squeeze(0)

    border_pixels = torch.nonzero(eroded)
    if border_pixels.size(0) == 0:
        raise ValueError("No border pixels found after erosion.")

    selected_index = random.randint(0, border_pixels.size(0) - 1)
    y, x = border_pixels[selected_index]

    return y, x, eroded

"""
def sample_background(mask):
    background_pixels = torch.nonzero(mask <= 0.5)
    selected_index = random.randint(0, background_pixels.size(0) - 1)
    coords = background_pixels[selected_index].to(mask.device)
    y, x = coords[0], coords[1]
    return y, x
"""

def extract_sam_masks_v1(scene_data, cameras, pixels, config):
    if cameras is None or len(cameras) == 0 and config['verbose']:
        print("    No cameras selected. Skipping SAM mask extraction.")
        return {}
    
    sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(config['device'])
    predictor = SamPredictor(sam)

    masks_out = {}

    for cam_id, pixel in tqdm(zip(cameras, pixels), total=len(cameras), desc="    Extracting SAM masks", disable=not config['verbose']):
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

        hg_mask = torch.tensor(masks[0], dtype=torch.uint8)
        lg_mask = torch.tensor(masks[2], dtype=torch.uint8)

        masks_out[str(cam_id.item())] = {
            "high_granularity": hg_mask,
            "low_granularity": lg_mask,
        }

        if config['visualize']:
            visualize_camera_with_mask_with_point(scene_data, cam_id, hg_mask, (x.cpu().item(), y.cpu().item()))
            visualize_camera_with_mask_with_point(scene_data, cam_id, lg_mask, (x.cpu().item(), y.cpu().item()))

    if not masks_out:
        return {}
    
    return masks_out