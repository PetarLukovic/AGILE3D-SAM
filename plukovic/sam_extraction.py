import torch
import numpy as np

from tqdm import tqdm
import torch.nn.functional as F

from segment_anything import (
    sam_model_registry, 
    SamPredictor,
)

from plukovic.visualisation import (
    visualize_camera_with_mask_with_points,
)

def extract_foreground_mask(mask, opening_iters=7, erosion_iters=4):

    if mask.dim() == 4 and mask.size(0) == 1 and mask.size(1) == 1:
        mask = mask.squeeze(0).squeeze(0)

    bin_mask = (mask > 0.5).float().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, 3, 3), device=mask.device)

    eroded = bin_mask.clone()
    for _ in range(opening_iters):
        eroded = F.conv2d(eroded, kernel, padding=1)
        eroded = (eroded == kernel.sum()).float()

    dilated = eroded.clone()
    for _ in range(erosion_iters):
        dilated = F.conv2d(dilated, kernel, padding=1)
        dilated = (dilated > 0).float()

    return dilated.squeeze(0).squeeze(0)

def extract_background_mask(mask, opening_iters=50, erosion_iters=10):
    if mask.dim() == 4 and mask.size(0) == 1 and mask.size(1) == 1:
        mask = mask.squeeze(0).squeeze(0)

    bin_mask = (mask > 0.5).float()
    kernel = torch.ones((1, 1, 3, 3), device=mask.device)

    pad = opening_iters
    padded_mask = F.pad(bin_mask.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode='constant', value=0)

    dilated = padded_mask
    for _ in range(opening_iters):
        dilated = F.conv2d(dilated, kernel, padding=1)
        dilated = (dilated > 0).float()
    dilated = dilated.squeeze(0).squeeze(0)[pad:-pad, pad:-pad]

    outside_border_mask = (dilated - bin_mask) > 0
    eroded = outside_border_mask.float().unsqueeze(0).unsqueeze(0)
    for _ in range(erosion_iters):
        eroded = F.conv2d(eroded, kernel, padding=1)
        eroded = (eroded == kernel.sum()).float()

    return eroded.squeeze(0).squeeze(0)

def extract_sam_masks(scene_data, cameras, pixels, config):
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

        foreground = extract_foreground_mask(torch.tensor(masks[0], dtype=torch.uint8))
        background = extract_background_mask(torch.tensor(masks[2], dtype=torch.uint8))

        masks_out[str(cam_id.item())] = {
            "foreground": foreground,
            "background": background,
        }

        if config['visualize']:
            visualize_camera_with_mask_with_points(scene_data, cam_id, foreground, [(x.cpu().item(), y.cpu().item())], f"Camera {cam_id} - Foreground Mask")
            visualize_camera_with_mask_with_points(scene_data, cam_id, background, [(x.cpu().item(), y.cpu().item())], f"Camera {cam_id} - Background Mask")

    if not masks_out:
        return {}
    
    return masks_out