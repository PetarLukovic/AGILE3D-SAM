import cv2
import torch
import random
import numpy as np

from tqdm import tqdm

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
    return y, x

def sample_background(mask):
    background_pixels = torch.nonzero(mask <= 0.5)
    selected_index = random.randint(0, background_pixels.size(0) - 1)
    coords = background_pixels[selected_index].to(mask.device)
    y, x = coords[0], coords[1]
    return y, x

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