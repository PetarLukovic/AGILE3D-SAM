import torch
import numpy as np

from tqdm import tqdm

from segment_anything import (
    sam_model_registry, 
    SamPredictor,
)


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

        # Rescale pixel coordinates from depth map to RGB image size
        rgb_height, rgb_width = scene_data.__get_camera_resolution__(cam_id)
        depth_height, depth_width = scene_data.__get_depth_resolution__(cam_id)

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