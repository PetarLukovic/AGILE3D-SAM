import numpy as np
import torch
import random

from interactive_tool.utils import find_nearest
from plukovic.augument_clicks import process_click

config = {
    'num_new_clicks_fg': 2,
    'num_new_clicks_bg': 2,
    'max_attempts_camera_selection': 50,
    'max_attemps_pixel_sampling': 5,
    'object_click_padding': 5,
    'verbose': True,
    'visualize': False,
    'projection_near_m': 0.01,
    'projection_far_m': 10000.0,
    'depth_threshold_mm': 100,
    'device': "cpu"
}

def get_simualted_clicks_scene_sam(raw_coords, click_idx, sensors):

    clicks_list = []
    click_times_list = []

    for click in click_idx['1']:
        
        click_coords = raw_coords[click].cpu() + sensors.min_values
        click_coords = click_coords.cpu().numpy()
    
        new_clicks_fg, new_clicks_bg, _ = process_click(sensors, click_coords, config)

        for new_click in new_clicks_fg:
            new_click = new_click - sensors.min_values
            point_idx = find_nearest(raw_coords, new_click)
            clicks_list.append({
                '1': [point_idx]
            })

        for new_click in new_clicks_bg:
            new_click = new_click - sensors.min_values
            point_idx = find_nearest(raw_coords, new_click)
            clicks_list.append({
                '0': [point_idx]
            })

    random.shuffle(clicks_list)

    for i, click in enumerate(clicks_list):
        for j, obj in enumerate(click.keys()):
            num = (i+1) * (j+1)
            click_times_list.append({
                obj: [0]
            })

    return clicks_list, click_times_list