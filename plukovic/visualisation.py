import os
import cv2
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def create_camera_frustum(scale=0.1, color=[0, 1, 0]):
    points = np.array([
        [0, 0, 0], [1, 1, 2], [1, -1, 2], [-1, -1, 2], [-1, 1, 2]
    ]) * scale
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]
    ]
    colors = [color for _ in lines]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def create_sphere_at_point(center, radius=0.1, color=[1, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere.paint_uniform_color(color)
    point_3d = np.array(center)
    point_3d = point_3d.astype(np.float64)
    sphere.translate(point_3d)
    return sphere

def visualize_scene_with_trajectory(scene_data, cameras, click_coordinates_fg, click_coordinates_bg, subsample_frustrums=False):
    ply_file = scene_data.ply_file
    print(f"    Visualizing scene: {scene_data.scene_name} with PLY file: {ply_file}")

    if not os.path.exists(ply_file):
        print(f"    PLY file not found: {ply_file}")
        return
    
    pcd = o3d.io.read_point_cloud(ply_file)

    camera_positions = []
    for camera in cameras:
        camera_pose = scene_data.__get_camera_pose__(camera)
        camera_position = camera_pose[:3, 3].cpu().numpy()
        camera_positions.append(camera_position)

    lines = [[i, i + 1] for i in range(len(camera_positions) - 1)]
    trajectory_lines = o3d.geometry.LineSet()
    trajectory_lines.points = o3d.utility.Vector3dVector(camera_positions)
    trajectory_lines.lines = o3d.utility.Vector2iVector(lines)
    trajectory_lines.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])

    frustums = []
    for i in cameras:
        frustum = create_camera_frustum(scale=0.1, color=[0, 1, 0])
        frustum.transform(scene_data.__get_camera_pose__(i).cpu().numpy())
        frustums.append(frustum)
        
    if subsample_frustrums:
        frustums = frustums[::10]

    geometries = [pcd]
    for id, click in enumerate(click_coordinates_fg):

        if id == 0:
            color = [0, 0, 1]
        else:
            color = [1, 0, 0]

        sphere = create_sphere_at_point(click, radius=0.05, color=color)
        geometries.append(sphere)

    for id, click in enumerate(click_coordinates_bg):
        color = [0, 1, 0]
        sphere = create_sphere_at_point(click, radius=0.05, color=color)
        geometries.append(sphere)

    geometries += frustums

    o3d.visualization.draw_geometries(geometries)

def visualize_camera_with_point(scene_data, cam_idx, point):
    rgb = scene_data.__get_camera_rgb__(cam_idx)
    rgb_vis = rgb.cpu().numpy()
    x, y = point

    if x < 0 or x >= rgb_vis.shape[1] or y < 0 or y >= rgb_vis.shape[0]:
        print(f"    Warning: Pixel coordinates ({x}, {y}) are out of bounds for camera {cam_idx}.")
        return
    
    cv2.circle(rgb_vis, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)
    cv2.imshow(f'Camera {cam_idx} - RGB with Selected Pixel', rgb_vis)
    cv2.waitKey(0)
    cv2.destroyWindow(f'Camera {cam_idx} - RGB with Selected Pixel')

def visualize_camera_with_mask_with_point(scene_data, cam_idx, mask, point):
    rgb = scene_data.__get_camera_rgb__(cam_idx).permute(2, 0, 1)
    rgb_vis = rgb.permute(1, 2, 0).cpu().numpy()
    target_size = (812, 512)

    h, w = scene_data.__get_depth_resolution__(cam_idx)

    scale_x = target_size[0] / w
    scale_y = target_size[1] / h

    if isinstance(mask, torch.Tensor):
        mask_vis = mask.cpu().numpy().astype(np.uint8)
    else:
        mask_vis = mask.astype(np.uint8)

    mask_vis = np.stack([mask_vis] * 3, axis=-1)
    rgb_vis = cv2.resize(rgb_vis, target_size)
    mask_vis = cv2.resize(mask_vis, target_size)

    x, y = point
    x_scaled = int(x * scale_x)
    y_scaled = int(y * scale_y)

    cv2.circle(rgb_vis, (x_scaled, y_scaled), radius=3, color=(0, 255, 0), thickness=-1)
    cv2.circle(rgb_vis, (x_scaled, y_scaled), radius=3, color=(0, 255, 0), thickness=-1)
    combined = np.hstack((rgb_vis, rgb_vis * mask_vis))
    cv2.imshow(f'Camera {cam_idx} - RGB and SAM Mask', combined)
    cv2.waitKey(0)
    cv2.destroyWindow(f'Camera {cam_idx} - RGB and SAM Mask')

def visualize_iou_single(coords, preds, labels):
    """Visualize the IoU for a single object
    """
    coords = coords.cpu().numpy()
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*coords.T, c='b', label='Scene', s=1, alpha=0.1)
    ax.scatter(*coords[labels ^ preds].T, c='r', label='Mistakes', s=1.2, alpha=1.0)
    ax.legend()
    plt.show()

def visualize_gt_single(coords, labels):
    """Visualize the IoU for a single object
    """
    coords = coords.cpu().numpy()
    labels = labels.cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*coords.T, c='b', label='Scene', s=1, alpha=0.1)
    ax.scatter(*coords[labels].T, c='r', label='GT', s=1.2, alpha=1.0)
    ax.legend()
    plt.show()


def visualize_iou_scene(coords, preds, labels):
    """Visualize the IoU for all objects
    """
    obj_ids = torch.unique(labels)
    obj_ids = obj_ids[obj_ids!=0]

    for obj_id in obj_ids:
        preds_obj = preds == obj_id
        labels_obj = labels == obj_id

        visualize_iou_single(coords, preds_obj, labels_obj)

def visualize_gt_scene(coords, labels):
    """Visualize the IoU for all objects
    """
    obj_ids = torch.unique(labels)
    obj_ids = obj_ids[obj_ids!=0]

    for obj_id in obj_ids:
        labels_obj = labels == obj_id

        visualize_gt_single(coords, labels_obj)
