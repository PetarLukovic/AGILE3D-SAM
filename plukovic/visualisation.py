import os
import cv2
import torch
import numpy as np
import open3d as o3d


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

def visualize_scene_with_trajectory(scene_data, cameras, click_coordinates, subsample_frustrums=False):

    ply_file = scene_data.ply_file
    print(f"    Visualizing scene: {scene_data.scene_name} with PLY file: {ply_file}")

    if not os.path.exists(ply_file):
        print(f"    PLY file not found: {ply_file}")
        return
    
    pcd = o3d.io.read_point_cloud(ply_file)

    camera_positions = []
    for camera in cameras:
        camera_pose = scene_data.__get_camera_pose__(camera)
        camera_position = camera_pose[:3, 3].numpy()
        camera_positions.append(camera_position)

    lines = [[i, i + 1] for i in range(len(camera_positions) - 1)]
    trajectory_lines = o3d.geometry.LineSet()
    trajectory_lines.points = o3d.utility.Vector3dVector(camera_positions)
    trajectory_lines.lines = o3d.utility.Vector2iVector(lines)
    trajectory_lines.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])

    frustums = []
    for i in cameras:
        frustum = create_camera_frustum(scale=0.1, color=[0, 1, 0])
        frustum.transform(scene_data.__get_camera_pose__(i))
        frustums.append(frustum)
        
    if subsample_frustrums:
        frustums = frustums[::10]

    geometries = [pcd]
    for id, click in enumerate(click_coordinates):

        if id == 0:
            color = [0, 0, 1]
        else:
            color = [1, 0, 0]

        click = np.array(click.numpy()).flatten() if isinstance(click, torch.Tensor) else np.array(click)
        sphere = create_sphere_at_point(click, radius=0.05, color=color)
        geometries.append(sphere)

    geometries += frustums

    o3d.visualization.draw_geometries(geometries)