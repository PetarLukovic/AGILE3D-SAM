import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.ply import read_ply

class SensorData(Dataset):
    def __init__(self, hdf5_path, transform=None):
        super().__init__()
        
        self.h5_file = h5py.File(hdf5_path, 'r')  # Open and store the handle
        self.rgb = self.h5_file['rgb']                    # (N, H, W, 3), uint8   
        self.depth = self.h5_file['depth']                # (N, H, W), float32
        self.poses = self.h5_file['poses']                # (N, 4, 4), float32
        self.intrinsics = self.h5_file['intrinsics']      # (N, 3, 3), float32
        self.length = self.rgb.shape[0]

        self.scene_name = hdf5_path.split('/')[-1].split('.')[0]
        self.path = hdf5_path
        self.transform = transform

        base_dir = os.path.dirname(hdf5_path)
        # this one is agile3d ply files 
        self.ply_file_agile3d = base_dir + '.ply'
        # this one is scannet_v2 .ply colored
        self.ply_file = os.path.join(base_dir, f"{self.scene_name}_vh_clean_2.ply")

        point_cloud = read_ply(self.ply_file_agile3d)

        self.min_values = np.array([
                point_cloud['x'].min(),
                point_cloud['y'].min(),
                point_cloud['z'].min()
            ], dtype=np.float32)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):

        return {
            "rgb": self.__get_camera_rgb__(idx),
            "depth": self.__get_camera_depth__(idx),
            "pose": self.__get_camera_pose__(idx),
            "intrinsics": self.__get_camera_intrinsics__(idx),
        }

    def __get_camera_rgb__(self, idx):
        rgb = self.rgb[idx]
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  
        if self.transform:
            rgb = self.transform(rgb)
        return rgb
    
    def __get_camera_depth__(self, idx):
        depth = self.depth[idx]
        depth = torch.from_numpy(depth)
        if self.transform:
            depth = self.transform(depth)
        return depth
    
    def __get_camera_pose__(self, idx):
        pose = self.poses[idx]
        pose = torch.from_numpy(pose)
        return pose

    def __get_camera_intrinsics__(self, idx):
        intrinsics = self.intrinsics[idx]
        intrinsics = torch.from_numpy(intrinsics)
        return intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    
    def __get_camera_resolution__(self, idx):
        height, width, _ = self.rgb[idx].shape
        return height, width
    
    def __get_depth_resolution__(self, idx):
        height, width = self.depth[idx].shape
        return height, width
