import cv2
import torch
import numpy as np
import open3d as o3d
from pathlib import Path
from torch.utils.data import Dataset

class KITTI360SensorData(Dataset):
    def __init__(self, exp_ply_path, transform=None, device="cpu"):

        super().__init__()
        exp_ply_path = Path(exp_ply_path)
        self.device = device
        self.transform = transform
        self.scene_name = exp_ply_path.stem
        self.base_dir = exp_ply_path.parents[3]

        self.ply_file = exp_ply_path
        self.point_cloud = o3d.io.read_point_cloud(self.ply_file)
        points = np.asarray(self.point_cloud.points, dtype=np.float32)
        self.min_values = points.min(axis=0)
        padding_m = 5
        self.boundaries = {
            "xmax": points[:, 0].max() + padding_m,
            "xmin": points[:, 0].min() - padding_m,
            "ymax": points[:, 1].max() + padding_m,
            "ymin": points[:, 1].min() - padding_m,
            "zmax": points[:, 2].max() + padding_m,
            "zmin": points[:, 2].min() - padding_m,
        }

        # --- Parse frame IDs ---
        parts = self.scene_name.split("_")
        start_fid, end_fid = int(parts[0]), int(parts[1])
        self.frame_ids = list(range(start_fid, end_fid + 1))

        # --- Load RGB images ---
        rgb_dir_00 = self.base_dir / "data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect"
        rgb_dir_01 = self.base_dir / "data_2d_raw/2013_05_28_drive_0000_sync/image_01/data_rect"
        imgs_00 = {int(p.stem): p for p in sorted(rgb_dir_00.glob("*.png"))}
        imgs_01 = {int(p.stem): p for p in sorted(rgb_dir_01.glob("*.png"))}

        # --- Load IMU -> world poses ---
        imu_to_world_file = self.base_dir / "data_poses/2013_05_28_drive_0000_sync/poses.txt"
        self.imu_poses = self.load_imu_poses(imu_to_world_file)

        # --- Load IMU -> camera extrinsics (from calibration) ---
        calib_cam_to_pose_file = self.base_dir / "calibration/calib_cam_to_pose.txt"
        T_cam0_to_imu = self.load_cam_to_imu(calib_cam_to_pose_file, "image_00")
        T_cam1_to_imu = self.load_cam_to_imu(calib_cam_to_pose_file, "image_01")

        # --- Load rectification for perspective cameras ---
        calib_file = self.base_dir / "calibration/perspective.txt"
        R_rect_00, R_rect_01, K_00, K_01 = self.load_intrinsics_and_rect(calib_file)

        # --- Build dataset samples with correct KITTI-360 transforms ---
        samples = []
        camera_count = 0

        for fid in self.frame_ids:
            if fid in self.imu_poses:

                pose_imu_world = self.imu_poses[fid]

                if fid in imgs_00:
                    pose0_world = pose_imu_world @ T_cam0_to_imu @ R_rect_00
                    pose0_world = pose0_world.astype(np.float32)
                    samples.append((camera_count, imgs_00[fid], pose0_world, K_00))
                    camera_count += 1

                if fid in imgs_01:
                    pose1_world = pose_imu_world @ T_cam1_to_imu @ R_rect_01
                    pose1_world = pose1_world.astype(np.float32)
                    samples.append((camera_count, imgs_01[fid], pose1_world, K_01))
                    camera_count += 1

        self.length = len(samples)
        if self.length > 0:
            self.rgb = [s[1] for s in samples]
            self.poses = np.stack([s[2] for s in samples], axis=0)
            self.intrinsics = np.stack([s[3] for s in samples], axis=0)
        else:
            self.rgb = []
            self.poses = np.zeros((0, 4, 4), dtype=np.float32)
            self.intrinsics = np.zeros((0, 3, 3), dtype=np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "rgb": self.__get_camera_rgb__(idx),
            "pose": self.__get_camera_pose__(idx),
            "intrinsics": self.__get_camera_intrinsics__(idx),
        }

    def __get_camera_rgb__(self, idx):
        rgb_path = self.rgb[idx]
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = torch.from_numpy(rgb).to(self.device)
        if self.transform:
            rgb = self.transform(rgb)
        return rgb

    def __get_camera_pose__(self, idx):
        pose = self.poses[idx]
        return torch.from_numpy(pose).to(self.device)

    def __get_camera_intrinsics__(self, idx):
        K = self.intrinsics[idx]
        return K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def __get_camera_resolution__(self, idx):
        rgb_path = self.rgb[idx]
        rgb = cv2.imread(str(rgb_path))
        h, w, _ = rgb.shape
        return h, w
    
    def check_pose(self, pose):
        translation = pose[:3, 3]
        if translation[0] >= self.boundaries['xmax'] or translation[0] <= self.boundaries['xmin']: return False
        if translation[1] >= self.boundaries['ymax'] or translation[1] <= self.boundaries['ymin']: return False
        if translation[2] >= self.boundaries['zmax'] or translation[2] <= self.boundaries['zmin']: return False
        return True
    
    def load_imu_poses(self, file_path):
        poses = {}
        with open(file_path, "r") as f:
            for line in f.readlines():
                vals = list(map(float, line.strip().split()))
                fid = int(vals[0])
                mat = np.array(vals[1:]).reshape(3, 4)
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :] = mat

                if self.check_pose(mat):
                    poses[fid] = pose

        return poses
    
    def load_cam_to_imu(self, file_path, cam_name):
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith(cam_name):
                    vals = list(map(float, line.strip().split()[1:]))
                    T = np.eye(4, dtype=np.float32)
                    T[:3, :4] = np.array(vals).reshape(3, 4)
                    return T
        raise ValueError(f"{cam_name} transform not found in {file_path}")

    def load_intrinsics_and_rect(self, calib_file):
        K_00, K_01 = None, None
        R_rect_00, R_rect_01 = np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)

        with open(calib_file, "r") as f:
            for line in f.readlines():
                if line.startswith("P_rect_00"):
                    vals = list(map(np.float32, line.split()[1:]))
                    P_rect_00 = np.array(vals).reshape(3, 4)
                    K_00 = P_rect_00[:, :3]  # intrinsic matrix

                elif line.startswith("P_rect_01"):
                    vals = list(map(np.float32, line.split()[1:]))
                    P_rect_01 = np.array(vals).reshape(3, 4)
                    K_01 = P_rect_01[:, :3]

                elif line.startswith("R_rect_00"):
                    vals = list(map(np.float32, line.split()[1:]))
                    R_rect_00[:3, :3] = np.array(vals).reshape(3, 3)

                elif line.startswith("R_rect_01"):
                    vals = list(map(np.float32, line.split()[1:]))
                    R_rect_01[:3, :3] = np.array(vals).reshape(3, 3)

        return R_rect_00, R_rect_01, K_00, K_01
