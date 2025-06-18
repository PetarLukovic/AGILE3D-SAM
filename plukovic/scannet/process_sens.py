import os
import sys
import h5py
import argparse
import builtins
import numpy as np

from functools import partial
from plukovic.scannet.SensorData import SensorData
from concurrent.futures import ThreadPoolExecutor

from plukovic.scannet import download_scannet

parser = argparse.ArgumentParser()
parser.add_argument('--scannet_folder', required=True, help='Path to scans folder.')
parser.add_argument('--scans_file', required=True, help='Path to scans .txt file.')
args = parser.parse_args()

# eg. python3 plukovic/scannet/process_sens.py --scannet_folder "/cluster/scratch/plukovic/scannet/scannet_v2_test" --scans_file "/cluster/scratch/plukovic/scannet/scannet_v2_test/scannetv2_val.txt"
#num_workers = os.cpu_count()
num_workers = 4

def process_scan(scan, scannet_folder):
    scans_folder = os.path.join(scannet_folder, 'scans')
    scan_path = os.path.join(scans_folder, scan)
    sens_path = os.path.join(scan_path, scan + ".sens")
    hdf5_file = os.path.join(scan_path, scan + '.hdf5')

    if os.path.exists(hdf5_file):
        print(f"    Scan {scan} already has hdf5 file at {hdf5_file}, skipping download.")
        return

    if os.path.exists(sens_path):
        print(f"    Scan {scan} already exists at {scan_path}, skipping download.")
    else:
        builtins.input = lambda prompt='': 'y'
        sys.argv = [
            'download_scannet.py',
            '-o', scannet_folder,
            '--id', scan,
            '--type', '.sens'
        ]
        download_scannet.main()
 

    print(f"    Processing {scan_path}")
    # Load .sens file
    print(f"        Loading SensorData from {sens_path}...")
    sd = SensorData(sens_path)
    print(f"        Loaded {sens_path}.")

    H, W = sd.color_height, sd.color_width
    rgb_array = np.zeros((len(sd.frames), H, W, 3), dtype=np.uint8)
    depth_array = np.zeros((len(sd.frames), sd.depth_height, sd.depth_width), dtype=np.float32)
    poses = np.zeros((len(sd.frames), 4, 4), dtype=np.float32)
    intrinsics = np.tile(sd.intrinsic_depth[:3, :3][None, :, :], (len(sd.frames), 1, 1)).astype(np.float32)

    for i, frame in enumerate(sd.frames):
        # Decompress color and depth images
        color_image = frame.decompress_color(sd.color_compression_type)
        depth_image = np.frombuffer(frame.decompress_depth(sd.depth_compression_type), dtype=np.uint16).reshape(sd.depth_height, sd.depth_width)
        
        # Store decompressed images in the arrays
        rgb_array[i] = color_image
        depth_array[i] = depth_image
        poses[i] = frame.camera_to_world

    # Save to HDF5
    print(f"        Writing to {hdf5_file}...")
    try:
        with h5py.File(hdf5_file, 'w') as f:
            f.create_dataset("scan_name", data=np.bytes_(scan))
            f.create_dataset("rgb", data=rgb_array, compression="gzip")
            f.create_dataset("depth", data=depth_array, compression="gzip")
            f.create_dataset("poses", data=poses, compression="gzip")
            f.create_dataset("intrinsics", data=intrinsics, compression="gzip")
        print(f"        Done writing {hdf5_file}")
    except Exception as e:
        print(f"        Error writing {hdf5_file}: {e}")


def main():
    with open(args.scans_file, 'r') as f:
        scans = [line.strip() for line in f.readlines()]

    print(f"    Found {len(scans)} scans listed in {args.scannet_folder}")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        func = partial(process_scan, scannet_folder=args.scannet_folder)
        executor.map(func, scans)
   
if __name__ == '__main__':
    main()
