#!/usr/bin/env python3
"""
Convert AMtown02 UAVScenes format to COLMAP format
"""
import json
import numpy as np
import struct
import os
from pathlib import Path
import shutil

def write_binary_cameras(cameras, output_path):
    """Write COLMAP binary format cameras.bin"""
    with open(output_path, 'wb') as f:
        f.write(struct.pack('Q', len(cameras)))  # Number of cameras
        
        for camera_id, camera in cameras.items():
            f.write(struct.pack('i', camera_id))  # camera_id
            f.write(struct.pack('i', camera['model']))  # model (OPENCV=4)
            f.write(struct.pack('Q', camera['width']))  # width
            f.write(struct.pack('Q', camera['height']))  # height
            
            # Parameters: fx, fy, cx, cy, k1, k2, p1, p2
            for param in camera['params']:
                f.write(struct.pack('d', param))

def write_binary_images(images, output_path):
    """Write COLMAP binary format images.bin"""
    with open(output_path, 'wb') as f:
        f.write(struct.pack('Q', len(images)))  # Number of images
        
        for image_id, image in images.items():
            f.write(struct.pack('i', image_id))  # image_id
            
            # Quaternion (qw, qx, qy, qz)
            for q in image['qvec']:
                f.write(struct.pack('d', q))
            
            # Translation vector (tx, ty, tz)
            for t in image['tvec']:
                f.write(struct.pack('d', t))
            
            f.write(struct.pack('i', image['camera_id']))  # camera_id
            
            # Image name
            name_bytes = image['name'].encode('utf-8')
            f.write(name_bytes)
            f.write(b'\x00')  # null terminator
            
            # 2D points (empty for now)
            f.write(struct.pack('Q', 0))

def write_binary_points3D(points, output_path):
    """Write COLMAP binary format points3D.bin"""
    with open(output_path, 'wb') as f:
        f.write(struct.pack('Q', len(points)))  # Number of points
        
        for point_id, point in points.items():
            f.write(struct.pack('Q', point_id))  # point3D_id
            
            # XYZ coordinates
            for coord in point['xyz']:
                f.write(struct.pack('d', coord))
            
            # RGB color
            for color in point['rgb']:
                f.write(struct.pack('B', color))
            
            # Error
            f.write(struct.pack('d', point['error']))
            
            # Track (observation list, empty for now)
            f.write(struct.pack('Q', 0))

def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion (w, x, y, z)"""
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])

def load_ply_points(ply_path, sample_rate=10):
    """Load 3D points from PLY file (with sampling to reduce point count)"""
    print(f"Loading point cloud: {ply_path}")
    points = {}
    point_id = 1
    
    with open(ply_path, 'rb') as f:
        # Skip header
        line = f.readline().decode('utf-8')
        while not line.startswith('end_header'):
            line = f.readline().decode('utf-8')
        
        # Read binary data
        # Format: float x, y, z, uchar r, g, b
        sample_count = 0
        while True:
            data = f.read(15)  # 3*4(float) + 3*1(uchar)
            if len(data) < 15:
                break
            
            sample_count += 1
            if sample_count % sample_rate != 0:  # Sampling
                continue
            
            x, y, z = struct.unpack('fff', data[:12])
            r, g, b = struct.unpack('BBB', data[12:15])
            
            points[point_id] = {
                'xyz': [x, y, z],
                'rgb': [r, g, b],
                'error': 1.0
            }
            point_id += 1
            
            if point_id % 10000 == 0:
                print(f"Loaded {point_id} points...")
    
    print(f"Complete! Total loaded {len(points)} points")
    return points

def convert_uavscenes_to_colmap(json_path, images_dir, ply_path, output_dir, sample_rate=10):
    """Main conversion function"""
    
    print("=" * 60)
    print("Converting UAVScenes -> COLMAP (AMtown02)")
    print("=" * 60)
    
    # Create output directories
    output_path = Path(output_dir)
    sparse_path = output_path / "sparse" / "0"
    sparse_path.mkdir(parents=True, exist_ok=True)
    
    images_output = output_path / "images"
    images_output.mkdir(parents=True, exist_ok=True)
    
    # Load JSON data
    print(f"\n[1/5] Reading pose data: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Found pose information for {len(data)} images")
    
    # Get camera parameters from first image
    first_sample = data[0]
    P = np.array(first_sample['P3x3'])
    fx, fy = P[0, 0], P[1, 1]
    cx, cy = P[0, 2], P[1, 2]
    k1 = first_sample.get('K1', 0.0)
    k2 = first_sample.get('K2', 0.0)
    
    # Read one image to get dimensions
    first_image_path = Path(images_dir) / first_sample['OriginalImageName']
    from PIL import Image
    img = Image.open(first_image_path)
    width, height = img.size
    
    print(f"\nCamera parameters:")
    print(f"  Resolution: {width} x {height}")
    print(f"  Focal length: fx={fx:.2f}, fy={fy:.2f}")
    print(f"  Principal point: cx={cx:.2f}, cy={cy:.2f}")
    print(f"  Distortion: k1={k1}, k2={k2}")
    
    # Create camera dictionary
    cameras = {
        1: {
            'model': 4,  # OPENCV model
            'width': width,
            'height': height,
            'params': [fx, fy, cx, cy, k1, k2, 0.0, 0.0]  # p1=0, p2=0
        }
    }
    
    # Get list of existing images
    images_dir_path = Path(images_dir)
    existing_images = {f.name for f in images_dir_path.glob("*.jpg")}
    print(f"Actual image count: {len(existing_images)}")
    
    # Create image dictionary (only process existing images)
    print(f"\n[2/5] Processing image poses")
    images = {}
    processed = 0
    for idx, sample in enumerate(data):
        image_name = sample['OriginalImageName']
        
        # Skip non-existing images
        if image_name not in existing_images:
            continue
        
        image_id = processed + 1
        processed += 1
        
        # Get 4x4 transformation matrix
        T = np.array(sample['T4x4'])
        R = T[:3, :3]  # Rotation matrix
        t = T[:3, 3]   # Translation vector
        
        # COLMAP uses camera-to-world transformation, need to invert
        R_inv = R.T
        t_inv = -R.T @ t
        
        # Convert to quaternion
        qvec = rotation_matrix_to_quaternion(R_inv)
        
        images[image_id] = {
            'qvec': qvec.tolist(),
            'tvec': t_inv.tolist(),
            'camera_id': 1,
            'name': image_name
        }
        
        if processed % 50 == 0:
            print(f"Processed {processed}/{len(existing_images)} images")
    
    # Copy images
    print(f"\n[3/5] Copying image files")
    copied = 0
    for image_name in existing_images:
        src = Path(images_dir) / image_name
        dst = images_output / image_name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    print(f"Copied {copied} images")
    
    # Load 3D points
    print(f"\n[4/5] Loading 3D point cloud (sample rate: 1/{sample_rate})")
    points = load_ply_points(ply_path, sample_rate=sample_rate)
    
    # Write COLMAP format files
    print(f"\n[5/5] Writing COLMAP binary files")
    write_binary_cameras(cameras, sparse_path / "cameras.bin")
    print("  ✓ cameras.bin")
    
    write_binary_images(images, sparse_path / "images.bin")
    print("  ✓ images.bin")
    
    write_binary_points3D(points, sparse_path / "points3D.bin")
    print("  ✓ points3D.bin")
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"  - {len(cameras)} cameras")
    print(f"  - {len(images)} images")
    print(f"  - {len(points)} 3D points")
    print()
    print("Now you can run OpenSplat!")
    print(f"  cd /root/OpenSplat/build")
    print(f"  ./opensplat {output_dir} --cpu -n 30000 -o amtown02_output.ply")

if __name__ == "__main__":
    json_path = "/root/OpenSplat/data/interval5_AMtown02/sampleinfos_interpolated.json"
    images_dir = "/root/OpenSplat/data/interval5_AMtown02/interval5_CAM"
    ply_path = "/root/OpenSplat/data/AMtown/cloud_merged.ply"
    output_dir = "/root/OpenSplat/data/AMtown02_colmap"
    
    # Sample rate: 10 means take every 10th point, reduces memory usage
    convert_uavscenes_to_colmap(json_path, images_dir, ply_path, output_dir, sample_rate=10)


