#!/usr/bin/env python3
"""
Merge LIDAR txt files to PLY point cloud for AMtown02 dataset
"""
import numpy as np
import struct
import os
from pathlib import Path
from tqdm import tqdm

def write_ply_header(f, num_points):
    """Write PLY file header"""
    header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    f.write(header.encode('ascii'))

def merge_lidar_to_ply(lidar_dir, output_ply, sample_rate=1):
    """
    Merge all LIDAR txt files into a single PLY file
    
    Args:
        lidar_dir: Directory containing LIDAR txt files
        output_ply: Output PLY file path
        sample_rate: Sample every Nth point (1 = all points, 10 = every 10th point)
    """
    print("=" * 60)
    print("Merging LIDAR txt files to PLY")
    print("=" * 60)
    
    lidar_path = Path(lidar_dir)
    txt_files = sorted(lidar_path.glob("*.txt"))
    
    print(f"\nFound {len(txt_files)} LIDAR txt files")
    print(f"Sample rate: 1/{sample_rate}")
    
    # First pass: count total points after sampling
    print("\n[1/2] Counting points...")
    total_points = 0
    for txt_file in tqdm(txt_files, desc="Counting"):
        with open(txt_file, 'r') as f:
            num_lines = sum(1 for _ in f)
            total_points += (num_lines + sample_rate - 1) // sample_rate
    
    print(f"Total points after sampling: {total_points:,}")
    
    # Second pass: read points and write to PLY
    print("\n[2/2] Writing PLY file...")
    
    with open(output_ply, 'wb') as f:
        write_ply_header(f, total_points)
        
        points_written = 0
        for txt_file in tqdm(txt_files, desc="Processing"):
            with open(txt_file, 'r') as txt_f:
                for line_idx, line in enumerate(txt_f):
                    # Sample points
                    if line_idx % sample_rate != 0:
                        continue
                    
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        
                        # Default gray color (no color info in LIDAR)
                        r, g, b = 128, 128, 128
                        
                        # Write binary data: 3 floats + 3 uchars
                        f.write(struct.pack('fff', x, y, z))
                        f.write(struct.pack('BBB', r, g, b))
                        
                        points_written += 1
                    except (ValueError, IndexError):
                        continue
    
    print(f"\n✓ Successfully wrote {points_written:,} points to {output_ply}")
    print(f"File size: {os.path.getsize(output_ply) / (1024**2):.2f} MB")
    
    print("\n" + "=" * 60)
    print("Merge completed!")
    print("=" * 60)

if __name__ == "__main__":
    lidar_dir = "/root/OpenSplat/data/interval5_AMtown02/interval5_LIDAR"
    output_ply = "/root/OpenSplat/data/interval5_AMtown02/cloud_merged.ply"
    
    # Sample rate: 10 means take every 10th point
    # This reduces 34M points to ~3.4M points
    sample_rate = 10
    
    merge_lidar_to_ply(lidar_dir, output_ply, sample_rate=sample_rate)


