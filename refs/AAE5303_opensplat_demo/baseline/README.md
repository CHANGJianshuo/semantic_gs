# 🎯 Baseline Results - AMtown02 Dataset

## Overview

This directory contains the **baseline implementation** and results for the AAE5303 3D Gaussian Splatting assignment using the **AMtown02** sequence from the UAVScenes dataset.

The baseline serves as a reference implementation to help students understand the complete workflow and provides benchmark metrics for comparison.

---

## 📊 Baseline Performance Metrics

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Dataset** | AMtown02 | Urban aerial sequence from UAVScenes |
| **Number of Images** | 1,380 | Total training images |
| **Image Resolution** | 2448 × 2048 | Original resolution |
| **Downscale Factor** | 4× | Applied to reduce memory usage |
| **Training Resolution** | 612 × 512 | Actual training resolution |
| **Initial 3D Points** | 8,335,917 | From merged LiDAR point cloud |
| **Training Iterations** | 300 | Limited due to computational constraints |
| **Training Device** | CPU | No GPU acceleration |
| **Training Time** | ~25 minutes | On CPU-only system |

### Loss Statistics

| Metric | Value |
|--------|-------|
| **Initial Loss** | 0.2164 |
| **Final Loss** | 0.0888 |
| **Minimum Loss** | 0.0454 |
| **Maximum Loss** | 0.2147 |
| **Mean Loss** | 0.1334 |
| **Std Deviation** | 0.0346 |
| **Loss Reduction** | 58.9% |

### Output Model

| Property | Value |
|----------|-------|
| **File Size** | 2.0 GB |
| **Format** | PLY (Stanford Polygon File) |
| **Number of Gaussians** | ~8.3 million |
| **Spherical Harmonics Degree** | 3 |

---

## 🔧 Implementation Details

### 1. Data Preparation

#### Point Cloud Generation

The AMtown02 dataset provides LiDAR point clouds in text format. These were merged into a single PLY file:

```bash
python3 merge_lidar_to_ply.py
```

**Process:**
- **Input**: 1,380 LiDAR text files (34.4M points total)
- **Sampling**: 1/10 rate (every 10th point)
- **Output**: `cloud_merged.ply` (3.4M points, 49 MB)

#### COLMAP Conversion

Convert UAVScenes format to COLMAP-compatible format:

```bash
python3 convert_amtown02_to_colmap.py
```

**Output Structure:**
```
AMtown02_colmap/
├── images/                    # 1,380 images
└── sparse/0/
    ├── cameras.bin           # Camera parameters
    ├── images.bin            # Image poses
    └── points3D.bin          # 3D points (343,815 after sampling)
```

### 2. Training Command

```bash
cd /root/OpenSplat/build
./opensplat /root/OpenSplat/data/AMtown02_colmap \
    --cpu \
    -n 300 \
    -d 4 \
    -o amtown02_output_300.ply
```

**Parameter Breakdown:**

- `--cpu`: Force CPU execution (no GPU required)
- `-n 300`: Number of training iterations
- `-d 4`: Downscale images by 4× (memory optimization)
- `-o amtown02_output_300.ply`: Output file path

**Default Hyperparameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `sh-degree` | 3 | Maximum spherical harmonics degree |
| `ssim-weight` | 0.2 | SSIM loss weight |
| `refine-every` | 100 | Densification interval (steps) |
| `warmup-length` | 500 | Warmup before densification |
| `num-downscales` | 2 | Resolution progression stages |
| `resolution-schedule` | 3000 | Steps between resolution increases |

### 3. Key Differences from HKisland Demo

| Aspect | HKisland | AMtown02 |
|--------|----------|----------|
| **Images** | 534 | 1,380 (2.6× more) |
| **Point Source** | Pre-processed PLY | LiDAR merge |
| **Initial Points** | 1.4M | 8.3M (6× more) |
| **Downscale Factor** | 1× | 4× (required) |
| **Training Time** | ~50 min | ~25 min |
| **Output Size** | 341 MB | 2.0 GB (6× larger) |

---

## 💡 Important Notes for Students

### Memory Considerations

⚠️ **Critical**: The AMtown02 dataset requires careful memory management due to:

1. **Large Image Count**: 1,380 images vs. 534 in HKisland
2. **High-Resolution Images**: 2448 × 2048 pixels each
3. **Large Point Cloud**: 8.3M initial Gaussians

**Memory Usage Without Downscaling:**
- Image loading: ~30GB RAM
- Gaussian storage: ~8GB RAM
- **Total**: 38-40GB RAM (exceeds typical system limits)

**Solution**: Use downscale factor `-d 4` to reduce memory to ~8-10GB.

### Training Iterations

The baseline uses only **300 iterations** due to computational constraints. For better quality:

| Iterations | Expected Quality | Training Time (CPU) | Recommended For |
|------------|------------------|---------------------|-----------------|
| 300 | Basic | ~25 min | Quick test / baseline |
| 1,000 | Acceptable | ~1.5 hours | Development |
| 3,000 | Good | ~4 hours | Submission |
| 7,000+ | High | ~10+ hours | Competition |

### Optimization Strategies

Students are encouraged to improve upon the baseline by:

1. **GPU Acceleration**: 50-100× speedup over CPU
2. **More Iterations**: Train for 3,000-30,000 steps
3. **Hyperparameter Tuning**:
   - Adjust `densify-grad-thresh` for finer detail
   - Tune `ssim-weight` for better perceptual quality
   - Modify `refine-every` interval
4. **Data Quality**: Use higher-quality point cloud initialization
5. **Resolution Strategy**: Start with lower downscale factor if memory permits

---

## 📈 Training Progress Analysis

### Loss Curve Characteristics

The training exhibits several phases:

**Phase 1 (Steps 1-100): Rapid Convergence**
- Loss drops from 0.216 to ~0.15
- Fast initial optimization of Gaussian parameters
- High variability due to random camera sampling

**Phase 2 (Steps 101-200): Stabilization**
- Loss oscillates around 0.12-0.14
- Gaussians adapting to scene geometry
- Occasional spikes from difficult viewpoints

**Phase 3 (Steps 201-300): Refinement**
- Loss converges toward 0.09-0.11
- Fine-tuning of colors and opacity
- More stable with reduced variance

### Loss Statistics by Phase

| Phase | Steps | Mean Loss | Min Loss | Max Loss | Std Dev |
|-------|-------|-----------|----------|----------|---------|
| Phase 1 | 1-100 | 0.1589 | 0.0454 | 0.2164 | 0.0398 |
| Phase 2 | 101-200 | 0.1298 | 0.0650 | 0.1952 | 0.0321 |
| Phase 3 | 201-300 | 0.1115 | 0.0467 | 0.2147 | 0.0356 |

---

## 🎓 Learning Objectives

This baseline implementation demonstrates:

1. ✅ **Data Pipeline**: Converting UAVScenes format to COLMAP
2. ✅ **Point Cloud Processing**: Merging and sampling LiDAR data
3. ✅ **Memory Management**: Handling large-scale datasets
4. ✅ **Training Workflow**: Running OpenSplat with appropriate parameters
5. ✅ **Analysis**: Understanding loss curves and convergence

---

## 📁 Files in This Directory

```
baseline/
├── README.md                          # This file
├── training_log.txt                   # Complete training output
├── training_config.json               # Configuration parameters
├── scripts/
│   ├── merge_lidar_to_ply.py         # LiDAR merging script
│   └── convert_amtown02_to_colmap.py # Format conversion script
└── docs/
    ├── MEMORY_OPTIMIZATION.md         # Memory management guide
    └── HYPERPARAMETER_GUIDE.md        # Tuning recommendations
```

---

## 🚀 Quick Start for Students

### Step 1: Prepare Data

```bash
# Merge LiDAR point clouds
cd /root/OpenSplat
python3 merge_lidar_to_ply.py

# Convert to COLMAP format
python3 convert_amtown02_to_colmap.py
```

### Step 2: Train Model

```bash
# Basic training (300 iterations)
cd /root/OpenSplat/build
./opensplat /root/OpenSplat/data/AMtown02_colmap \
    --cpu -n 300 -d 4 -o amtown02_output.ply

# Extended training (3000 iterations, recommended)
./opensplat /root/OpenSplat/data/AMtown02_colmap \
    --cpu -n 3000 -d 4 -o amtown02_output_3k.ply
```

### Step 3: Evaluate Results

See [../leaderboard/README.md](../leaderboard/README.md) for evaluation metrics and submission instructions.

---

## 🏆 Beating the Baseline

To achieve better results than this baseline:

### Essential Improvements

1. **Increase Training Iterations**: 300 → 3,000+ iterations
   - Expected PSNR improvement: +2-5 dB
   - Training time: ~4 hours on CPU

2. **GPU Acceleration**: If available
   - 50-100× faster training
   - Enables 30,000 iteration training in reasonable time

### Advanced Improvements

3. **Point Cloud Quality**:
   - Reduce sampling rate from 1/10 to 1/5 or 1/2
   - Use more accurate initial geometry

4. **Hyperparameter Optimization**:
   ```bash
   # Example: Aggressive densification
   ./opensplat ... \
       --densify-grad-thresh 0.0001 \
       --densify-size-thresh 0.005 \
       --refine-every 50
   ```

5. **Multi-Resolution Strategy**:
   - Start with `-d 4`, switch to `-d 2` mid-training
   - Requires manual intervention and resuming

---

## ❓ Common Issues and Solutions

### Issue 1: Out of Memory (OOM)

**Symptom**: Process killed with exit code 137

**Solution**:
- Increase downscale factor: `-d 4` → `-d 8`
- Reduce number of images (subsample dataset)
- Close other applications to free RAM

### Issue 2: Training Too Slow

**Symptom**: <1 iteration per minute

**Solution**:
- This is expected on CPU
- Consider GPU setup or reduce iterations
- Use smaller image resolution

### Issue 3: Poor Rendering Quality

**Symptom**: Blurry or incomplete reconstruction

**Solution**:
- Train for more iterations (3,000-30,000)
- Check point cloud quality
- Verify camera poses are correct

---

## 📚 Additional Resources

- **OpenSplat GitHub**: https://github.com/pierotofy/OpenSplat
- **UAVScenes Dataset**: https://github.com/sijieaaa/UAVScenes
- **3DGS Paper**: [Kerbl et al., SIGGRAPH 2023]
- **COLMAP Documentation**: https://colmap.github.io/

---

## 📝 Changelog

- **2024-12-25**: Initial baseline release
  - 300 iterations training
  - CPU-only execution
  - Memory-optimized with 4× downscaling

---

<div align="center">

**AAE5303 - Robust Control Technology in Low-Altitude Aerial Vehicle**

*Department of Aeronautical and Aviation Engineering*

*The Hong Kong Polytechnic University*

December 2024

</div>

