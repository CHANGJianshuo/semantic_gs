#!/usr/bin/env bash
# End-to-end semantic-GS pipeline for AMtown02.
# Each stage writes into data/ or output/; stages can be re-run independently.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# gsplat builds CUDA extensions on first import -> needs a real CUDA toolkit.
export CUDA_HOME="${CUDA_HOME:-/home/chang/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

echo "=== [0/8] Prepare working image set ==="
python3 scripts/00_prepare_data.py --stride 2 --scale 0.5

echo "=== [1/8] COLMAP SfM (camera poses) ==="
python3 scripts/01_run_colmap.py --reset

echo "=== [2/8] Train 3D Gaussian Splatting ==="
python3 scripts/02_train_gs.py --iterations 7000

echo "=== [3/8] Prepare U-Net semantic labels ==="
python3 scripts/03_prepare_labels.py --scale 0.5

echo "=== [4/8] Train U-Net ==="
python3 scripts/04_train_unet.py --epochs 30 --batch-size 4

echo "=== [5/8] U-Net inference on all frames ==="
python3 scripts/05_infer_unet.py

echo "=== [6/8] Semantic back-projection onto Gaussians ==="
python3 scripts/06_semantic_backproject.py --gs-ply output/gs/gs_final.ply

echo "=== [7/8] Render semantic splats ==="
python3 scripts/07_render_semantic.py --gs-ply output/gs/gs_final.ply

echo "=== [8/8] Figures ==="
python3 scripts/08_make_figures.py

echo "=== DONE -> output/semantic/semantic_points.ply ==="
