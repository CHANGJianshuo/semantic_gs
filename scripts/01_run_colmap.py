#!/usr/bin/env python3
"""Run COLMAP SfM (pycolmap, CPU SIFT) over the working image set.

Pipeline:
    extract_features    -> SIFT keypoints into data/colmap/database.db
    match_sequential    -> sequential matching with loop closure (overlap N)
    incremental_mapping -> SfM reconstruction into data/colmap/sparse/

Aerial video is sequential, so sequential matching is the right choice
(O(n*overlap) pairs vs O(n^2) for exhaustive).  PyPI pycolmap is CPU-only,
so we keep feature counts moderate.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import pycolmap

IMAGES_DIR = Path("/home/chang/semantic_gs/data/images")
WORK_DIR = Path("/home/chang/semantic_gs/data/colmap")
DB_PATH = WORK_DIR / "database.db"
SPARSE_DIR = WORK_DIR / "sparse"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-features", type=int, default=4096,
                    help="Max SIFT features per image (CPU). Default 4096.")
    ap.add_argument("--overlap", type=int, default=15,
                    help="Sequential matching overlap window. Default 15.")
    ap.add_argument("--quad-overlap", action="store_true", default=True,
                    help="Quadratic overlap for loop closure (default on).")
    ap.add_argument("--single-camera", action="store_true", default=True,
                    help="All frames share intrinsics (aerial = same camera).")
    ap.add_argument("--reset", action="store_true",
                    help="Delete existing database and sparse dir before running.")
    args = ap.parse_args()

    if args.reset and WORK_DIR.exists():
        print(f"Resetting {WORK_DIR}")
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"Found {len(images)} images in {IMAGES_DIR}")

    # ---------------- 1. Feature extraction -----------------
    print("\n[1/3] Extracting SIFT features (CPU)...")
    t0 = time.time()
    extract_opts = pycolmap.FeatureExtractionOptions()
    extract_opts.use_gpu = False
    extract_opts.sift.max_num_features = args.max_features
    extract_opts.sift.estimate_affine_shape = False
    extract_opts.sift.domain_size_pooling = False

    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = "SIMPLE_RADIAL"   # f, cx, cy + 1 distortion

    camera_mode = pycolmap.CameraMode.SINGLE if args.single_camera else pycolmap.CameraMode.AUTO

    pycolmap.extract_features(
        database_path=DB_PATH,
        image_path=IMAGES_DIR,
        camera_mode=camera_mode,
        reader_options=reader_opts,
        extraction_options=extract_opts,
        device=pycolmap.Device.cpu,
    )
    print(f"  done in {time.time()-t0:.1f}s")

    # ---------------- 2. Sequential matching -----------------
    print(f"\n[2/3] Sequential matching (overlap={args.overlap}, "
          f"quad={args.quad_overlap})...")
    t0 = time.time()
    match_opts = pycolmap.FeatureMatchingOptions()
    pair_opts = pycolmap.SequentialPairingOptions()
    pair_opts.overlap = args.overlap
    pair_opts.quadratic_overlap = args.quad_overlap
    pair_opts.loop_detection = False  # vocab tree not available -> skip
    pycolmap.match_sequential(
        database_path=DB_PATH,
        matching_options=match_opts,
        pairing_options=pair_opts,
    )
    print(f"  done in {time.time()-t0:.1f}s")

    # ---------------- 3. Incremental mapping -----------------
    print("\n[3/3] Incremental mapping (SfM)...")
    t0 = time.time()
    map_opts = pycolmap.IncrementalPipelineOptions()
    map_opts.ba_use_gpu = False                  # pycolmap PyPI = CPU BA
    map_opts.extract_colors = True               # populates point3D RGB
    map_opts.min_num_matches = 15
    map_opts.multiple_models = False             # keep one connected model
    recs = pycolmap.incremental_mapping(
        database_path=DB_PATH,
        image_path=IMAGES_DIR,
        output_path=SPARSE_DIR,
        options=map_opts,
    )
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"\nReconstructions: {len(recs)}")
    for k, rec in recs.items():
        print(f"  model {k}: {rec.num_images()} imgs, "
              f"{rec.num_points3D()} 3D points, "
              f"{rec.num_reg_images()} registered")


if __name__ == "__main__":
    main()
