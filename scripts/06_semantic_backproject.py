#!/usr/bin/env python3
"""Semantic back-projection: aggregate U-Net per-pixel softmax across all
visible views for every 3D Gaussian and assign a class label + palette colour.

Input
    --gs-ply       3DGS PLY (must contain x,y,z; SH/scale/etc. are ignored)
    --colmap-dir   COLMAP sparse/0/ with cameras/images/points3D.bin
    --probs-dir    Directory of uint8 (H,W,C) softmax .npy files from
                   scripts/05_infer_unet.py
    --image-size   Pixel size at which COLMAP poses were estimated
                   (must match the resolution used for U-Net inference)

Output
    semantic_points.ply   XYZ + RGB per-Gaussian semantic PLY
    semantic_labels.npy   uint8 label per Gaussian
    semantic_probs.npy    (N, C) float16 aggregated probabilities
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from uavscenes_classes import CLASS_NAMES, NUM_CLASSES, PALETTE  # noqa: E402


def load_gaussian_xyz(ply_path: Path) -> np.ndarray:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    return xyz


def load_colmap(colmap_dir: Path):
    """Return: cameras dict by id, images dict by id, scaled list of
    (image_name, K(3x3), R(3x3), t(3,), W, H)."""
    rec = pycolmap.Reconstruction(str(colmap_dir))
    views = []
    for img_id, img in rec.images.items():
        cam = rec.cameras[img.camera_id]
        # intrinsics
        if cam.model.name == "SIMPLE_RADIAL":
            f, cx, cy, _k = cam.params
            fx = fy = f
        elif cam.model.name == "PINHOLE":
            fx, fy, cx, cy = cam.params
        elif cam.model.name == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params
            fx = fy = f
        else:
            # fall back to focal/principal point
            fx = fy = float(cam.params[0])
            cx, cy = float(cam.width / 2), float(cam.height / 2)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # world->cam: img.cam_from_world() returns a Rigid3d (R, t)
        cw = img.cam_from_world()
        R = cw.rotation.matrix().astype(np.float32)   # 3x3
        t = cw.translation.astype(np.float32)          # 3
        views.append((img.name, K, R, t, int(cam.width), int(cam.height)))
    return views


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs-ply", required=True, help="Trained 3DGS PLY")
    ap.add_argument("--colmap-dir", default=str(REPO / "data/colmap/sparse/0"))
    ap.add_argument("--probs-dir", default=str(REPO / "output/unet/predictions/probs"))
    ap.add_argument("--out-dir", default=str(REPO / "output/semantic"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--depth-cull", action="store_true", default=False,
                    help="(Optional) cull Gaussians far behind their max-prob view -- not used by default")
    ap.add_argument("--chunk", type=int, default=500_000,
                    help="Gaussians per chunk to limit GPU memory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading Gaussian PLY: {args.gs_ply}")
    xyz = load_gaussian_xyz(Path(args.gs_ply))
    N = xyz.shape[0]
    print(f"  {N:,} Gaussians")
    xyz_t = torch.from_numpy(xyz).to(device)

    print(f"Loading COLMAP reconstruction: {args.colmap_dir}")
    views = load_colmap(Path(args.colmap_dir))
    print(f"  {len(views)} registered views")

    probs_dir = Path(args.probs_dir)
    # we read prob maps for views whose stem matches; skip missing
    used_views = []
    for name, K, R, t, w, h in views:
        stem = Path(name).stem
        p = probs_dir / f"{stem}.npy"
        if p.exists():
            used_views.append((name, K, R, t, w, h, p))
    print(f"  {len(used_views)} views have U-Net predictions")
    if not used_views:
        raise SystemExit("No matching prediction files found.")

    # Probe one prob map to learn the prediction resolution.
    sample_arr = np.load(used_views[0][6])
    pred_h, pred_w, pred_c = sample_arr.shape
    assert pred_c == NUM_CLASSES, f"prob map has {pred_c} channels, expected {NUM_CLASSES}"
    print(f"  prediction map: {pred_h}x{pred_w}x{pred_c}")

    # Accumulator on GPU: (N, C) float32
    acc = torch.zeros((N, NUM_CLASSES), dtype=torch.float32, device=device)
    nvis = torch.zeros((N,), dtype=torch.int32, device=device)

    t0 = time.time()
    for name, K, R, t, vw, vh, ppath in tqdm(used_views, desc="back-project"):
        # Load and move prob map to GPU as (1, C, H, W) float in [0,1]
        probs_u8 = np.load(ppath)  # (H, W, C) uint8
        probs = torch.from_numpy(probs_u8).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        ph, pw = probs_u8.shape[:2]

        # scale factor from camera (vw,vh) to prediction (pw,ph)
        sx = pw / float(vw)
        sy = ph / float(vh)

        K_t = torch.from_numpy(K).to(device)
        R_t = torch.from_numpy(R).to(device)
        t_t = torch.from_numpy(t).to(device)

        for i0 in range(0, N, args.chunk):
            i1 = min(i0 + args.chunk, N)
            X = xyz_t[i0:i1]                            # (M, 3)
            # world -> cam
            Xc = X @ R_t.T + t_t                       # (M, 3)
            z = Xc[:, 2]
            in_front = z > 1e-3
            # project: K @ Xc -> (M, 3)
            uvw = Xc @ K_t.T
            u_pix = uvw[:, 0] / uvw[:, 2].clamp(min=1e-6)
            v_pix = uvw[:, 1] / uvw[:, 2].clamp(min=1e-6)
            # scale to prediction grid
            u_pred = u_pix * sx
            v_pred = v_pix * sy
            in_bounds = ((u_pred >= 0) & (u_pred < pw - 1) &
                         (v_pred >= 0) & (v_pred < ph - 1))
            visible = in_front & in_bounds
            if not visible.any():
                continue
            idx = visible.nonzero(as_tuple=True)[0]
            # grid_sample expects normalised coords in [-1, 1]
            gx = (u_pred[idx] / (pw - 1) * 2 - 1).view(1, 1, -1, 1)
            gy = (v_pred[idx] / (ph - 1) * 2 - 1).view(1, 1, -1, 1)
            grid = torch.cat([gx, gy], dim=-1)         # (1, 1, M', 2)
            sampled = F.grid_sample(probs, grid, mode="bilinear",
                                    padding_mode="border", align_corners=True)
            sampled = sampled[0, :, 0, :].T            # (M', C)
            acc[i0 + idx] += sampled
            nvis[i0 + idx] += 1

    elapsed = time.time() - t0
    print(f"Back-projection: {elapsed/60:.2f} min")

    nvis_cpu = nvis.cpu().numpy()
    acc_cpu = acc.cpu().numpy()
    # normalise (avoid div-by-zero)
    mean_probs = acc_cpu / np.maximum(nvis_cpu[:, None], 1)
    label = mean_probs.argmax(axis=1).astype(np.uint8)
    label[nvis_cpu == 0] = 255  # unseen Gaussians

    seen = (nvis_cpu > 0).sum()
    print(f"Seen Gaussians: {seen:,} / {N:,} "
          f"(mean visible views per seen = {nvis_cpu[nvis_cpu>0].mean():.1f})")
    for c in range(NUM_CLASSES):
        cnt = int((label == c).sum())
        print(f"  {c} {CLASS_NAMES[c]:<11} {cnt:>10,}  ({100*cnt/max(seen,1):.2f}%)")
    print(f"  unseen           {int((label==255).sum()):>10,}")

    np.save(out_dir / "semantic_labels.npy", label)
    np.save(out_dir / "semantic_probs.npy", mean_probs.astype(np.float16))
    np.save(out_dir / "semantic_nvis.npy", nvis_cpu.astype(np.int32))

    rgb = np.zeros((N, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        rgb[label == c] = PALETTE[c]
    rgb[label == 255] = [128, 128, 128]   # grey for unseen

    verts = np.empty(N, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(
        str(out_dir / "semantic_points.ply"))
    print(f"\nWrote {out_dir/'semantic_points.ply'} ({N:,} verts)")


if __name__ == "__main__":
    main()
