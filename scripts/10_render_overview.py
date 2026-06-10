#!/usr/bin/env python3
"""Render a single high-resolution top-down overview of the semantic
Gaussian splat -- treats the whole AMtown02 reconstruction as one map.

Strategy
    1. Estimate the "ground plane" up direction as the average camera up axis
       across all COLMAP poses (UAV downward-looking camera -> avg cam-z in
       world is roughly the down direction).
    2. Centre the camera at the centroid of all Gaussians (xy) and lift it
       far above along the up direction.
    3. Pick a focal length so the in-plane bbox just fills the image.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from plyfile import PlyData

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from uavscenes_classes import PALETTE  # noqa: E402

from gsplat import rasterization  # noqa: E402


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / 0.28209479177387814


def load_gs(path: Path):
    p = PlyData.read(str(path))["vertex"].data
    xyz = np.stack([p["x"], p["y"], p["z"]], 1).astype(np.float32)
    sc = np.stack([p["scale_0"], p["scale_1"], p["scale_2"]], 1).astype(np.float32)
    qu = np.stack([p["rot_0"], p["rot_1"], p["rot_2"], p["rot_3"]], 1).astype(np.float32)
    op = np.asarray(p["opacity"], dtype=np.float32)
    return xyz, np.exp(sc), qu, 1.0 / (1.0 + np.exp(-op))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs-ply", default=str(REPO / "output/gs/gs_final.ply"))
    ap.add_argument("--labels", default=str(REPO / "output/semantic/semantic_labels.npy"))
    ap.add_argument("--colmap-dir", default=str(REPO / "data/colmap/sparse/0"))
    ap.add_argument("--out", default=str(REPO / "output/semantic/overview.jpg"))
    ap.add_argument("--width", type=int, default=2400)
    ap.add_argument("--height", type=int, default=2400)
    ap.add_argument("--altitude-scale", type=float, default=1.4,
                    help="Multiplier on bbox-diagonal for the camera altitude.")
    args = ap.parse_args()

    device = torch.device("cuda")

    xyz, scales, quats, opacs = load_gs(Path(args.gs_ply))
    label = np.load(args.labels)
    N = xyz.shape[0]
    print(f"Loaded {N:,} Gaussians")

    # ---- per-Gaussian colour from semantic palette -------------------------
    rgb = np.full((N, 3), 0.5, dtype=np.float32)
    for c in range(len(PALETTE)):
        rgb[label == c] = PALETTE[c].astype(np.float32) / 255.0
    sh0 = rgb_to_sh0(torch.from_numpy(rgb)).unsqueeze(1).to(device)

    means = torch.from_numpy(xyz).to(device)
    scales_t = torch.from_numpy(scales).to(device)
    quats_t = F.normalize(torch.from_numpy(quats).to(device), dim=-1)
    opacs_t = torch.from_numpy(opacs).to(device)

    # ---- camera pose: top-down on the COLMAP trajectory --------------------
    rec = pycolmap.Reconstruction(args.colmap_dir)
    cam_pos = []
    cam_down = []  # world-space direction the camera "looks" along (cam +Z)
    for img in rec.images.values():
        cw = img.cam_from_world()
        R = cw.rotation.matrix()
        t = cw.translation
        # camera centre in world: -R^T t
        c = -R.T @ t
        cam_pos.append(c)
        cam_down.append(R[2, :])  # row 2 of R is the look direction in world
    cam_pos = np.stack(cam_pos)
    cam_down = np.stack(cam_down).mean(0)
    cam_down /= (np.linalg.norm(cam_down) + 1e-9)
    up = -cam_down  # for a downward UAV, "up" in world = -look-direction

    centroid = cam_pos.mean(0)
    # use camera positions to pick frame, not Gaussian bbox (avoids outliers)
    rel = cam_pos - centroid
    span = float(np.percentile(np.linalg.norm(rel, axis=1), 95)) * 2.0
    altitude = span * args.altitude_scale

    # virtual camera centre: above centroid along -look (= +up)
    cam_centre = centroid + up * altitude
    # build look-at: looking back down (cam +Z = -up)
    look = -up
    # pick right axis orthogonal to up (use world +x projected if not parallel)
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    right = np.cross(look, ref); right /= np.linalg.norm(right) + 1e-9
    cam_up = np.cross(right, look); cam_up /= np.linalg.norm(cam_up) + 1e-9

    R = np.stack([right, -cam_up, look], axis=0).astype(np.float32)  # 3x3 world->cam
    t = (-R @ cam_centre).astype(np.float32)
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t

    # ---- intrinsics: pick focal so 95th percentile bbox just fits ---------
    rel_in_cam = (R @ rel.T).T  # (V, 3)
    px = np.percentile(np.abs(rel_in_cam[:, 0]), 95)
    py = np.percentile(np.abs(rel_in_cam[:, 1]), 95)
    margin = 1.15
    fx = args.width * 0.5 * altitude / (px * margin)
    fy = args.height * 0.5 * altitude / (py * margin)
    f = float(min(fx, fy))
    K = np.array([[f, 0, args.width / 2], [0, f, args.height / 2], [0, 0, 1]], dtype=np.float32)

    K_t = torch.from_numpy(K).unsqueeze(0).to(device)
    VM_t = torch.from_numpy(viewmat).unsqueeze(0).to(device)

    print(f"camera centre = {cam_centre}, altitude = {altitude:.2f}, f = {f:.1f}")

    with torch.no_grad():
        render, _, _ = rasterization(
            means, quats_t, scales_t, opacs_t, sh0,
            VM_t, K_t, args.width, args.height,
            sh_degree=0, packed=False,
        )
    img = render[0].clamp(0, 1).cpu().numpy()  # (H, W, 3) RGB
    out_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Wrote {args.out} ({args.width}x{args.height})")


if __name__ == "__main__":
    main()
