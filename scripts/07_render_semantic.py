#!/usr/bin/env python3
"""Render semantic-coloured Gaussian splats from the back-projection output.

For each (or a subset of) registered COLMAP view, replace each Gaussian's
DC SH term with its palette colour, rasterise via gsplat, and save a PNG
side-by-side with the original RGB image.
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
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from uavscenes_classes import PALETTE  # noqa: E402

from gsplat import rasterization  # noqa: E402


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def load_gs_ply(path: Path):
    p = PlyData.read(str(path))["vertex"].data
    N = len(p)
    xyz = np.stack([p["x"], p["y"], p["z"]], axis=1).astype(np.float32)
    scales = np.stack([p["scale_0"], p["scale_1"], p["scale_2"]], 1).astype(np.float32)
    quats = np.stack([p["rot_0"], p["rot_1"], p["rot_2"], p["rot_3"]], 1).astype(np.float32)
    opac = np.asarray(p["opacity"], dtype=np.float32)
    return xyz, np.exp(scales), quats, 1.0 / (1.0 + np.exp(-opac))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs-ply", required=True)
    ap.add_argument("--colmap-dir", default=str(REPO / "data/colmap/sparse/0"))
    ap.add_argument("--image-dir", default=str(REPO / "data/images"))
    ap.add_argument("--labels-npy", default=str(REPO / "output/semantic/semantic_labels.npy"))
    ap.add_argument("--out-dir", default=str(REPO / "output/semantic/renders"))
    ap.add_argument("--stride", type=int, default=20,
                    help="Render 1 of every N views (default: 20)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xyz, scales, quats, opacs = load_gs_ply(Path(args.gs_ply))
    label = np.load(args.labels_npy)
    if label.shape[0] != xyz.shape[0]:
        raise SystemExit(f"label count {label.shape[0]} != gaussians {xyz.shape[0]}")
    rgb = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    for c in range(len(PALETTE)):
        rgb[label == c] = PALETTE[c].astype(np.float32) / 255.0
    rgb[label == 255] = [0.5, 0.5, 0.5]
    sh0 = rgb_to_sh0(torch.from_numpy(rgb)).unsqueeze(1)  # (N, 1, 3)

    means = torch.from_numpy(xyz).to(device)
    scales_t = torch.from_numpy(scales).to(device)
    quats_t = F.normalize(torch.from_numpy(quats).to(device), dim=-1)
    opacs_t = torch.from_numpy(opacs).to(device)
    colors_t = sh0.to(device)

    rec = pycolmap.Reconstruction(args.colmap_dir)
    img_items = list(rec.images.items())
    img_items = img_items[::args.stride]
    print(f"Rendering {len(img_items)} views into {out_dir}")

    for img_id, img in tqdm(img_items, desc="render"):
        cam = rec.cameras[img.camera_id]
        if cam.model.name == "SIMPLE_RADIAL":
            f, cx, cy, _ = cam.params; fx = fy = f
        elif cam.model.name == "PINHOLE":
            fx, fy, cx, cy = cam.params
        else:
            f, cx, cy = cam.params[0], cam.width / 2, cam.height / 2; fx = fy = f
        K = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], device=device, dtype=torch.float32)
        cw = img.cam_from_world()
        viewmat = torch.eye(4, device=device).unsqueeze(0)
        viewmat[0, :3, :3] = torch.from_numpy(cw.rotation.matrix().astype(np.float32))
        viewmat[0, :3, 3] = torch.from_numpy(cw.translation.astype(np.float32))

        H, W = int(cam.height), int(cam.width)
        with torch.no_grad():
            render, _, _ = rasterization(means, quats_t, scales_t, opacs_t,
                                         colors_t, viewmat, K, W, H,
                                         sh_degree=0, packed=False)
        sem = render[0].clamp(0, 1).cpu().numpy()  # (H, W, 3) RGB float

        gt_path = Path(args.image_dir) / img.name
        gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
        if gt is not None:
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if gt.shape[:2] != sem.shape[:2]:
                gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_AREA)
            cat = np.concatenate([gt, sem], axis=1)
        else:
            cat = sem
        out_path = out_dir / f"{Path(img.name).stem}_semantic.jpg"
        cv2.imwrite(str(out_path),
                    cv2.cvtColor((cat * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Wrote {len(img_items)} pairs to {out_dir}")


if __name__ == "__main__":
    main()
