#!/usr/bin/env python3
"""Minimal 3D Gaussian Splatting trainer (gsplat backend).

Trains from a COLMAP sparse model + images and exports a PLY of Gaussian
parameters compatible with the INRIA gaussian-splatting PLY layout
(positions, normals=0, f_dc, f_rest=0, opacity, scale, rotation).

This is intentionally minimal: spherical-harmonics fixed at degree 0
(constant per-Gaussian colour), single resolution, periodic densification
via gsplat's DefaultStrategy.  Geometry accuracy is not the goal of this
project (the goal is semantic labelling); we just need a dense Gaussian
cloud anchored on the COLMAP poses.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from gsplat import DefaultStrategy, rasterization  # noqa: E402
from plyfile import PlyData, PlyElement  # noqa: E402


def load_colmap(colmap_dir: Path, image_dir: Path):
    rec = pycolmap.Reconstruction(str(colmap_dir))

    # Per-view data: name, K(3,3), viewmat(4,4), W, H, image (uint8 RGB on CPU)
    views = []
    for img_id, img in rec.images.items():
        cam = rec.cameras[img.camera_id]
        if cam.model.name == "SIMPLE_RADIAL":
            f, cx, cy, _ = cam.params
            fx = fy = f
        elif cam.model.name == "PINHOLE":
            fx, fy, cx, cy = cam.params
        elif cam.model.name == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params
            fx = fy = f
        else:
            fx = fy = float(cam.params[0])
            cx, cy = cam.width / 2.0, cam.height / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        cw = img.cam_from_world()
        R = cw.rotation.matrix().astype(np.float32)
        t = cw.translation.astype(np.float32)
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t

        ipath = image_dir / img.name
        bgr = cv2.imread(str(ipath))
        if bgr is None:
            print(f"WARN: missing image {ipath}, skipping view")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[1] != int(cam.width) or rgb.shape[0] != int(cam.height):
            rgb = cv2.resize(rgb, (int(cam.width), int(cam.height)),
                             interpolation=cv2.INTER_AREA)

        views.append({"name": img.name, "K": K, "viewmat": viewmat,
                      "W": int(cam.width), "H": int(cam.height),
                      "rgb": rgb})

    # Initial point cloud from COLMAP
    pts, cols = [], []
    for pid, p in rec.points3D.items():
        pts.append(p.xyz)
        cols.append(p.color)
    if not pts:
        raise SystemExit("COLMAP reconstruction has no 3D points.")
    pts = np.asarray(pts, dtype=np.float32)
    cols = np.asarray(cols, dtype=np.float32) / 255.0
    print(f"Loaded {len(views)} views, {len(pts):,} init points from COLMAP.")
    return views, pts, cols


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    """SH degree-0 DC term that, after gsplat's SH eval and +0.5, recovers RGB.
    sh0 = (rgb - 0.5) / C0   with  C0 = 0.28209479177387814 (=1/(2*sqrt(pi)))"""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap-dir", default=str(REPO / "data/colmap/sparse/0"))
    ap.add_argument("--image-dir", default=str(REPO / "data/images"))
    ap.add_argument("--out-dir", default=str(REPO / "output/gs"))
    ap.add_argument("--iterations", type=int, default=7000)
    ap.add_argument("--save-iters", nargs="+", type=int, default=[3000, 7000])
    ap.add_argument("--init-scale", type=float, default=0.005)
    ap.add_argument("--init-opacity", type=float, default=0.1)
    ap.add_argument("--sh-degree", type=int, default=0,
                    help="Keep at 0 for speed/simplicity; semantic stage only uses positions.")
    ap.add_argument("--ssim-lambda", type=float, default=0.2)
    ap.add_argument("--densify-start", type=int, default=500)
    ap.add_argument("--densify-stop-frac", type=float, default=0.5,
                    help="Stop densification after this fraction of iterations.")
    ap.add_argument("--densify-interval", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    views, init_xyz, init_rgb = load_colmap(Path(args.colmap_dir), Path(args.image_dir))
    N = init_xyz.shape[0]

    # ---- initialise Gaussians ----
    means = torch.tensor(init_xyz, device=device, dtype=torch.float32)
    rgb = torch.tensor(init_rgb, device=device, dtype=torch.float32)
    sh0 = rgb_to_sh0(rgb).unsqueeze(1)   # (N, 1, 3)
    shN = torch.zeros((N, (args.sh_degree + 1) ** 2 - 1, 3),
                      device=device, dtype=torch.float32) if args.sh_degree > 0 else \
          torch.zeros((N, 0, 3), device=device, dtype=torch.float32)

    # initial scale heuristic: use median nearest-neighbour distance
    with torch.no_grad():
        idx = torch.randperm(N, device=device)[:min(N, 50_000)]
        sample = means[idx]
        d = torch.cdist(sample, sample)
        d.fill_diagonal_(float("inf"))
        nn = d.min(dim=1).values
        med_nn = nn.median().item()
    init_scale = max(args.init_scale, 0.5 * med_nn)
    print(f"init scale = {init_scale:.4f} (median NN = {med_nn:.4f})")

    scales = torch.log(torch.full((N, 3), init_scale, device=device))     # log-space
    quats = torch.zeros((N, 4), device=device); quats[:, 0] = 1.0          # w,x,y,z
    opacs = torch.logit(torch.full((N,), args.init_opacity, device=device))  # inv-sigmoid

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacs),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    }).to(device)

    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=1.6e-4),
        "scales": torch.optim.Adam([params["scales"]], lr=5e-3),
        "quats": torch.optim.Adam([params["quats"]], lr=1e-3),
        "opacities": torch.optim.Adam([params["opacities"]], lr=5e-2),
        "sh0": torch.optim.Adam([params["sh0"]], lr=2.5e-3),
        "shN": torch.optim.Adam([params["shN"]], lr=2.5e-3 / 20),
    }

    strategy = DefaultStrategy(
        refine_start_iter=args.densify_start,
        refine_stop_iter=int(args.iterations * args.densify_stop_frac),
        refine_every=args.densify_interval,
        reset_every=3000,
        prune_opa=0.005,
        grow_grad2d=0.0002,
        grow_scale3d=0.01,
        prune_scale3d=0.1,
        verbose=False,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    # ---- training loop ----
    n_views = len(views)
    K_all = torch.stack([torch.from_numpy(v["K"]) for v in views]).to(device)
    VM_all = torch.stack([torch.from_numpy(v["viewmat"]) for v in views]).to(device)
    # pre-load GT to GPU lazily (one at a time) to limit RAM

    bar = tqdm(range(1, args.iterations + 1), desc="train")
    losses = []
    t0 = time.time()
    for it in bar:
        i = int(torch.randint(0, n_views, (1,)).item())
        v = views[i]
        H, W = v["H"], v["W"]
        gt = torch.from_numpy(v["rgb"]).to(device).float() / 255.0  # (H, W, 3)

        K = K_all[i:i+1]            # (1, 3, 3)
        viewmat = VM_all[i:i+1]     # (1, 4, 4)

        colors = torch.cat([params["sh0"], params["shN"]], dim=1)  # (N, K, 3)
        means_n = params["means"]
        scales_n = torch.exp(params["scales"])
        quats_n = F.normalize(params["quats"], dim=-1)
        opacs_n = torch.sigmoid(params["opacities"])

        render, alpha, info = rasterization(
            means_n, quats_n, scales_n, opacs_n, colors,
            viewmat, K, W, H,
            sh_degree=args.sh_degree, packed=False,
        )
        # Must be called BEFORE loss.backward() so it can retain_grad on
        # info[key_for_gradient] (typically means2d) for densification.
        strategy.step_pre_backward(params, optimizers, state, it, info)

        pred = render[0]            # (H, W, 3)
        l1 = (pred - gt).abs().mean()
        loss = l1
        if args.ssim_lambda > 0:
            # cheap "SSIM-lite": gradient L1 to encourage edges
            dx_p = pred[:, 1:] - pred[:, :-1]; dx_g = gt[:, 1:] - gt[:, :-1]
            dy_p = pred[1:, :] - pred[:-1, :]; dy_g = gt[1:, :] - gt[:-1, :]
            grad_l1 = (dx_p - dx_g).abs().mean() + (dy_p - dy_g).abs().mean()
            loss = (1 - args.ssim_lambda) * l1 + args.ssim_lambda * grad_l1

        loss.backward()
        for opt in optimizers.values():
            opt.step(); opt.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, optimizers, state, it, info)

        losses.append(loss.item())
        if it % 50 == 0:
            bar.set_postfix(loss=f"{loss.item():.4f}", N=params["means"].shape[0])
        if it in args.save_iters:
            ply = out_dir / f"gs_{it}.ply"
            save_inria_ply(params, args.sh_degree, ply)
            print(f"\n[iter {it}] saved {ply} ({params['means'].shape[0]:,} Gaussians)")

    print(f"\nTraining done in {(time.time()-t0)/60:.2f} min")
    ply = out_dir / "gs_final.ply"
    save_inria_ply(params, args.sh_degree, ply)
    print(f"Saved {ply}")
    np.save(out_dir / "loss_curve.npy", np.asarray(losses, dtype=np.float32))


def save_inria_ply(params, sh_degree: int, path: Path) -> None:
    """Save Gaussians in the INRIA 3DGS PLY layout."""
    means = params["means"].detach().cpu().numpy()
    scales = params["scales"].detach().cpu().numpy()
    quats = params["quats"].detach().cpu().numpy()  # gsplat uses (w,x,y,z)
    opacs = params["opacities"].detach().cpu().numpy()
    sh0 = params["sh0"].detach().cpu().numpy().reshape(means.shape[0], 3)  # (N,3)
    shN = params["shN"].detach().cpu().numpy().reshape(means.shape[0], -1)  # (N, K*3)
    N = means.shape[0]

    rest_dim = shN.shape[1]  # = ((sh_degree+1)**2 - 1) * 3
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    dtype += [(f"f_dc_{i}", "f4") for i in range(3)]
    dtype += [(f"f_rest_{i}", "f4") for i in range(rest_dim)]
    dtype += [("opacity", "f4"),
              ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
              ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]

    verts = np.empty(N, dtype=dtype)
    verts["x"], verts["y"], verts["z"] = means[:, 0], means[:, 1], means[:, 2]
    verts["nx"] = verts["ny"] = verts["nz"] = 0.0
    for i in range(3):
        verts[f"f_dc_{i}"] = sh0[:, i]
    for i in range(rest_dim):
        verts[f"f_rest_{i}"] = shN[:, i]
    verts["opacity"] = opacs
    for i in range(3):
        verts[f"scale_{i}"] = scales[:, i]
    for i in range(4):
        verts[f"rot_{i}"] = quats[:, i]

    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(path))


if __name__ == "__main__":
    main()
