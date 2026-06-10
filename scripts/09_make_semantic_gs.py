#!/usr/bin/env python3
"""Rewrite the trained 3DGS PLY with SH-DC colours replaced by the per-Gaussian
semantic palette colour, keeping all other parameters (scale/rotation/opacity).

The result is a *real* 3DGS PLY (INRIA layout) that loads in SuperSplat /
gsplat-viewer / Spark and renders the scene in semantic colours.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from uavscenes_classes import PALETTE  # noqa: E402

GS_PLY = REPO / "output/gs/gs_final.ply"
LABELS = REPO / "output/semantic/semantic_labels.npy"
OUT_PLY = REPO / "output/semantic/semantic_gs.ply"

# SH-0 inverse: stored f_dc = (rgb - 0.5) / C0, with C0 = 1/(2*sqrt(pi)).
# Viewers decode pixel = 0.5 + C0 * f_dc.
C0 = 0.28209479177387814


def main() -> None:
    print(f"Loading {GS_PLY}")
    ply = PlyData.read(str(GS_PLY))
    v = ply["vertex"].data.copy()
    N = len(v)
    print(f"  {N:,} Gaussians, properties: {[p.name for p in ply['vertex'].properties][:12]}...")

    print(f"Loading {LABELS}")
    label = np.load(LABELS)
    if label.shape[0] != N:
        raise SystemExit(f"label count {label.shape[0]} != gaussians {N}")

    # Build per-Gaussian RGB in [0,1]; unseen -> mid grey
    rgb = np.full((N, 3), 0.5, dtype=np.float32)
    for c in range(len(PALETTE)):
        m = (label == c)
        rgb[m] = PALETTE[c].astype(np.float32) / 255.0

    sh0 = (rgb - 0.5) / C0   # invert SH-0 evaluation
    v["f_dc_0"] = sh0[:, 0]
    v["f_dc_1"] = sh0[:, 1]
    v["f_dc_2"] = sh0[:, 2]

    # Zero out the higher-order SH coefficients so the colour stays constant
    # across viewing directions (semantic labels are view-independent).
    for name in v.dtype.names:
        if name.startswith("f_rest_"):
            v[name] = 0.0

    OUT_PLY.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(OUT_PLY))
    size_mb = OUT_PLY.stat().st_size / 1e6
    print(f"Wrote {OUT_PLY} ({size_mb:.1f} MB, {N:,} Gaussians)")


if __name__ == "__main__":
    main()
