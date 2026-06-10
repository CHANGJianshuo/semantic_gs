#!/usr/bin/env python3
"""Extract AMtown02 semantic label-id PNGs from interval5_CAM_label.zip and
remap raw UAVScenes class IDs into the 8-class consolidated training space
(see src/uavscenes_classes.py)."""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from uavscenes_classes import (
    CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, PALETTE, build_remap_lut, colorize,
)

ZIP_PATH = Path("/home/chang/semantic_gs/data/labels_raw/interval5_CAM_label.zip")
FRAMES_TXT = Path("/home/chang/semantic_gs/data/frames.txt")
OUT_LABEL_ID = Path("/home/chang/semantic_gs/data/labels")          # remapped 0..7
OUT_LABEL_VIZ = Path("/home/chang/semantic_gs/data/labels_color")   # RGB viz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.5,
                    help="Downscale factor; must match images. Default 0.5.")
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()

    if not ZIP_PATH.exists():
        raise SystemExit(f"Missing zip: {ZIP_PATH}")
    if not FRAMES_TXT.exists():
        raise SystemExit(f"Missing frame list: {FRAMES_TXT} -- run 00_prepare_data.py first")

    stems = [Path(n).stem for n in FRAMES_TXT.read_text().splitlines() if n.strip()]
    OUT_LABEL_ID.mkdir(parents=True, exist_ok=True)
    if not args.no_viz:
        OUT_LABEL_VIZ.mkdir(parents=True, exist_ok=True)

    lut = build_remap_lut()
    train_hist = np.zeros(NUM_CLASSES + 1, dtype=np.int64)  # +1 for ignore bucket

    with zipfile.ZipFile(ZIP_PATH) as z:
        members = set(z.namelist())
        kept, missing = 0, []
        for stem in tqdm(stems, desc="extract"):
            id_name = f"interval5_CAM_label/interval5_AMtown02/interval5_CAM_label_id/{stem}.png"
            if id_name not in members:
                missing.append(stem); continue
            with z.open(id_name) as f:
                buf = np.frombuffer(f.read(), dtype=np.uint8)
            raw = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if raw is None:
                missing.append(stem); continue
            if raw.ndim == 3:
                raw = raw[..., 0]
            h, w = raw.shape
            nw, nh = int(round(w * args.scale)), int(round(h * args.scale))
            raw_small = cv2.resize(raw, (nw, nh), interpolation=cv2.INTER_NEAREST)
            train = lut[raw_small]
            cv2.imwrite(str(OUT_LABEL_ID / f"{stem}.png"), train)

            # histogram (ignore mapped to bucket NUM_CLASSES)
            tmp = np.where(train == IGNORE_INDEX, NUM_CLASSES, train)
            train_hist += np.bincount(tmp.ravel(), minlength=NUM_CLASSES + 1)

            if not args.no_viz:
                viz = colorize(train)
                cv2.imwrite(str(OUT_LABEL_VIZ / f"{stem}.png"),
                            cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
            kept += 1

    print(f"Extracted {kept} / {len(stems)} label maps")
    if missing:
        print(f"Missing {len(missing)} (first 3): {missing[:3]}")
    total = train_hist.sum()
    print("Train-ID histogram:")
    for c in range(NUM_CLASSES):
        pct = 100 * train_hist[c] / max(total, 1)
        print(f"  {c} {CLASS_NAMES[c]:<11} {train_hist[c]:>14,}  ({pct:5.2f}%)")
    print(f"  255 ignore     {train_hist[NUM_CLASSES]:>14,}  "
          f"({100 * train_hist[NUM_CLASSES] / max(total,1):5.2f}%)")
    print(f"Output IDs:   {OUT_LABEL_ID}")
    if not args.no_viz:
        print(f"Output viz:   {OUT_LABEL_VIZ}")


if __name__ == "__main__":
    main()
