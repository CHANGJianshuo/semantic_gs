#!/usr/bin/env python3
"""Subset + downscale the AMtown02 image set for the semantic-GS pipeline.

Source images live in
    /home/chang/2026_2_to_5/5303_3dgs/AMtown02_colmap/images/
(1380 jpgs at 2448x2048). We keep every Nth frame and resize to a working
resolution that's tractable for CPU COLMAP and an 8GB GPU.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

SRC_DIR = Path("/home/chang/2026_2_to_5/5303_3dgs/AMtown02_colmap/images")
DST_DIR = Path("/home/chang/semantic_gs/data/images")
FRAMES_TXT = Path("/home/chang/semantic_gs/data/frames.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2,
                    help="Keep 1 of every N source frames (default: 2 -> ~690 imgs)")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="Downscale factor (default: 0.5 -> 1224x1024)")
    args = ap.parse_args()

    src = sorted(SRC_DIR.glob("*.jpg"))
    if not src:
        raise SystemExit(f"No images found under {SRC_DIR}")

    chosen = src[::args.stride]
    DST_DIR.mkdir(parents=True, exist_ok=True)

    h0, w0 = None, None
    for p in tqdm(chosen, desc="resize"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"WARN: failed to read {p}, skipping")
            continue
        h, w = img.shape[:2]
        h0 = h0 or h
        w0 = w0 or w
        new_w = int(round(w * args.scale))
        new_h = int(round(h * args.scale))
        out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(DST_DIR / p.name), out, [cv2.IMWRITE_JPEG_QUALITY, 92])

    FRAMES_TXT.write_text("\n".join(p.name for p in chosen) + "\n")
    print(f"Kept {len(chosen)} / {len(src)} frames")
    print(f"Source size : {w0}x{h0}")
    print(f"Working size: {int(w0 * args.scale)}x{int(h0 * args.scale)}")
    print(f"Wrote to    : {DST_DIR}")
    print(f"Frame list  : {FRAMES_TXT}")


if __name__ == "__main__":
    main()
