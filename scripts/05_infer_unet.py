#!/usr/bin/env python3
"""Run the trained U-Net over every working frame and save per-pixel
softmax probabilities (uint8-quantised) for the back-projection stage."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from uavscenes_classes import NUM_CLASSES, PALETTE  # noqa: E402
from unet.model import UNet  # noqa: E402

FRAMES_TXT = REPO / "data" / "frames.txt"
IMG_DIR = REPO / "data" / "images"
OUT_DIR = REPO / "output" / "unet" / "predictions"
OUT_LABEL_DIR = OUT_DIR / "labels"   # argmax PNGs (uint8)
OUT_PROB_DIR = OUT_DIR / "probs"     # uint8 softmax (H,W,C) .npy
OUT_VIZ_DIR = OUT_DIR / "viz"        # RGB visualisation


def colorize(label: np.ndarray) -> np.ndarray:
    h, w = label.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        out[label == c] = PALETTE[c]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "output/unet/unet_best.pt"))
    ap.add_argument("--scale", type=float, default=0.5,
                    help="Inference downscale; predictions saved at this scale.")
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--save-probs", action="store_true", default=True,
                    help="Also save uint8-quantised softmax (.npy) for soft back-projection.")
    args = ap.parse_args()

    OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PROB_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_viz:
        OUT_VIZ_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    base = ck["args"].get("base", 64)
    model = UNet(n_channels=3, n_classes=NUM_CLASSES, base=base).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"Loaded {args.ckpt} (epoch {ck.get('epoch','?')}, "
          f"val mIoU {ck.get('metrics',{}).get('miou', float('nan'))*100:.2f}%)")

    stems = [Path(n).stem for n in FRAMES_TXT.read_text().splitlines() if n.strip()]
    t0 = time.time()
    with torch.no_grad():
        for stem in tqdm(stems, desc="infer"):
            img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"), cv2.IMREAD_COLOR)
            if img is None:
                print(f"WARN: missing {stem}.jpg"); continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if abs(args.scale - 1.0) > 1e-6:
                h, w = img.shape[:2]
                img = cv2.resize(img, (int(round(w*args.scale)), int(round(h*args.scale))),
                                 interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x)
            probs = F.softmax(logits.float(), dim=1)[0]  # (C, H, W)
            label = probs.argmax(0).to(torch.uint8).cpu().numpy()
            cv2.imwrite(str(OUT_LABEL_DIR / f"{stem}.png"), label)
            if args.save_probs:
                probs_u8 = (probs.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()  # (C,H,W)
                # store as (H,W,C) for friendlier indexing
                np.save(OUT_PROB_DIR / f"{stem}.npy", np.transpose(probs_u8, (1, 2, 0)))
            if not args.no_viz:
                viz = colorize(label)
                cv2.imwrite(str(OUT_VIZ_DIR / f"{stem}.png"),
                            cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

    print(f"\nWrote labels to {OUT_LABEL_DIR}")
    if args.save_probs:
        print(f"Wrote probs  to {OUT_PROB_DIR}")
    if not args.no_viz:
        print(f"Wrote viz    to {OUT_VIZ_DIR}")
    print(f"Elapsed: {(time.time()-t0)/60:.2f} min")


if __name__ == "__main__":
    main()
