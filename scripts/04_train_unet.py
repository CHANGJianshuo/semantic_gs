#!/usr/bin/env python3
"""Train the U-Net on AMtown02 images + masks.

Default: scale 0.5 of working images (= ~0.25 of native), batch 4, 30 epochs,
Adam 1e-4, CE+Dice loss, on GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from uavscenes_classes import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES  # noqa: E402
from unet.dataset import AMtown02Seg, make_splits  # noqa: E402
from unet.model import UNet, dice_loss  # noqa: E402

FRAMES_TXT = REPO / "data" / "frames.txt"
OUT_DIR = REPO / "output" / "unet"


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for img, lbl in loader:
        img = img.to(device, non_blocking=True)
        lbl = lbl.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(img)
        pred = logits.argmax(dim=1)
        valid = lbl != IGNORE_INDEX
        p = pred[valid].cpu().numpy()
        t = lbl[valid].cpu().numpy()
        idx = t * NUM_CLASSES + p
        conf += np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)

    inter = np.diag(conf).astype(np.float64)
    union = conf.sum(0) + conf.sum(1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    miou = float(np.nanmean(iou))
    acc = float(inter.sum() / max(conf.sum(), 1))
    return {"miou": miou, "iou": iou.tolist(), "pixel_acc": acc,
            "confusion": conf.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="Downscale factor on top of working images. Default 0.5 -> 612x512.")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=int, default=64, help="UNet base channels")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", default=True)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_stems, val_stems = make_splits(FRAMES_TXT, args.val_frac, args.seed)
    print(f"Train: {len(train_stems)}  Val: {len(val_stems)}")

    train_ds = AMtown02Seg(train_stems, train=True, scale=args.scale)
    val_ds = AMtown02Seg(val_stems, train=False, scale=args.scale)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # class-balanced CE weight (inverse-frequency, capped) based on the
    # global histogram measured during label prep.
    # Approx %: bg 18.9, roof 9.7, road 6.3, water 0.0, green 61.4,
    #          wild 3.4, vehicle 0.2, structure 0.1
    freq = torch.tensor([18.9, 9.7, 6.3, 0.01, 61.4, 3.4, 0.2, 0.1],
                        dtype=torch.float32)
    weight = (1.0 / (freq + 0.5)).clamp(max=20.0)
    weight = weight / weight.mean()
    weight = weight.to(device)
    print("CE class weights:", weight.cpu().numpy().round(3).tolist())

    model = UNet(n_channels=3, n_classes=NUM_CLASSES, base=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"UNet base={args.base}, {n_params:.2f} M params")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device=device.type, enabled=args.amp)

    history = []
    best_miou = -1.0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        run_loss = 0.0
        pbar = tqdm(train_dl, desc=f"ep {ep+1}/{args.epochs}", leave=False)
        for img, lbl in pbar:
            img = img.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
                logits = model(img)
                loss_ce = F.cross_entropy(logits, lbl, weight=weight,
                                          ignore_index=IGNORE_INDEX)
                loss_d = dice_loss(logits, lbl, NUM_CLASSES, ignore_index=IGNORE_INDEX)
                loss = loss_ce + loss_d
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            run_loss += loss.item() * img.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        sched.step()
        train_loss = run_loss / len(train_ds)

        metrics = evaluate(model, val_dl, device)
        elapsed = time.time() - t0
        line = (f"ep {ep+1:02d} | loss {train_loss:.4f} | "
                f"val mIoU {metrics['miou']*100:5.2f} | "
                f"acc {metrics['pixel_acc']*100:5.2f} | "
                f"lr {sched.get_last_lr()[0]:.2e} | {elapsed/60:.1f} min")
        print(line)
        history.append({"epoch": ep + 1, "train_loss": train_loss, **metrics})

        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            torch.save({"model": model.state_dict(),
                        "args": vars(args),
                        "epoch": ep + 1,
                        "metrics": metrics}, OUT_DIR / "unet_best.pt")
            print(f"  -> new best, saved {OUT_DIR/'unet_best.pt'}")

    torch.save({"model": model.state_dict(), "args": vars(args),
                "epoch": args.epochs,
                "metrics": history[-1] if history else None},
               OUT_DIR / "unet_last.pt")
    with open(OUT_DIR / "history.json", "w") as f:
        json.dump({"args": vars(args), "history": history,
                   "best_miou": best_miou}, f, indent=2)
    print(f"\nBest val mIoU: {best_miou*100:.2f}%")
    print(f"Per-class IoU (last epoch):")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {c} {name:<11} {history[-1]['iou'][c]*100:5.2f}")


if __name__ == "__main__":
    main()
