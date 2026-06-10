#!/usr/bin/env python3
"""Generate summary figures for the semantic-GS pipeline.

Produces, under figures/:
    unet_training.png       train loss + val mIoU curves
    unet_per_class_iou.png  per-class IoU bar chart (last epoch)
    semantic_class_dist.png class distribution of labelled Gaussians
    gs_loss.png             3DGS training loss curve
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from uavscenes_classes import CLASS_NAMES, NUM_CLASSES, PALETTE  # noqa: E402

FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)


def fig_unet_training():
    hist_path = REPO / "output/unet/history.json"
    if not hist_path.exists():
        print("skip unet_training (no history.json)"); return
    data = json.loads(hist_path.read_text())
    h = data["history"]
    ep = [r["epoch"] for r in h]
    loss = [r["train_loss"] for r in h]
    miou = [r["miou"] * 100 for r in h]
    acc = [r["pixel_acc"] * 100 for r in h]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ep, loss, "o-", color="tab:red", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(ep, miou, "s-", color="tab:blue", label="val mIoU")
    ax2.plot(ep, acc, "^--", color="tab:green", label="val pixel acc")
    ax2.set_ylabel("val metric (%)")
    ax2.legend(loc="center right")
    plt.title("U-Net training on AMtown02")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "unet_training.png", dpi=130)
    plt.close(fig)
    print("wrote unet_training.png")


def fig_unet_per_class():
    hist_path = REPO / "output/unet/history.json"
    if not hist_path.exists():
        print("skip per-class (no history.json)"); return
    data = json.loads(hist_path.read_text())
    iou = np.array(data["history"][-1]["iou"]) * 100
    colors = [PALETTE[c] / 255.0 for c in range(NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(CLASS_NAMES, iou, color=colors, edgecolor="black")
    for i, v in enumerate(iou):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("IoU (%)"); ax.set_ylim(0, 105)
    ax.set_title("U-Net per-class IoU (validation, final epoch)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "unet_per_class_iou.png", dpi=130)
    plt.close(fig)
    print("wrote unet_per_class_iou.png")


def fig_semantic_dist():
    lbl_path = REPO / "output/semantic/semantic_labels.npy"
    if not lbl_path.exists():
        print("skip semantic_dist (no semantic_labels.npy)"); return
    label = np.load(lbl_path)
    counts = [int((label == c).sum()) for c in range(NUM_CLASSES)]
    unseen = int((label == 255).sum())
    colors = [PALETTE[c] / 255.0 for c in range(NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(CLASS_NAMES, counts, color=colors, edgecolor="black")
    total = sum(counts) + unseen
    for i, v in enumerate(counts):
        ax.text(i, v, f"{100*v/max(total,1):.1f}%", ha="center",
                va="bottom", fontsize=9)
    ax.set_ylabel("# Gaussians")
    ax.set_title(f"Semantic label distribution of 3D Gaussians "
                 f"(unseen: {unseen:,})")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "semantic_class_dist.png", dpi=130)
    plt.close(fig)
    print("wrote semantic_class_dist.png")


def fig_gs_loss():
    loss_path = REPO / "output/gs/loss_curve.npy"
    if not loss_path.exists():
        print("skip gs_loss (no loss_curve.npy)"); return
    loss = np.load(loss_path)
    # moving average
    k = max(1, len(loss) // 100)
    ma = np.convolve(loss, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss, color="lightgray", lw=0.6, label="raw")
    ax.plot(np.arange(len(ma)) + k // 2, ma, color="tab:purple",
            lw=1.5, label=f"moving avg ({k})")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss")
    ax.set_title("3D Gaussian Splatting training loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gs_loss.png", dpi=130)
    plt.close(fig)
    print("wrote gs_loss.png")


if __name__ == "__main__":
    fig_unet_training()
    fig_unet_per_class()
    fig_semantic_dist()
    fig_gs_loss()
    print(f"Figures in {FIG_DIR}")
