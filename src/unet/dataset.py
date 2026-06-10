"""AMtown02 segmentation dataset (images + remapped 8-class masks)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_DIR = Path("/home/chang/semantic_gs/data/images")
LBL_DIR = Path("/home/chang/semantic_gs/data/labels")


class AMtown02Seg(Dataset):
    """Returns (image_chw_float in [0,1], label_hw_long) pairs."""

    def __init__(self, stems: List[str], train: bool = False,
                 scale: float = 0.5):
        super().__init__()
        self.stems = stems
        self.train = train
        self.scale = scale

    def __len__(self) -> int:
        return len(self.stems)

    def _load(self, stem: str):
        img = cv2.imread(str(IMG_DIR / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        lbl = cv2.imread(str(LBL_DIR / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if img is None or lbl is None:
            raise FileNotFoundError(stem)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, lbl

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img, lbl = self._load(stem)

        if abs(self.scale - 1.0) > 1e-6:
            h, w = img.shape[:2]
            nh, nw = int(round(h * self.scale)), int(round(w * self.scale))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            lbl = cv2.resize(lbl, (nw, nh), interpolation=cv2.INTER_NEAREST)

        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :].copy()
                lbl = lbl[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img = img[::-1, :, :].copy()
                lbl = lbl[::-1, :].copy()

        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        lbl_t = torch.from_numpy(lbl.astype(np.int64))
        return img_t, lbl_t


def make_splits(frames_txt: Path, val_frac: float = 0.1, seed: int = 0):
    stems = [Path(n).stem for n in frames_txt.read_text().splitlines() if n.strip()]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(stems))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(stems) * val_frac)))
    val_ids = set(idx[:n_val].tolist())
    train_stems = [s for i, s in enumerate(stems) if i not in val_ids]
    val_stems = [s for i, s in enumerate(stems) if i in val_ids]
    return train_stems, val_stems
