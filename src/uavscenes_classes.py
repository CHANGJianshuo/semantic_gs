"""UAVScenes label scheme consolidated for the AMtown02 semantic-GS pipeline.

The full UAVScenes label space has 26 classes (cmap.py upstream).  AMtown02
uses 15 of them with a very long-tailed distribution (green_field is 62%,
many classes < 0.1%).  We consolidate into 8 contiguous training classes that
match what is *actually* visible in this scene -- this is also what the
AAE5303 UNet demo used (8 classes).

Training ID  Name              Raw IDs merged in           Approx. % in AMtown02
0            background        0                            18.9
1            roof              1, 17 (transparent_roof)      9.6
2            road              2, 3, 10, 18, 19              6.2
3            water             4, 5                          ~0.0  (rare)
4            green_field       13                           61.7
5            wild_field        14                            3.4
6            vehicle           20, 24                        0.2
7            structure         6, 9, 11, 15, 16, 7, 8, 12,   0.1
                              21, 22, 23, 25
"""
from __future__ import annotations
import numpy as np

NUM_CLASSES = 8
IGNORE_INDEX = 255

CLASS_NAMES = [
    "background", "roof", "road", "water",
    "green_field", "wild_field", "vehicle", "structure",
]

# 8-color RGB palette (vibrant, distinguishable on aerial imagery)
PALETTE = np.array([
    [ 60,  60,  60],   # 0 background   dark grey
    [220,  20,  60],   # 1 roof         crimson
    [255, 255,   0],   # 2 road         yellow
    [ 30, 144, 255],   # 3 water        dodger blue
    [124, 252,   0],   # 4 green_field  lawn green
    [210, 180, 140],   # 5 wild_field   tan
    [255,   0, 255],   # 6 vehicle      magenta
    [255, 140,   0],   # 7 structure    orange
], dtype=np.uint8)

# raw UAVScenes id -> consolidated training id
_RAW_TO_TRAIN = {
    0: 0,                              # background
    1: 1, 17: 1,                       # roof
    2: 2, 3: 2, 10: 2, 18: 2, 19: 2,   # road family
    4: 3, 5: 3,                        # water
    13: 4,                             # green_field
    14: 5,                             # wild_field
    20: 6, 24: 6,                      # vehicle
    6: 7, 9: 7, 11: 7, 15: 7, 16: 7,   # structure
    7: 7, 8: 7, 12: 7,
    21: 7, 22: 7, 23: 7, 25: 7,
}


def build_remap_lut() -> np.ndarray:
    """uint8 LUT of length 256 mapping raw IDs to training IDs."""
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for raw, train in _RAW_TO_TRAIN.items():
        lut[raw] = train
    return lut


def colorize(label_id: np.ndarray) -> np.ndarray:
    """label_id: H,W uint8 in [0..NUM_CLASSES-1] (255=ignore) -> H,W,3 RGB uint8."""
    h, w = label_id.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        out[label_id == c] = PALETTE[c]
    return out
