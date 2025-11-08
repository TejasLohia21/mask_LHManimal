import os
import re
import cv2
import numpy as np
from typing import Iterable

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def is_image(path: str) -> bool:
    return path.lower().endswith(IMG_EXTS)

def list_images(root: str) -> list[str]:
    return [os.path.join(root, f) for f in sorted(os.listdir(root)) if is_image(f)]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def stem(path: str) -> str:
    s = os.path.basename(path)
    return os.path.splitext(s)[0]

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def save_mask_u8(mask: np.ndarray, path: str):
    ensure_dir(os.path.dirname(path))
    m = (mask > 0).astype(np.uint8) * 255
    cv2.imwrite(path, m)

def merge_instances(masks: Iterable[np.ndarray], keep_largest=True) -> np.ndarray:
    masks = [ (m > 0).astype(np.uint8) for m in masks ]
    if not masks:
        return None
    merged = np.clip(np.sum(masks, axis=0), 0, 1).astype(np.uint8)
    if keep_largest:
        num, lab = cv2.connectedComponents(merged)
        if num > 1:
            counts = np.bincount(lab.ravel())
            counts[0] = 0
            idx = counts.argmax()
            merged = (lab == idx).astype(np.uint8)
    return merged

def parse_box(s: str):
    # "x1,y1,x2,y2"
    vals = re.split(r"[,\s]+", s.strip())
    if len(vals) != 4:
        raise ValueError("Box must be x1,y1,x2,y2")
    return [int(v) for v in vals]
