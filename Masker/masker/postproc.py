import numpy as np
import cv2

def largest_cc(mask):
    num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if num <= 1:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    idx = counts.argmax()
    return (labels == idx).astype(np.uint8)

def refine(mask, k=3):
    mask = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = largest_cc(mask)
    return mask

def to_u8(mask):
    return (mask > 0).astype(np.uint8) * 255
