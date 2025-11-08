import os
import cv2
import numpy as np
from .config import MaskerConfig
from .backends.sam2_backend import SAM2Backend
from .backends.detectron2_backend import Detectron2Backend
from .postproc import refine, to_u8

class MaskerPredictor:
    def __init__(self, cfg: MaskerConfig):
        self.cfg = cfg
        self.sam = SAM2Backend(cfg.sam_ckpt, cfg.sam_model_type) if cfg.sam_ckpt else None
        self.det = Detectron2Backend(score_thresh=cfg.det_conf)
        if self.sam: self.sam.load()
        self.det.load()

    def _det_box(self, det_out, H, W):
        inst = det_out["instances"].to("cpu")
        if len(inst) == 0:
            return None
        idx = int(inst.scores.argmax().item())
        x1, y1, x2, y2 = inst.pred_boxes.tensor[idx].numpy().astype(int)
        x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
        x2 = max(0, min(W-1, x2)); y2 = max(0, min(H-1, y2))
        if x2 <= x1 or y2 <= y1: return None
        return [x1, y1, x2, y2]

    def predict(self, image_bgr: np.ndarray) -> dict:
        H, W = image_bgr.shape[:2]
        det_out = self.det.pred(image_bgr)
        box = self._det_box(det_out, H, W) if self.cfg.use_box_prompt else None

        mask, score = (None, 0.0)
        if self.sam and box is not None:
            mask, score = self.sam.infer(image_bgr, box_xyxy=box)
        if mask is None and self.sam:
            mask, score = self.sam.infer(image_bgr, box_xyxy=None)
        if mask is None:
            mask, score = self.det.infer(image_bgr, None)

        if mask is None:
            return {"mask": None, "score": 0.0, "box": box}

        mask = refine(mask, k=self.cfg.post_k)
        return {"mask": mask, "score": score, "box": box, "size": (H, W)}

    def save(self, mask: np.ndarray, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, to_u8(mask))

def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img
