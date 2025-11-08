import numpy as np
import cv2
from .base import BaseBackend
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

class Detectron2Backend(BaseBackend):
    def __init__(self, config="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                 score_thresh=0.5):
        self.config = config
        self.score_thresh = score_thresh
        self.pred = None

    def load(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_thresh
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.config)
        self.pred = DefaultPredictor(cfg)

    def infer(self, image_bgr: np.ndarray, box_xyxy=None):
        out = self.pred(image_bgr)
        inst = out["instances"].to("cpu")
        if len(inst) == 0 or not inst.has("pred_masks"):
            return None, 0.0
        idx = int(inst.scores.argmax().item())
        m = inst.pred_masks[idx].numpy().astype(np.uint8)
        H, W = image_bgr.shape[:2]
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.uint8), float(inst.scores[idx].item())
