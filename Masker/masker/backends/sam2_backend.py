import numpy as np
import cv2
from .base import BaseBackend

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

class SAM2Backend(BaseBackend):
    def __init__(self, ckpt_path: str, model_type="vit_h", device=None,
                 auto_cfg=None, multimask=True):
        self.ckpt_path = ckpt_path
        self.model_type = model_type
        self.device = device
        self.auto_cfg = auto_cfg or dict(points_per_side=32, pred_iou_thresh=0.86,
                                         stability_score_thresh=0.9, box_nms_thresh=0.7,
                                         min_mask_region_area=256)
        self.multimask = multimask
        self.sam = None

    def load(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.ckpt_path)
        if self.device is None:
            use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
            self.device = "cuda" if use_cuda else "cpu"
        sam.to(self.device)
        self.sam = sam

    def _best_mask(self, masks, scores=None):
        if masks is None or len(masks) == 0:
            return None, 0.0
        if scores is not None:
            idx = int(np.argmax(scores))
            return masks[idx].astype(np.uint8), float(scores[idx])
        # automatic generator dict list
        masks.sort(key=lambda m: (m.get("predicted_iou", 0.0), m["area"]), reverse=True)
        m0 = masks[0]
        return m0["segmentation"].astype(np.uint8), float(m0.get("predicted_iou", 0.0))

    def infer(self, image_bgr: np.ndarray, box_xyxy=None):
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if box_xyxy is not None:
            predictor = SamPredictor(self.sam)
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(box=np.array(box_xyxy), multimask_output=self.multimask)
            return self._best_mask(masks, scores)
        # auto
        gen = SamAutomaticMaskGenerator(model=self.sam, **self.auto_cfg)
        masks = gen.generate(image)
        return self._best_mask(masks, None)
