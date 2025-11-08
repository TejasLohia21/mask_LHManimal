from abc import ABC, abstractmethod
import numpy as np

class BaseBackend(ABC):
    @abstractmethod
    def load(self): ...
    @abstractmethod
    def infer(self, image_bgr: np.ndarray, box_xyxy=None) -> tuple[np.ndarray, float] | tuple[None, float]:
        """Return (mask[H,W]{0,1}, score[0..1]) or (None, 0.0)."""
        ...
