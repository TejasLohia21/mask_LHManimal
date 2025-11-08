from dataclasses import dataclass

@dataclass
class MaskerConfig:
    sam_ckpt: str | None = None
    sam_model_type: str = "vit_h"
    det_conf: float = 0.5
    use_box_prompt: bool = True
    post_k: int = 3
