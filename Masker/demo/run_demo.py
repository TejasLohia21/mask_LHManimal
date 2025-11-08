import os
from masker import MaskerConfig, MaskerPredictor, load_image

IN_DIR = "./demo/images"
OUT_DIR = "./demo/masks"
SAM_CKPT = "./models/sam/sam_vit_h.pth"  # set to your checkpoint

def main():
    cfg = MaskerConfig(sam_ckpt=SAM_CKPT, sam_model_type="vit_h",
                       det_conf=0.5, use_box_prompt=True, post_k=3)
    model = MaskerPredictor(cfg)

    os.makedirs(OUT_DIR, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for name in sorted(os.listdir(IN_DIR)):
        if not name.lower().endswith(exts): continue
        img = load_image(os.path.join(IN_DIR, name))
        out = model.predict(img)
        if out["mask"] is None:
            print(f"[WARN] no mask: {name}")
            continue
        stem = os.path.splitext(name)[0]
        model.save(out["mask"], os.path.join(OUT_DIR, f"{stem}_mask.png"))
        print(f"[OK] {name} score={out['score']:.3f} box={out['box']}")

if __name__ == "__main__":
    main()
