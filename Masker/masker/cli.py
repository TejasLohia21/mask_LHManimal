import argparse, os
from . import MaskerConfig
from .predictor import MaskerPredictor
from .utils import list_images, read_image, stem, save_mask_u8

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--sam_type", default="vit_h")
    ap.add_argument("--det_conf", type=float, default=0.5)
    ap.add_argument("--no_box", action="store_true")
    args = ap.parse_args()

    cfg = MaskerConfig(
        sam_ckpt=args.sam_ckpt,
        sam_model_type=args.sam_type,
        det_conf=args.det_conf,
        use_box_prompt=not args.no_box,
    )
    predictor = MaskerPredictor(cfg)
    os.makedirs(args.out_dir, exist_ok=True)

    for ipath in list_images(args.in_dir):
        img = read_image(ipath)
        out = predictor.predict(img)
        if out["mask"] is None:
            print(f"[WARN] no mask: {ipath}")
            continue
        save_mask_u8(out["mask"], os.path.join(args.out_dir, f"{stem(ipath)}_mask.png"))
        print(f"[OK] {ipath}")

if __name__ == "__main__":
    main()
