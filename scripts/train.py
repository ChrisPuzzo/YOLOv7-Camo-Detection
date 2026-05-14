#!/usr/bin/env python3
"""
train.py
--------
Fine-tune Ultralytics YOLO26-nano (or YOLO11-nano) on the camo/gun dataset.

Usage
-----
    # Local GPU
    python scripts/train.py --epochs 100 --imgsz 640 --batch 16

    # CPU smoke test (slow, just to verify the pipeline)
    python scripts/train.py --epochs 1 --imgsz 320 --batch 4 --device cpu

    # Fall back to YOLO11n if YOLO26 weights are unavailable
    python scripts/train.py --model yolo11n.pt

Outputs land in `runs/detect/<name>/`. The best checkpoint is also copied to
`models/best.pt` for convenient downstream use (eval, export).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=str, default=str(REPO / "data.yaml"))
    p.add_argument("--model", type=str, default="yolo26n.pt",
                   help="Pretrained checkpoint to fine-tune from. yolo26n.pt | yolo11n.pt | yolo26s.pt | ...")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640,
                   help="Training resolution. 640 is standard; drop to 320 for max speed on weak hardware.")
    p.add_argument("--batch", type=int, default=16,
                   help="-1 = auto, sized to GPU memory.")
    p.add_argument("--device", type=str, default="",
                   help="'' = auto (cuda/mps/cpu), '0' = first CUDA, 'cpu', 'mps'")
    p.add_argument("--name", type=str, default="camo_gun")
    p.add_argument("--patience", type=int, default=30,
                   help="Early-stop patience (epochs without improvement)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="Resume from last checkpoint of a previous run with the same name")
    # Augmentation knobs tuned for camouflage detection.
    # Camo is HARD because by definition it blends into backgrounds — so we lean
    # heavily on color-space and mosaic augmentation, plus copy-paste to multiply
    # examples of partially-occluded targets.
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.15)
    p.add_argument("--copy-paste", type=float, default=0.30, dest="copy_paste")
    p.add_argument("--hsv-h", type=float, default=0.015, dest="hsv_h")
    p.add_argument("--hsv-s", type=float, default=0.7, dest="hsv_s")
    p.add_argument("--hsv-v", type=float, default=0.5, dest="hsv_v")
    p.add_argument("--degrees", type=float, default=10.0)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--fliplr", type=float, default=0.5)
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Install dependencies first: pip install -r requirements.txt")

    model_arg = args.model
    print(f"Loading base model: {model_arg}")
    try:
        model = YOLO(model_arg)
    except Exception as e:
        if "yolo26" in model_arg.lower():
            print(f"  YOLO26 unavailable ({e}). Falling back to yolo11n.pt")
            model = YOLO("yolo11n.pt")
        else:
            raise

    print(f"Training on {args.data}  |  imgsz={args.imgsz}  epochs={args.epochs}  batch={args.batch}")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        resume=args.resume,
        # Augmentations
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        scale=args.scale,
        fliplr=args.fliplr,
        # Sensible quality-of-life defaults
        cos_lr=True,
        amp=True,
        plots=True,
        verbose=True,
    )

    # Copy best.pt to models/ for downstream scripts
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path("runs/detect") / args.name
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        target = REPO / "models" / "best.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, target)
        print(f"\nCopied best checkpoint → {target}")
    print(f"\nTraining complete. Run dir: {save_dir}")


if __name__ == "__main__":
    main()
