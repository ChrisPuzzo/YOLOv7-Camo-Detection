#!/usr/bin/env python3
"""
eval.py
-------
Evaluate a trained checkpoint on the val split. Produces:
  - mAP50 / mAP50-95 / per-class precision & recall
  - Confusion matrix PNG (Ultralytics generates it for us)
  - A 4x4 grid of sample predictions for a quick visual sanity check

Usage
-----
    python scripts/eval.py --weights models/best.pt
    python scripts/eval.py --weights runs/detect/camo_gun/weights/best.pt --imgsz 640
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=REPO / "models" / "best.pt")
    p.add_argument("--data", type=str, default=str(REPO / "data.yaml"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--samples", type=int, default=16, help="Number of val images to predict for the visual grid")
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}")

    from ultralytics import YOLO
    model = YOLO(str(args.weights))

    print(f"Validating {args.weights} on {args.data}")
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device or None,
        plots=True,
        verbose=True,
    )

    # Pretty print summary
    box = metrics.box
    print("\n=== Metrics ===")
    print(f"  mAP50      : {box.map50:.4f}")
    print(f"  mAP50-95   : {box.map:.4f}")
    print(f"  precision  : {box.mp:.4f}")
    print(f"  recall     : {box.mr:.4f}")
    if hasattr(box, "ap_class_index") and len(box.ap_class_index):
        names = model.names
        print("\n  Per-class mAP50-95:")
        for cls_idx, ap in zip(box.ap_class_index, box.maps[box.ap_class_index]):
            print(f"    {names[int(cls_idx)]:12s} {ap:.4f}")

    # Visual sample grid
    val_dir = Path(metrics.save_dir) if hasattr(metrics, "save_dir") else REPO / "runs" / "val"
    grid_dir = val_dir / "samples"
    grid_dir.mkdir(parents=True, exist_ok=True)

    val_images_dir = REPO / "dataset" / "images" / "val"
    candidates = sorted(val_images_dir.glob("*.*")) if val_images_dir.is_dir() else []
    if candidates:
        random.seed(0)
        chosen = random.sample(candidates, min(args.samples, len(candidates)))
        print(f"\nGenerating {len(chosen)} sample predictions in: {grid_dir}")
        model.predict(
            source=[str(c) for c in chosen],
            conf=args.conf,
            imgsz=args.imgsz,
            save=True,
            project=str(val_dir),
            name="samples",
            exist_ok=True,
            verbose=False,
        )
    else:
        print(f"\n(no images at {val_images_dir} — skipping sample grid)")

    print(f"\nResults dir: {val_dir}")


if __name__ == "__main__":
    main()
