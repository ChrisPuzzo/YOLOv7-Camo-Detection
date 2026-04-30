#!/usr/bin/env python3
"""
bootstrap_labels.py
-------------------
Use the legacy YOLOv7-tiny darknet model (camo.weights + yolov7-tiny.cfg) to
auto-label a folder of new images, producing YOLO-format .txt labels you can
then hand-correct in Label Studio / Roboflow / labelImg.

This shortcuts the cold-start labeling problem: instead of starting from scratch,
you start from "the old model's best guess" and just fix mistakes.

Usage
-----
    python scripts/bootstrap_labels.py \
        --images /path/to/raw_images \
        --out dataset/labels/train \
        --copy-images dataset/images/train \
        --conf 0.25 \
        --weights legacy/dnCustom/camo.weights \
        --cfg    legacy/dnCustom/yolov7-tiny.cfg

Outputs
-------
For each image foo.jpg, writes foo.txt next to it (in --out) with lines:
    <class_id> <x_center> <y_center> <width> <height>
all values normalized 0..1, YOLO convention.

Optional --copy-images also copies the image into the dataset/images/ folder
so train/val splits stay tidy.

Notes
-----
- Uses OpenCV DNN, no darknet build required.
- Writes empty .txt for images with no detections (valid YOLO "background" sample).
- Class indices come straight from the old cfg (2 classes). Confirm the order
  by running --preview on a few images first.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_NAMES = ["camo", "gun"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_legacy_net(cfg: Path, weights: Path) -> cv2.dnn_Net:
    if not cfg.exists():
        sys.exit(f"cfg not found: {cfg}")
    if not weights.exists():
        sys.exit(f"weights not found: {weights}")
    net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def detect(net: cv2.dnn_Net, img: np.ndarray, conf_thr: float, nms_thr: float, input_size: int):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img, 1 / 255.0, (input_size, input_size), swapRB=True, crop=False
    )
    net.setInput(blob)
    out_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(out_names)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        # darknet YOLO output: [cx, cy, w, h, obj, c0, c1, ...]
        for det in output:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf < conf_thr:
                continue
            cx, cy, bw, bh = det[0:4]
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            boxes.append([float(x), float(y), float(bw * w), float(bh * h)])
            confidences.append(conf)
            class_ids.append(class_id)

    if not boxes:
        return [], [], []

    keep = cv2.dnn.NMSBoxes(boxes, confidences, conf_thr, nms_thr)
    if len(keep) == 0:
        return [], [], []
    keep = np.array(keep).flatten().tolist()
    return (
        [boxes[i] for i in keep],
        [confidences[i] for i in keep],
        [class_ids[i] for i in keep],
    )


def to_yolo_line(box, class_id: int, w: int, h: int) -> str:
    x, y, bw, bh = box
    cx = (x + bw / 2) / w
    cy = (y + bh / 2) / h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}"


def draw_preview(img, boxes, confs, class_ids, names):
    out = img.copy()
    for (x, y, bw, bh), c, cid in zip(boxes, confs, class_ids):
        x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
        color = (0, 255, 0) if cid == 0 else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{names[cid] if cid < len(names) else cid}: {c:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, required=True, help="Folder of raw images to auto-label")
    p.add_argument("--out", type=Path, required=True, help="Folder to write YOLO .txt labels into")
    p.add_argument("--copy-images", type=Path, default=None,
                   help="Also copy images into this folder (e.g. dataset/images/train)")
    p.add_argument("--weights", type=Path, default=Path("legacy/dnCustom/camo.weights"))
    p.add_argument("--cfg", type=Path, default=Path("legacy/dnCustom/yolov7-tiny.cfg"))
    p.add_argument("--names", nargs="*", default=DEFAULT_NAMES,
                   help="Class names in cfg order (default: camo gun)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--nms", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=320,
                   help="Network input size (must match the cfg; YOLOv7-tiny here = 320)")
    p.add_argument("--preview", action="store_true",
                   help="Save side-by-side preview PNGs to <out>/_preview/ instead of writing labels")
    p.add_argument("--limit", type=int, default=0, help="Process only the first N images (0 = all)")
    args = p.parse_args()

    if not args.images.is_dir():
        sys.exit(f"--images dir not found: {args.images}")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        args.copy_images.mkdir(parents=True, exist_ok=True)
    preview_dir = args.out / "_preview"
    if args.preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading legacy YOLOv7-tiny: {args.weights.name} + {args.cfg.name}")
    net = load_legacy_net(args.cfg, args.weights)

    images = sorted([p for p in args.images.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if args.limit:
        images = images[: args.limit]
    print(f"Found {len(images)} images")

    n_with, n_without = 0, 0
    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip] unreadable: {img_path}")
            continue
        h, w = img.shape[:2]
        boxes, confs, class_ids = detect(net, img, args.conf, args.nms, args.imgsz)

        if args.preview:
            preview = draw_preview(img, boxes, confs, class_ids, args.names)
            cv2.imwrite(str(preview_dir / img_path.name), preview)
        else:
            label_path = args.out / (img_path.stem + ".txt")
            lines = [to_yolo_line(b, c, w, h) for b, c in zip(boxes, class_ids)]
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
            if args.copy_images:
                shutil.copy2(img_path, args.copy_images / img_path.name)

        if boxes:
            n_with += 1
        else:
            n_without += 1
        if i % 25 == 0 or i == len(images):
            print(f"  [{i}/{len(images)}] with-detections={n_with} empty={n_without}")

    if args.preview:
        print(f"\nPreviews written to: {preview_dir}")
        print("Inspect them, then re-run without --preview to write real labels.")
    else:
        print(f"\nLabels written to: {args.out}")
        if args.copy_images:
            print(f"Images copied to:  {args.copy_images}")
        print("Next: open these in Label Studio / Roboflow / labelImg and correct mistakes.")


if __name__ == "__main__":
    main()
