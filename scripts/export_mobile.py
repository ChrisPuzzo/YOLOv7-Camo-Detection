#!/usr/bin/env python3
"""
export_mobile.py
----------------
Export a trained checkpoint to mobile-ready formats:

  - TFLite (Android / cross-platform via react-native-fast-tflite)
      * float32 by default
      * INT8 quantized (smaller + faster) when --int8 is set, requires a
        small calibration set of representative images
  - CoreML (iOS, only works on macOS hosts)
  - ONNX (handy for desktop testing / fallback runtimes)

Outputs land in models/ alongside the source .pt.

Usage
-----
    # Everything (skip CoreML if you're not on macOS)
    python scripts/export_mobile.py --weights models/best.pt --formats tflite coreml onnx --int8

    # Just TFLite (most common)
    python scripts/export_mobile.py --weights models/best.pt --formats tflite --int8

After this runs, copy the .tflite into the app:
    cp models/best_int8.tflite app/assets/models/camo_gun_int8.tflite
"""
from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=REPO / "models" / "best.pt")
    p.add_argument("--formats", nargs="+", default=["tflite", "onnx"],
                   choices=["tflite", "coreml", "onnx", "saved_model"])
    p.add_argument("--imgsz", type=int, default=640,
                   help="Export resolution. 320 for max speed on phone, 640 for max accuracy.")
    p.add_argument("--int8", action="store_true",
                   help="Quantize to INT8. Smaller + faster, ~minor accuracy hit. Requires --calib-data.")
    p.add_argument("--half", action="store_true",
                   help="FP16 export (half precision). Smaller than FP32, no calibration needed.")
    p.add_argument("--calib-data", type=str, default=None,
                   help="Path to a YAML or folder with calibration images. Defaults to data.yaml's val split.")
    p.add_argument("--nms", action="store_true", default=True,
                   help="Embed NMS in the exported graph (YOLO11 only; YOLO26 is NMS-free natively).")
    args = p.parse_args()

    if not args.weights.exists():
        sys.exit(f"weights not found: {args.weights}")

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Install dependencies first: pip install -r requirements.txt")

    model = YOLO(str(args.weights))
    print(f"Loaded: {args.weights}  task={model.task}")

    out_files = []
    for fmt in args.formats:
        if fmt == "coreml" and platform.system() != "Darwin":
            print(f"\n[skip] CoreML export only works on macOS (you're on {platform.system()}).")
            continue

        kwargs = dict(format=fmt, imgsz=args.imgsz)
        if fmt == "tflite":
            kwargs["int8"] = args.int8
            kwargs["half"] = args.half and not args.int8
            if args.calib_data:
                kwargs["data"] = args.calib_data
            elif args.int8:
                # Need calibration data for INT8. Default to data.yaml.
                kwargs["data"] = str(REPO / "data.yaml")
        elif fmt == "coreml":
            kwargs["int8"] = args.int8
            kwargs["half"] = args.half and not args.int8
            kwargs["nms"] = args.nms
        elif fmt == "onnx":
            kwargs["half"] = args.half
            kwargs["dynamic"] = False
            kwargs["simplify"] = True
            kwargs["nms"] = args.nms

        print(f"\n--- Exporting {fmt} ---  {kwargs}")
        try:
            path = model.export(**kwargs)
        except Exception as e:
            print(f"  [fail] {fmt}: {e}")
            continue
        path = Path(path) if isinstance(path, (str, Path)) else Path(str(path))
        out_files.append(path)
        print(f"  wrote: {path}")

    if not out_files:
        sys.exit("Nothing exported.")

    # Stage copies in models/ with predictable names
    stage = REPO / "models"
    stage.mkdir(parents=True, exist_ok=True)
    print("\n=== Staged in models/ ===")
    for f in out_files:
        if not f.exists():
            continue
        suffix = "_int8" if args.int8 and f.suffix == ".tflite" else ""
        target_name = args.weights.stem + suffix + f.suffix
        target = stage / target_name
        if f.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(f, target)
        else:
            shutil.copy2(f, target)
        print(f"  {target}")

    print("\nNext: copy the TFLite/CoreML model into app/assets/models/ and rebuild the Expo app.")


if __name__ == "__main__":
    main()
