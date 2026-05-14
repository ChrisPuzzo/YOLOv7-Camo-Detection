#!/usr/bin/env python3
"""
fetch_datasets.py
-----------------
Pull pre-labeled camo datasets from Roboflow Universe and merge them into a
single YOLO-format dataset rooted at ./dataset/.

Why: hand-labeling is the slowest part of training. Roboflow Universe has
multiple CC-licensed camo/soldier datasets already in YOLO format. This script
downloads them, remaps every class to our single `camo` class (id=0), and
splits the merged result into train/val.

Sources (defaults; override with --datasets):
  - project-nluoo/camouflaged-soldiers          (~42 imgs)
  - camouflage/soldier-civilian-detection       (~806 imgs, only the soldier class
                                                  is kept; civilians are dropped)

Setup
-----
1. Create a free Roboflow account at https://roboflow.com
2. Get your API key: https://app.roboflow.com/settings/api
3. export ROBOFLOW_API_KEY=your_key_here
4. pip install roboflow
5. python scripts/fetch_datasets.py

Usage
-----
    # Default sources, default split
    python scripts/fetch_datasets.py

    # Custom sources + split ratio
    python scripts/fetch_datasets.py \\
        --datasets project-nluoo/camouflaged-soldiers/2 camouflage/soldier-civilian-detection/3 \\
        --val-ratio 0.15 \\
        --keep-classes soldier camo person

    # Dry run (no download, just show what would happen)
    python scripts/fetch_datasets.py --dry-run

Outputs
-------
    dataset/
    ├── images/{train,val}/
    └── labels/{train,val}/      # .txt files, all class ids remapped to 0

Existing files are preserved by default; pass --reset to wipe dataset/ first.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO / "dataset"

# Default sources: workspace/project/version
DEFAULT_DATASETS = [
    "project-nluoo/camouflaged-soldiers/2",
    "camouflage/soldier-civilian-detection/3",
]

# Class names (case-insensitive, partial match) to keep & remap to 'camo'.
# Anything else gets dropped at label-rewrite time.
DEFAULT_KEEP_CLASSES = [
    "camo", "camouflage", "camouflaged", "soldier", "military", "soldiers",
    "camo-soldier", "camo_soldier", "person-camo",
]


def parse_ref(ref: str):
    """workspace/project/version  →  (workspace, project, int(version))"""
    parts = ref.strip("/").split("/")
    if len(parts) != 3:
        sys.exit(f"Bad dataset ref '{ref}'. Expected workspace/project/version (e.g. 'foo/bar/2').")
    ws, proj, ver = parts
    try:
        return ws, proj, int(ver)
    except ValueError:
        sys.exit(f"Version must be an int, got '{ver}'")


def class_matches(name: str, keep: list[str]) -> bool:
    n = name.lower().strip().replace(" ", "_").replace("-", "_")
    return any(k.lower().replace(" ", "_").replace("-", "_") in n or n in k.lower() for k in keep)


def remap_split(split_dir: Path, keep_classes: list[str], names: list[str]) -> tuple[int, int, int]:
    """
    Rewrite YOLO label files in <split_dir>/labels/, keeping only class ids
    whose name matches keep_classes, remapping all of them to id 0.
    Returns (n_label_files, n_kept_lines, n_dropped_lines).
    """
    labels_dir = split_dir / "labels"
    if not labels_dir.is_dir():
        return 0, 0, 0
    n_files = n_kept = n_dropped = 0
    for txt in labels_dir.glob("*.txt"):
        n_files += 1
        new_lines = []
        for line in txt.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            cname = names[cid] if 0 <= cid < len(names) else str(cid)
            if class_matches(cname, keep_classes):
                new_lines.append("0 " + " ".join(parts[1:]))
                n_kept += 1
            else:
                n_dropped += 1
        txt.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
    return n_files, n_kept, n_dropped


def merge_split(src_split: Path, dest_root: Path, prefix: str, val_ratio: float):
    """Move images+labels from src into dest_root/{images,labels}/{train,val}, with random split."""
    src_imgs = src_split / "images"
    src_lbls = src_split / "labels"
    if not src_imgs.is_dir():
        return 0
    moved = 0
    rng = random.Random(0)
    for img in sorted(src_imgs.iterdir()):
        if not img.is_file():
            continue
        # Some Roboflow exports already split into train/valid/test directories;
        # if so, src_split.name tells us where to put it. Otherwise random split.
        bucket_name = src_split.name.lower()
        if bucket_name in ("train",):
            bucket = "train"
        elif bucket_name in ("valid", "val"):
            bucket = "val"
        elif bucket_name in ("test",):
            # roll test images into val so we don't waste them; we'll hand-pick a
            # final test set later
            bucket = "val"
        else:
            bucket = "val" if rng.random() < val_ratio else "train"

        new_name = f"{prefix}_{img.name}"
        dst_img = dest_root / "images" / bucket / new_name
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst_img)

        lbl = src_lbls / (img.stem + ".txt")
        if lbl.is_file():
            dst_lbl = dest_root / "labels" / bucket / (Path(new_name).stem + ".txt")
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lbl, dst_lbl)
        moved += 1
    return moved


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                   help="Roboflow refs (workspace/project/version)")
    p.add_argument("--keep-classes", nargs="+", default=DEFAULT_KEEP_CLASSES,
                   help="Class-name substrings to keep (everything else dropped)")
    p.add_argument("--val-ratio", type=float, default=0.15,
                   help="If a source has no train/val split, randomly assign this fraction to val")
    p.add_argument("--reset", action="store_true", help="Wipe dataset/ before downloading")
    p.add_argument("--workdir", type=Path, default=REPO / ".rf_cache",
                   help="Where Roboflow's downloads land before we merge them")
    p.add_argument("--dry-run", action="store_true", help="Print plan, don't download")
    args = p.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key and not args.dry_run:
        sys.exit("Set ROBOFLOW_API_KEY env var. Get one at https://app.roboflow.com/settings/api")

    if args.reset and DATASET_DIR.exists():
        print(f"Removing {DATASET_DIR}")
        shutil.rmtree(DATASET_DIR)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (DATASET_DIR / sub).mkdir(parents=True, exist_ok=True)

    args.workdir.mkdir(parents=True, exist_ok=True)

    print("Plan:")
    for ds in args.datasets:
        ws, proj, ver = parse_ref(ds)
        print(f"  - {ws}/{proj} v{ver}")
    print(f"  → merge into {DATASET_DIR.relative_to(REPO)} (val ratio {args.val_ratio})")
    print(f"  → keep classes matching: {args.keep_classes}")
    if args.dry_run:
        return

    try:
        from roboflow import Roboflow  # type: ignore
    except ImportError:
        sys.exit("Install roboflow: pip install roboflow")

    rf = Roboflow(api_key=api_key)

    total_kept = total_dropped = total_files = 0
    for ds in args.datasets:
        ws, proj, ver = parse_ref(ds)
        print(f"\n=== Downloading {ws}/{proj} v{ver} ===")
        try:
            project = rf.workspace(ws).project(proj)
            version = project.version(ver)
            local_dir = args.workdir / f"{ws}__{proj}__v{ver}"
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(parents=True)
            dl = version.download(model_format="yolov8", location=str(local_dir))
        except Exception as e:
            print(f"  [skip] failed to download: {e}")
            continue

        # Read class names from the per-source data.yaml
        ds_yaml = Path(dl.location) / "data.yaml"
        names = []
        if ds_yaml.exists():
            import yaml
            with open(ds_yaml) as f:
                meta = yaml.safe_load(f)
            names = meta.get("names") or []
            if isinstance(names, dict):
                names = [names[i] for i in sorted(names)]

        print(f"  classes in source: {names}")

        # Remap+filter labels per split, then merge
        prefix = f"{ws[:6]}_{proj[:6]}"
        for split_name in ("train", "valid", "val", "test"):
            split_dir = Path(dl.location) / split_name
            if not split_dir.is_dir():
                continue
            nf, nk, nd = remap_split(split_dir, args.keep_classes, names)
            total_files += nf; total_kept += nk; total_dropped += nd
            moved = merge_split(split_dir, DATASET_DIR, prefix, args.val_ratio)
            print(f"    {split_name}: {moved} images merged  (lines kept={nk} dropped={nd})")

    # Summarize
    n_train = sum(1 for _ in (DATASET_DIR / "images/train").glob("*.*"))
    n_val = sum(1 for _ in (DATASET_DIR / "images/val").glob("*.*"))
    print("\n=== Done ===")
    print(f"  train images: {n_train}")
    print(f"  val images  : {n_val}")
    print(f"  label files processed: {total_files}")
    print(f"  bbox lines kept     : {total_kept}")
    print(f"  bbox lines dropped  : {total_dropped}  (non-camo classes filtered out)")
    print(f"\nNext: python scripts/train.py --epochs 100 --imgsz 640")


if __name__ == "__main__":
    main()
