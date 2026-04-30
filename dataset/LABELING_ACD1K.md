# Labeling ACD1K for v2

The [ACD1K dataset](https://www.kaggle.com/datasets/aalihhiader/military-camouflage-soldiers-dataset-mcs1k) (1,000 high-res images, CC0 license) has no bounding boxes. We use it as **augmentation data for v2**, after Beta is already trained on Roboflow data.

## Why label it ourselves

- 1,000 high-quality, high-resolution images is large for a niche detector.
- CC0 license = zero usage friction.
- Diverse environments (urban, forest, desert, marine) → improves generalization.
- Bootstrapping with our v1 model means most boxes are auto-drawn; we just fix mistakes.

## Workflow (estimated ~6–10 hours total once v1 exists)

1. **Download**:
   ```bash
   pip install kaggle
   # Get API token from https://www.kaggle.com/settings → "Create New API Token"
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   kaggle datasets download -d aalihhiader/military-camouflage-soldiers-dataset-mcs1k -p ./acd1k_raw --unzip
   ```

2. **Auto-label with the v1 model** (which we will have trained by then):
   ```bash
   python scripts/bootstrap_labels.py \
     --images ./acd1k_raw \
     --out dataset/labels/train \
     --copy-images dataset/images/train \
     --weights models/best.pt \
     --conf 0.30
   ```
   *(`bootstrap_labels.py` currently targets the legacy YOLOv7 model; minor edit will retarget it to a YOLO26 .pt — todo for v2.)*

3. **Hand-correct** in [Label Studio](https://labelstud.io) or [labelImg](https://github.com/HumanSignal/labelImg):
   - Add boxes the model missed (false negatives are common with hard camo)
   - Tighten loose boxes
   - Delete spurious detections (false positives on tree bark, rocks, etc.)
   - **Skip** images where camo is fundamentally invisible — those are saliency-task examples, not detection-task examples. Maybe ~10–15% of ACD1K falls in this bucket.

4. **Re-split** train/val and **retrain** with the merged Roboflow + ACD1K dataset:
   ```bash
   python scripts/train.py --epochs 150 --imgsz 640 --name camo_v2
   ```

## Quality bar

For a usable v2:
- mAP50 ≥ 0.65 on val (v1 baseline target: 0.50)
- False-positive rate < 5% on a "no camo present" test set you should also collect (just photos of woods with no people)
- Real-field test: at least 3 outings, log perceived hits/misses

## Out-of-scope notes

ACD1K leans toward staged photography (military demos, training exercises). This is somewhat different from a phone-camera view at an airsoft field. To bridge that gap, **also collect 100–200 of your own NH airsoft photos** and add them to the training mix. That domain-specific data will move the needle more than the entire ACD1K dataset combined.
