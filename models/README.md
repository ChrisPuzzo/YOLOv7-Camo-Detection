# Pre-trained model weights

Model binaries are **not committed to git** — they ship as release assets to
keep the repo small and the history clean.

## Downloads

Latest weights (single-class `camo`, trained on a merged Roboflow set of
~1,135 images):

- **`best.pt`** — 5.2 MB Ultralytics YOLO26-nano checkpoint. Use this with
  `scripts/eval.py`, `scripts/export_mobile.py`, or any further fine-tuning.
- **`camo_int8.tflite`** — 2.6 MB INT8-quantized TFLite, `imgsz=320`. Drop
  into `app/assets/models/camo_int8.tflite` for the Expo build.

Grab them from the latest GitHub Release:

```
gh release download v0.1.0-beta --repo ChrisPuzzo/YOLOv7-Camo-Detection \
    --pattern 'best.pt' --pattern 'camo_int8.tflite' --dir ./
```

Or visit https://github.com/ChrisPuzzo/YOLOv7-Camo-Detection/releases.

## Reproducing the weights

See `notebooks/train_colab.ipynb` (or the matching cells documented in PR #1).

## Re-exporting TFLite from `best.pt`

```bash
python scripts/export_mobile.py \
    --weights models/best.pt \
    --formats tflite \
    --imgsz 320 \
    --int8
```

## Why these aren't in git

- `.pt` and `.tflite` are large binary blobs git can't diff
- They're generated artifacts, not source — easy to regenerate
- Releases give us versioned, downloadable URLs without inflating clones

`.gitignore` blocks `models/*.pt`, `models/*.tflite`, and
`app/assets/models/*.tflite` so they can't accidentally land in a commit.
