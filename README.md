# Camo Detector

Real-time, on-device detection of **people wearing camouflage** for airsoft, paintball, and tactical training. Runs entirely on your phone — no server, no cloud, no recurring cost.

> Originally built on YOLOv7-tiny / Darknet (legacy weights preserved under [`legacy/`](legacy/)). Rebuilt on **Ultralytics YOLO26-nano** for better mobile performance, NMS-free inference, and clean TFLite/CoreML export.

## Why one class?

v1 detects only `camo`. We deliberately dropped the `gun`/`weapon` class to:

- Sidestep [Google's weapons-advertising policy](http://nlairsoft.com/item/google-banning-advertisement-of-airsoft-products) which bans airsoft/BB/paintball ads
- Keep the free Beta safely within Play Store / App Store guidelines
- Reserve weapon detection as a **paid Pro feature** in v2 (see [`MONETIZATION.md`](MONETIZATION.md))

## Stack

| Layer | Tech |
|---|---|
| Model | Ultralytics **YOLO26-nano** (fallback: YOLO11n) |
| Training | Ultralytics CLI, Python 3.10+, Google Colab T4 (free) |
| Mobile runtime | TensorFlow Lite (Android) / Core ML (iOS) |
| App | React Native + Expo + `react-native-vision-camera` + `react-native-fast-tflite` |

## Repo layout

```
.
├── data.yaml                 # class names + dataset paths
├── dataset/                  # images + YOLO-format labels (gitignored)
├── scripts/
│   ├── fetch_datasets.py     # pull pre-labeled Roboflow datasets, merge to single class
│   ├── bootstrap_labels.py   # auto-label new images using the legacy YOLOv7 model
│   ├── train.py              # fine-tune YOLO26n
│   ├── eval.py               # mAP, confusion matrix, sample predictions
│   └── export_mobile.py      # export to TFLite + CoreML + ONNX
├── notebooks/
│   └── train_colab.ipynb     # one-click Colab training
├── models/                   # trained .pt / .tflite / .mlpackage land here
├── app/                      # Expo React Native app (on-device inference)
└── legacy/                   # original YOLOv7-tiny darknet weights & cfg
```

## Quickstart — first training run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Pull pre-labeled datasets from Roboflow Universe (free; needs a Roboflow API key)
export ROBOFLOW_API_KEY=YOUR_KEY
python scripts/fetch_datasets.py

# 2. Fine-tune YOLO26-nano on the merged dataset
python scripts/train.py --epochs 100 --imgsz 640

# 3. Evaluate
python scripts/eval.py --weights models/best.pt

# 4. Export for mobile
python scripts/export_mobile.py --weights models/best.pt --int8 --formats tflite onnx
```

Don't have a GPU? Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) in [Google Colab](https://colab.research.google.com/) — free T4 GPU, no setup.

## Quickstart — phone app

```bash
cd app
npm install
cp ../models/best_int8.tflite assets/models/camo_int8.tflite
npx expo prebuild
npx expo run:ios       # or:  npx expo run:android
```

See [`app/README.md`](app/README.md) for full details.

## Roadmap

### v1 — Free Beta (current focus)
- [x] Repo restructure on `feat/yolo26-rebuild`
- [x] Single-class `camo` setup
- [ ] Pull pre-labeled datasets (Roboflow camouflaged-soldiers + soldier-civilian-detection)
- [ ] Train YOLO26n baseline on Colab
- [ ] Export INT8 TFLite + CoreML
- [ ] Wire up Expo app with live detection overlay
- [ ] Field-test on a real airsoft outing
- [ ] Free Beta release (TestFlight / Play internal test) — see [`MONETIZATION.md`](MONETIZATION.md)

### v2 — Paid Pro
- [ ] Hand-label ACD1K (1,000 high-res CC0 images)
- [ ] Add weapon-class detection (Pro-only)
- [ ] Detection history / replay
- [ ] Multi-camera / squad mode
- [ ] Paid release on App Store / Play Store

## License

TBD — likely MIT for code, with model weights under their training-data licenses.
