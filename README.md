# Camo & Gun Detector

Real-time, on-device detection of **camouflage clothing** and **guns/airsoft replicas** for use in airsoft, paintball, and tactical training scenarios. Runs entirely on your phone — no server, no cloud, no recurring cost.

> Originally built on YOLOv7-tiny / Darknet. Being rebuilt on **Ultralytics YOLO26-nano** for better mobile performance, NMS-free inference, and clean TFLite/CoreML export. Legacy YOLOv7 weights are preserved under [`legacy/`](legacy/).

## Stack

| Layer | Tech |
|---|---|
| Model | Ultralytics **YOLO26-nano** (fallback: YOLO11n) |
| Training | Ultralytics CLI, Python 3.10+, free Google Colab GPUs |
| Mobile runtime | TensorFlow Lite (Android) / Core ML (iOS) |
| App | React Native + Expo + `react-native-vision-camera` + `react-native-fast-tflite` |

## Repo layout

```
.
├── data.yaml                 # class names + dataset paths
├── dataset/                  # images + YOLO-format labels (gitignored)
├── scripts/
│   ├── bootstrap_labels.py   # auto-label new images using the legacy YOLOv7 model
│   ├── train.py              # fine-tune YOLO26n
│   ├── eval.py               # mAP, confusion matrix, sample-prediction grid
│   └── export_mobile.py      # export to TFLite + CoreML
├── notebooks/
│   └── train_colab.ipynb     # one-click Colab training
├── models/                   # trained .pt / .tflite / .mlpackage live here
├── app/                      # Expo React Native app (on-device inference)
└── legacy/                   # original YOLOv7-tiny darknet weights & cfg
```

## Classes

| ID | Name | Notes |
|---|---|---|
| 0 | `camo` | Person wearing camouflage clothing |
| 1 | `gun` | Firearm or airsoft/paintball replica |

> The "gun" class is broad on purpose. As the dataset grows we may split it into `airsoft_replica` / `real_firearm` / `paintball_marker`.

## Quickstart — training

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. drop labeled data into dataset/images/{train,val} and dataset/labels/{train,val}
# 2. fine-tune
python scripts/train.py --epochs 100 --imgsz 640 --model yolo26n.pt

# 3. evaluate
python scripts/eval.py --weights models/best.pt

# 4. export for mobile
python scripts/export_mobile.py --weights models/best.pt
```

Don't have a GPU? Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) in [Google Colab](https://colab.research.google.com/) — free T4 GPU, no setup.

## Quickstart — phone app

```bash
cd app
npm install
# put your exported model at app/assets/models/camo_gun_int8.tflite
npx expo prebuild
npx expo run:ios       # or  npx expo run:android
```

See [`app/README.md`](app/README.md) for full details (Expo Go vs dev build, EAS, etc).

## Roadmap

- [x] Repo restructure on `feat/yolo26-rebuild`
- [ ] Bootstrap labels from legacy model
- [ ] Collect & label v1 dataset (~500 images / class)
- [ ] Train YOLO26n baseline
- [ ] Export TFLite (INT8) + CoreML
- [ ] Wire up Expo app with live detection overlay
- [ ] On-device benchmark on real phone
- [ ] v1 release

## License

TBD.
