# Camo & Gun Detector — Mobile App

Expo React Native app that runs YOLO inference **fully on-device** using TensorFlow Lite.

## Stack

- [Expo](https://expo.dev) (managed workflow with prebuild)
- [react-native-vision-camera](https://github.com/mrousavy/react-native-vision-camera) for the live camera + frame processors
- [react-native-fast-tflite](https://github.com/mrousavy/react-native-fast-tflite) for on-device TFLite inference
- [react-native-worklets-core](https://github.com/margelo/react-native-worklets-core) + Reanimated for the worklet bridge

## Prerequisites

- Node 18+
- macOS + Xcode (for iOS) **or** Android Studio + JDK 17 (for Android)
- A real device — the simulator/emulator camera is not useful here

## Getting started

```bash
cd app
npm install
```

Drop your trained mobile model into `assets/models/`:

```bash
# from the repo root, after running scripts/export_mobile.py
cp ../models/best_int8.tflite assets/models/camo_gun_int8.tflite
```

> Don't have a trained model yet? You can drop a stock YOLO11n / YOLO26n COCO TFLite in there to verify the camera + inference pipeline works end-to-end. The app will show "person" boxes etc — useful for debugging.

### iOS

```bash
npx expo prebuild --platform ios
npx expo run:ios --device
```

### Android

```bash
npx expo prebuild --platform android
npx expo run:android --device
```

## Why a "dev build" instead of Expo Go?

Expo Go ships a fixed set of native modules. `react-native-vision-camera` and `react-native-fast-tflite` aren't included, so we need a custom dev build. `expo prebuild` + `run:ios|android` does this for you. After the first build, `npx expo start --dev-client` is enough for live reload.

## Tuning

Edit constants at the top of `src/CameraScreen.tsx`:

```ts
const CLASS_NAMES = ['camo', 'gun'];
const MODEL_INPUT_SIZE = 320; // must match what was set during scripts/export_mobile.py
const TARGET_FPS = 8;          // inference cadence; preview always renders at full FPS
```

## Frame preprocessing

The `runSync(frame)` call in `CameraScreen.tsx` passes the camera frame straight through. In practice you'll want to plug in [vision-camera-resize-plugin](https://github.com/mrousavy/vision-camera-resize-plugin) so the frame is resized + normalized to the model's expected input shape on the GPU. That plugin is the production-ready choice — install it and replace the marked TODO line in the worklet.

## Performance tips

- Use **INT8** quantization (`scripts/export_mobile.py --int8`) — typically 2–3× faster than FP32 on phone CPUs and ~25% the size.
- Drop `MODEL_INPUT_SIZE` to 320 if you can't hit your target FPS at 640.
- Lower `TARGET_FPS` to 4–6 on older phones; the preview still feels smooth because it renders independently.
- On iOS, the Neural Engine is auto-used by TFLite via Core ML delegate. If you ship a `.mlpackage` instead, switch the loader to a CoreML runtime — but TFLite + delegate is usually fast enough.
