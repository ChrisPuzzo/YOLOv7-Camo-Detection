# Build a test APK

This walks you through producing an installable Android APK of the Camo
Detector Beta with the trained model bundled in.

## Prerequisites (one time)

1. **Node 20+** — `node --version`
2. **EAS CLI** — `npm install -g eas-cli`
3. **Expo account** — free tier is fine. Sign up at https://expo.dev/signup
4. **Trained model in place** — `app/assets/models/camo_int8.tflite` must
   exist. The repo ships with the v0.1.0-beta model already; if you re-trained,
   regenerate with:

   ```bash
   python scripts/export_mobile.py --weights models/best.pt --formats tflite --imgsz 320 --int8
   cp models/best_int8.tflite app/assets/models/camo_int8.tflite
   ```

## Build the APK

```bash
cd app

# 1. Install deps
npm install

# 2. Log in to your Expo account
eas login

# 3. First-time only: link this directory to a new EAS project
eas init --id auto    # creates a project on expo.dev under your account

# 4. Build the APK on EAS' free cloud builder (takes ~10–15 min)
eas build --profile preview --platform android
```

When the build finishes, the CLI prints a URL like:

```
https://expo.dev/accounts/<you>/projects/camo-detector-beta/builds/<id>
```

That page has an `Install` button (scan with your phone's camera) and a direct
APK download link.

## Install on a phone

### Easiest — scan the QR code on the build page

Visit the build URL on your laptop, then scan the QR code with your Android
phone's camera. Tap the link, allow "install from unknown sources" if
prompted, and tap **Install**.

### Or — download the APK and sideload

```bash
# After the build finishes
eas build:list --platform android --limit 1
# Copy the APK URL it prints, then:

curl -L -o camo-detector-beta.apk "<APK_URL>"

# On the phone, transfer the file (USB or Files by Google) and open it.
# Allow "install from unknown sources" if prompted.
```

## Local build (no Expo account)

If you'd rather not use EAS' cloud:

```bash
cd app
npx expo prebuild --platform android   # generates android/ folder
cd android
./gradlew assembleRelease               # APK lands in android/app/build/outputs/apk/release/
```

You'll need Android Studio + the Android SDK installed. EAS is the path of
least resistance.

## Troubleshooting

### Build fails with `vision-camera` worklet errors

Make sure `react-native-worklets-core` and `react-native-reanimated` are at
the versions in `package.json` and that `babel.config.js` has the reanimated
plugin last in the list.

### App opens but camera is black

- Permission was denied. Long-press the app icon → App info → Permissions →
  enable Camera.
- The phone is older than Android 8 (`minSdkVersion: 26`). Use a newer
  device or lower `minSdkVersion` in `app.json`.

### "Failed to load model" on first open

`app/assets/models/camo_int8.tflite` wasn't bundled. Verify the file exists
locally before running `eas build`, and confirm `assetBundlePatterns` in
`app.json` includes `assets/models/*.tflite`.

### TFLite says "Cannot resolve operator"

The model uses an op the TFLite runtime in `react-native-fast-tflite` doesn't
know. Re-export with `nms=False` (decode in JS via `postprocess.ts` instead of
the baked NMS op) and rebuild.

## What the Beta APK includes

- Live camera preview with on-device YOLO inference
- Single-class `camo` detector (gun class deferred to Pro)
- ~8 FPS inference on most mid-range phones (preview renders at 60 FPS)
- Bundled model: `camo_int8.tflite` v0.1.0-beta — see `models/README.md`

## What it doesn't include yet

- Ads (planned for the open Beta release — see `MONETIZATION.md`)
- Detection history / saved clips
- Multi-camera switching
- Additional classes (gun, vehicle, etc.) — Pro features
