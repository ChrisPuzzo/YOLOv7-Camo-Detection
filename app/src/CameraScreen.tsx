/**
 * CameraScreen.tsx
 * ----------------
 * Live camera preview with on-device YOLO inference.
 *
 * Pipeline:
 *   VisionCamera frame  →  resize+normalize  →  TFLite forward (worklet)  →
 *   decode to Detection[]  →  draw overlay (Reanimated shared values)
 *
 * Everything runs on-device. No network calls.
 */
import React, { useEffect, useMemo, useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
  runAtTargetFps,
} from 'react-native-vision-camera';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { useSharedValue } from 'react-native-reanimated';
import { Worklets } from 'react-native-worklets-core';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { decodeYoloOutput, Detection } from './postprocess';
import BoundingBoxOverlay from './BoundingBoxOverlay';
import ErrorScreen from './ErrorScreen';

const CLASS_NAMES = ['camo'];   // Beta = single-class. Pro will reintroduce more.
const MODEL_INPUT_SIZE = 320;   // must match what was set during export
const TARGET_FPS = 8;           // run inference at ~8 FPS, render preview at 30/60

export default function CameraScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');

  // Force a remount by bumping `loadKey` — used as the retry mechanism for the
  // model-load failure screen.
  const [loadKey, setLoadKey] = useState(0);
  const tfModel = useTensorflowModel(
    require('../assets/models/camo_int8.tflite')
  );
  const model = tfModel.state === 'loaded' ? tfModel.model : undefined;

  // Track camera-level errors (device errors, permission revocation, etc).
  const [cameraError, setCameraError] = useState<Error | null>(null);

  // Track frame-processor errors. Worklet thread can’t setState directly, so
  // we bridge through a JS callback. After N consecutive bad frames we
  // surface the ErrorScreen instead of spamming logcat forever.
  const [frameError, setFrameError] = useState<Error | null>(null);
  const reportFrameErrorJs = useMemo(
    () =>
      Worklets.createRunOnJS((msg: string) => {
        setFrameError(new Error(msg));
      }),
    []
  );

  // Frame-resize plugin: converts a YUV/RGB Frame into a uint8 RGB tensor
  // sized for the model. Required — passing the raw Frame to TFLite crashes
  // the worklet thread ("no ArrayBuffer attached").
  const { resize } = useResizePlugin();

  // Worklet-side counter — stop reporting after we’ve surfaced one error so
  // logcat doesn’t fill with the same message 8x/sec.
  const errorReportedShared = useSharedValue(false);

  const [confThreshold, setConfThreshold] = useState(0.35);
  const [enabled, setEnabled] = useState<Record<string, boolean>>({ camo: true });
  const [fps, setFps] = useState(0);
  const [layout, setLayout] = useState({ width: 0, height: 0 });

  const detectionsShared = useSharedValue<Detection[]>([]);
  const [detectionsState, setDetectionsState] = useState<Detection[]>([]);

  // Bridge worklet -> JS so React can render labels/count.
  const setDetectionsJs = useMemo(
    () => Worklets.createRunOnJS(setDetectionsState),
    []
  );
  const setFpsJs = useMemo(() => Worklets.createRunOnJS(setFps), []);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (!model) return;

      runAtTargetFps(TARGET_FPS, () => {
        'worklet';
        try {
          const t0 = Date.now();

          // 1. Resize the incoming Frame into a uint8 RGB tensor of shape
          //    [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3]. We use uint8 (not
          //    float32) because our TFLite model is INT8-quantized — uint8
          //    input matches the quantization op directly, no scale
          //    conversion needed.
          //
          //    NOTE: We pass `resized` directly to model.runSync — matching
          //    the official react-native-fast-tflite example. The pinned
          //    versions of vision-camera (4.5.3) + worklets-core (1.5.0) +
          //    fast-tflite (1.6.1) are the known-working combo for the
          //    "no ArrayBuffer attached" bug (see GH #2409, #3517). Adding
          //    a Uint8Array copy here only made things worse on worklet
          //    threads — leave it out.
          const resized = resize(frame, {
            scale: { width: MODEL_INPUT_SIZE, height: MODEL_INPUT_SIZE },
            pixelFormat: 'rgb',
            dataType: 'uint8',
          });

          // 2. Run TFLite forward pass.
          const outputs = model.runSync([resized]);
          const raw = outputs[0] as unknown as Float32Array;
          const shape =
            (model.outputs?.[0]?.shape as number[]) ??
            [1, 4 + CLASS_NAMES.length, 0];

          // 3. Decode YOLO output into pixel-space Detections.
          const dets = decodeYoloOutput(raw, shape, {
            classNames: CLASS_NAMES,
            confThreshold,
            iouThreshold: 0.45,
            modelInputSize: MODEL_INPUT_SIZE,
            displayWidth: layout.width,
            displayHeight: layout.height,
            enabled,
          });
          detectionsShared.value = dets;
          setDetectionsJs(dets);

          const dt = Date.now() - t0;
          if (dt > 0) setFpsJs(Math.round(1000 / dt));
        } catch (e) {
          // Worklet exceptions silently kill the frame processor and on some
          // devices crash the whole app. Catch, log once, and surface the
          // ErrorScreen via the JS bridge.
          // eslint-disable-next-line no-console
          console.error('[frameProcessor]', e);
          if (!errorReportedShared.value) {
            errorReportedShared.value = true;
            const msg = (e as Error)?.message ?? String(e);
            reportFrameErrorJs(msg);
          }
        }
      });
    },
    [model, confThreshold, enabled, layout, resize]
  );

  // 1. Permission gate
  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.muted}>Camera permission required.</Text>
        <TouchableOpacity style={styles.btn} onPress={requestPermission}>
          <Text style={styles.btnText}>Grant access</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // 2. No usable camera (rare — emulator without webcam, or hardware fault)
  if (!device) {
    return (
      <ErrorScreen
        title="No camera available"
        message="We couldn't find a back-facing camera on this device."
        hint="Camo Detector needs a rear camera to scan a scene. If you're on an emulator, try a physical device."
      />
    );
  }

  // 3a. Camera reported a runtime error
  if (cameraError) {
    return (
      <ErrorScreen
        title="Camera error"
        message={cameraError.message || 'The camera stopped working.'}
        hint="Close any other app that might be using the camera, then try again."
        details={`${cameraError.name}: ${cameraError.message}\n\n${cameraError.stack ?? ''}`}
        onRetry={() => setCameraError(null)}
      />
    );
  }

  // 3b. Frame processor blew up. Most often this is a resize/TFLite buffer
  //     issue specific to a device's camera pixel format.
  if (frameError) {
    const m = frameError.message.toLowerCase();
    const hint = m.includes('arraybuffer')
      ? 'A buffer-sharing issue with the camera frame processor. This usually clears on retry. If it persists, your device\'s camera may not support the expected pixel format.'
      : 'The on-device inference pipeline hit an error. Tap Try again to restart it.';
    return (
      <ErrorScreen
        title="Inference error"
        message={frameError.message}
        hint={hint}
        details={frameError.stack ?? ''}
        onRetry={() => {
          errorReportedShared.value = false;
          setFrameError(null);
          setLoadKey((k) => k + 1);
        }}
      />
    );
  }

  // 4. Model still loading
  if (tfModel.state === 'loading') {
    return (
      <View style={styles.center}>
        <ActivityIndicator color="#a3b18a" />
        <Text style={styles.muted}>Loading model…</Text>
      </View>
    );
  }

  // 5. Model load failure — show useful diagnostics + retry
  if (tfModel.state === 'error') {
    const err = tfModel.error as Error | undefined;
    const msg = err?.message ?? String(tfModel.error ?? 'Unknown error');
    const hint = inferLoadHint(msg);
    return (
      <ErrorScreen
        title="Failed to load model"
        message="The on-device camo detection model couldn't be initialized."
        hint={hint}
        details={`${err?.name ?? 'Error'}: ${msg}\n\n${err?.stack ?? '(no stack)'}`}
        onRetry={() => setLoadKey((k) => k + 1)}
      />
    );
  }

  return (
    <View
      key={loadKey}
      style={styles.root}
      onLayout={(e) => setLayout({ width: e.nativeEvent.layout.width, height: e.nativeEvent.layout.height })}
    >
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive
        frameProcessor={frameProcessor}
        pixelFormat="rgb"
        onError={(e) => setCameraError(e as unknown as Error)}
      />
      <BoundingBoxOverlay detections={detectionsState} />

      <View style={styles.hud}>
        <Text style={styles.hudText}>FPS: {fps}</Text>
        <Text style={styles.hudText}>Det: {detectionsState.length}</Text>
      </View>

      <View style={styles.controls}>
        <View style={styles.row}>
          <Toggle
            label="camo"
            color="#22c55e"
            on={enabled.camo}
            onPress={() => setEnabled((s) => ({ ...s, camo: !s.camo }))}
          />
          <View style={styles.betaBadge}>
            <Text style={styles.betaBadgeText}>BETA</Text>
          </View>
        </View>
        <View style={styles.row}>
          <Text style={styles.confLabel}>conf {confThreshold.toFixed(2)}</Text>
          <TouchableOpacity onPress={() => setConfThreshold((c) => Math.max(0.05, c - 0.05))}>
            <Text style={styles.confBtn}>−</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => setConfThreshold((c) => Math.min(0.95, c + 0.05))}>
            <Text style={styles.confBtn}>+</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

/**
 * Heuristic mapping from a TFLite load-error message to a friendly hint.
 * Keep this list short and honest — false leads waste a tester's time.
 */
function inferLoadHint(msg: string): string {
  const m = msg.toLowerCase();
  if (m.includes('not found') || m.includes('no such file') || m.includes('asset')) {
    return 'The model file is missing from the app bundle. Reinstall the APK — if it persists, the build skipped bundling the .tflite asset.';
  }
  if (m.includes('unsupported') && m.includes('op')) {
    return 'This device\'s TFLite runtime doesn\'t support an operator in the model. Try the latest beta APK, or open an issue with the technical details below.';
  }
  if (m.includes('memory') || m.includes('alloc')) {
    return 'The device may be low on memory. Close other apps and tap Try again.';
  }
  if (m.includes('delegate') || m.includes('gpu') || m.includes('nnapi')) {
    return 'A hardware accelerator failed. The app should fall back to CPU automatically — tap Try again. If it still fails, your device may not support the accelerator.';
  }
  return 'Tap Try again. If this keeps happening, file an issue and include the technical details below.';
}

function Toggle({ label, color, on, onPress }: { label: string; color: string; on: boolean; onPress: () => void }) {
  return (
    <TouchableOpacity
      onPress={onPress}
      style={[styles.toggle, { borderColor: color, backgroundColor: on ? color : 'transparent' }]}
    >
      <Text style={[styles.toggleText, { color: on ? 'black' : color }]}>{label}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: 'black' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: 'black' },
  muted: { color: '#aaa', marginVertical: 8 },
  btn: { paddingHorizontal: 18, paddingVertical: 10, backgroundColor: '#fff', borderRadius: 8 },
  btnText: { color: 'black', fontWeight: '600' },
  hud: {
    position: 'absolute',
    top: 50,
    left: 16,
    backgroundColor: 'rgba(0,0,0,0.55)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
  },
  hudText: { color: 'white', fontVariant: ['tabular-nums'], fontSize: 12 },
  controls: {
    position: 'absolute',
    bottom: 32,
    left: 16,
    right: 16,
    gap: 10,
  },
  row: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  toggle: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 999,
    borderWidth: 2,
  },
  toggleText: { fontWeight: '700' },
  confLabel: {
    color: 'white',
    backgroundColor: 'rgba(0,0,0,0.55)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
    fontVariant: ['tabular-nums'],
  },
  confBtn: {
    color: 'white',
    backgroundColor: 'rgba(255,255,255,0.15)',
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 8,
    fontSize: 16,
    fontWeight: '700',
  },
  betaBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 999,
    backgroundColor: 'rgba(255,255,255,0.15)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.35)',
  },
  betaBadgeText: {
    color: 'white',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1,
  },
});
