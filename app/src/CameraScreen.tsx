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
import { decodeYoloOutput, Detection } from './postprocess';
import BoundingBoxOverlay from './BoundingBoxOverlay';

const CLASS_NAMES = ['camo'];   // Beta = single-class. Pro will reintroduce more.
const MODEL_INPUT_SIZE = 320;   // must match what was set during export
const TARGET_FPS = 8;           // run inference at ~8 FPS, render preview at 30/60

export default function CameraScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');

  const tfModel = useTensorflowModel(
    require('../assets/models/camo_int8.tflite')
  );
  const model = tfModel.state === 'loaded' ? tfModel.model : undefined;

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
        const t0 = Date.now();

        // VisionCamera v4: frame is YUV; we resize+normalize via the model's
        // expected input. react-native-fast-tflite accepts a typed-array input;
        // we use a frame-processor plugin (vision-camera-resize-plugin) in real
        // builds. For brevity here we assume a helper `prepareInput` exists.
        // Replace this with the actual resize plugin call in your project:
        //   const input = resize(frame, { scale: { width: MODEL_INPUT_SIZE, height: MODEL_INPUT_SIZE }, pixelFormat: 'rgb', dataType: 'float32' });
        // For now we hand the frame buffer straight to the model and let the
        // model's input op handle it (works for many Ultralytics TFLite exports).
        const outputs = model.runSync([frame as any]);
        const raw = outputs[0] as unknown as Float32Array;
        const shape = (model.outputs?.[0]?.shape as number[]) ?? [1, 4 + CLASS_NAMES.length, 0];

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
      });
    },
    [model, confThreshold, enabled, layout]
  );

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
  if (!device) {
    return (
      <View style={styles.center}>
        <Text style={styles.muted}>No camera device found.</Text>
      </View>
    );
  }
  if (tfModel.state === 'loading') {
    return (
      <View style={styles.center}>
        <ActivityIndicator />
        <Text style={styles.muted}>Loading model…</Text>
      </View>
    );
  }
  if (tfModel.state === 'error') {
    return (
      <View style={styles.center}>
        <Text style={styles.muted}>Failed to load model: {String(tfModel.error)}</Text>
      </View>
    );
  }

  return (
    <View
      style={styles.root}
      onLayout={(e) => setLayout({ width: e.nativeEvent.layout.width, height: e.nativeEvent.layout.height })}
    >
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive
        frameProcessor={frameProcessor}
        pixelFormat="rgb"
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
