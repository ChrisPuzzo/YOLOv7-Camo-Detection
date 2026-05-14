/**
 * postprocess.ts
 * --------------
 * Decodes raw YOLO output tensors into bounding boxes in screen coordinates.
 *
 * Supports both YOLO11/YOLOv8 (raw, NMS-needed) and YOLO26 (NMS-free) output shapes.
 * Ultralytics TFLite exports use shape [1, 4 + nc, N] (transposed) where:
 *   - rows 0..3 are box: cx, cy, w, h (normalized 0..1 to model input)
 *   - rows 4..(4+nc-1) are class scores
 *
 * For YOLO26 the model outputs already-deduplicated detections, but we still
 * apply a confidence filter for safety and a (cheap) NMS in case duplicates
 * leak through quantized inference.
 */

export type Detection = {
  classId: number;
  className: string;
  confidence: number;
  // box coords in DESTINATION (display) pixel space
  x: number;
  y: number;
  w: number;
  h: number;
};

export type PostprocessOpts = {
  /** Names indexed by class id (e.g. ['camo', 'gun']) */
  classNames: string[];
  /** Min confidence to keep a detection */
  confThreshold: number;
  /** IoU threshold for NMS */
  iouThreshold: number;
  /** Square model input size (px) */
  modelInputSize: number;
  /** Display canvas size to scale boxes into */
  displayWidth: number;
  displayHeight: number;
  /** Per-class enable flags, e.g. { camo: true, gun: true } */
  enabled?: Record<string, boolean>;
};

export function decodeYoloOutput(
  raw: Float32Array | number[],
  shape: number[], // typically [1, 4+nc, N] or [1, N, 4+nc]
  opts: PostprocessOpts
): Detection[] {
  'worklet';
  const { classNames, confThreshold, iouThreshold, modelInputSize, displayWidth, displayHeight, enabled } = opts;
  const nc = classNames.length;
  const stride = 4 + nc;

  // Detect orientation: Ultralytics TFLite usually exports [1, 4+nc, N].
  // If shape[1] === 4+nc, it's transposed; otherwise it's [1, N, 4+nc].
  let N: number;
  let transposed: boolean;
  if (shape.length === 3 && shape[1] === stride) {
    N = shape[2];
    transposed = true;
  } else if (shape.length === 3 && shape[2] === stride) {
    N = shape[1];
    transposed = false;
  } else {
    // Best-effort fallback
    N = Math.floor(raw.length / stride);
    transposed = false;
  }

  const sx = displayWidth / modelInputSize;
  const sy = displayHeight / modelInputSize;
  const cands: Detection[] = [];

  for (let i = 0; i < N; i++) {
    const at = (row: number) => (transposed ? raw[row * N + i] : raw[i * stride + row]);

    const cx = at(0);
    const cy = at(1);
    const w = at(2);
    const h = at(3);

    let bestClass = 0;
    let bestScore = -Infinity;
    for (let c = 0; c < nc; c++) {
      const s = at(4 + c);
      if (s > bestScore) {
        bestScore = s;
        bestClass = c;
      }
    }
    if (bestScore < confThreshold) continue;
    const name = classNames[bestClass] ?? String(bestClass);
    if (enabled && enabled[name] === false) continue;

    // Detect normalized vs pixel coordinates: most Ultralytics TFLite exports
    // emit normalized (0..1) coords. If values look like pixels, normalize.
    const norm = cx <= 1.5 && cy <= 1.5 && w <= 1.5 && h <= 1.5;
    const ncx = norm ? cx * modelInputSize : cx;
    const ncy = norm ? cy * modelInputSize : cy;
    const nw = norm ? w * modelInputSize : w;
    const nh = norm ? h * modelInputSize : h;

    cands.push({
      classId: bestClass,
      className: name,
      confidence: bestScore,
      x: (ncx - nw / 2) * sx,
      y: (ncy - nh / 2) * sy,
      w: nw * sx,
      h: nh * sy,
    });
  }

  return nms(cands, iouThreshold);
}

function iou(a: Detection, b: Detection): number {
  'worklet';
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const ua = a.w * a.h + b.w * b.h - inter;
  return ua > 0 ? inter / ua : 0;
}

function nms(dets: Detection[], iouThr: number): Detection[] {
  'worklet';
  const sorted = [...dets].sort((a, b) => b.confidence - a.confidence);
  const keep: Detection[] = [];
  for (const d of sorted) {
    if (keep.every((k) => k.classId !== d.classId || iou(k, d) < iouThr)) {
      keep.push(d);
    }
  }
  return keep;
}
