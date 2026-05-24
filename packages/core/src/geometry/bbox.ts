import type { BBox } from '../types.js';

/**
 * Convert `[x1, y1, x2, y2]` to `[x, y, w, h]` (top-left + size).
 */
export function xyxyToXywh(b: BBox): [number, number, number, number] {
  const [x1, y1, x2, y2] = b;

  return [x1, y1, x2 - x1, y2 - y1];
}

/**
 * Convert `[x, y, w, h]` (top-left + size) to `[x1, y1, x2, y2]`.
 * Exact inverse of {@link xyxyToXywh}.
 */
export function xywhToXyxy(b: BBox): [number, number, number, number] {
  const [x, y, w, h] = b;

  return [x, y, w + x, h + y];
}

/**
 * Convert `[x1, y1, x2, y2]` to `[cx, cy, w, h]` (center + size).
 */
export function xyxyToCxcywh(b: BBox): [number, number, number, number] {
  const [x1, y1, x2, y2] = b;

  return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1];
}

/**
 * Convert `[cx, cy, w, h]` (center + size) to `[x1, y1, x2, y2]`.
 * Exact inverse of {@link xyxyToCxcywh}.
 */
export function cxcywhToXyxy(b: BBox): [number, number, number, number] {
  const [cx, cy, w, h] = b;
  const halfW = 0.5 * w;
  const halfH = 0.5 * h;

  return [cx - halfW, cy - halfH, cx + halfW, cy + halfH];
}

/**
 * Convert `[x1, y1, x2, y2]` to `[cx, cy, a, h]` where `a = w / h`.
 * DeepSORT / ByteTrack / OC-SORT canonical state-vector format.
 *
 * Behavior at h == 0 is undefined (callers must filter degenerate boxes).
 */
export function xyxyToXyah(b: BBox): [number, number, number, number] {
  const [x1, y1, x2, y2] = b;

  return [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / (y2 - y1), y2 - y1];
}

/**
 * Convert `[cx, cy, a, h]` (center + aspect ratio + height) to `[x1, y1, x2, y2]`,
 * where `a = w / h`. Inverse of {@link xyxyToXyah} up to floating-point error.
 */
export function xyahToXyxy(b: BBox): [number, number, number, number] {
  const [cx, cy, a, h] = b;
  const halfW = 0.5 * a * h;
  const halfH = 0.5 * h;

  return [cx - halfW, cy - halfH, cx + halfW, cy + halfH];
}

/**
 * Area of a bbox in xyxy form. Returns 0 for degenerate boxes (x2 <= x1 or y2 <= y1).
 */
export function bboxArea(b: BBox): number {
  const [x1, y1, x2, y2] = b;
  const w = Math.max(0, x2 - x1);
  const h = Math.max(0, y2 - y1);

  return w * h;
}

/**
 * Clip a bbox to the inclusive image bounds `[0, 0, width, height]`.
 * Returned coordinates are clamped; the result may be degenerate (zero-area).
 */
export function clipBBox(
  b: BBox,
  width: number,
  height: number,
): [number, number, number, number] {
  const [x1, y1, x2, y2] = b;

  return [
    Math.min(width, Math.max(0, x1)),
    Math.min(height, Math.max(0, y1)),
    Math.min(width, Math.max(0, x2)),
    Math.min(height, Math.max(0, y2)),
  ];
}
