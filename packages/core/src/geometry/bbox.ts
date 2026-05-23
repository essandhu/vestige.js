import type { BBox } from '../types.js';

/**
 * Convert `[x1, y1, x2, y2]` to `[x, y, w, h]` (top-left + size).
 */
export function xyxyToXywh(_b: BBox): [number, number, number, number] {
  throw new Error('not implemented');
}

/**
 * Inverse of {@link xyxyToXywh}.
 */
export function xywhToXyxy(_b: BBox): [number, number, number, number] {
  throw new Error('not implemented');
}

/**
 * Convert `[x1, y1, x2, y2]` to `[cx, cy, w, h]` (center + size).
 */
export function xyxyToCxcywh(_b: BBox): [number, number, number, number] {
  throw new Error('not implemented');
}

/**
 * Inverse of {@link xyxyToCxcywh}.
 */
export function cxcywhToXyxy(_b: BBox): [number, number, number, number] {
  throw new Error('not implemented');
}

/**
 * Convert `[x1, y1, x2, y2]` to `[cx, cy, a, h]` where `a = w / h`.
 * DeepSORT / ByteTrack / OC-SORT canonical state-vector format.
 *
 * Behavior at h == 0 is undefined (callers must filter degenerate boxes).
 */
export function xyxyToXyah(_b: BBox): [number, number, number, number] {
  throw new Error('not implemented');
}

/**
 * Inverse of {@link xyxyToXyah}.
 */
export function xyahToXyxy(_b: BBox): [number, number, number, number] {
  throw new Error('not implemented');
}

/**
 * Area of a bbox in xyxy form. Returns 0 for degenerate boxes (x2 <= x1 or y2 <= y1).
 */
export function bboxArea(_b: BBox): number {
  throw new Error('not implemented');
}

/**
 * Clip a bbox to the inclusive image bounds `[0, 0, width, height]`.
 * Returned coordinates are clamped; the result may be degenerate (zero-area).
 */
export function clipBBox(
  _b: BBox,
  _width: number,
  _height: number,
): [number, number, number, number] {
  throw new Error('not implemented');
}
