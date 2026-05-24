import type { BBox } from '../types.js';

const FOUR_OVER_PI_SQ = 4 / (Math.PI * Math.PI);

/**
 * Intersection-over-Union of two xyxy bboxes.
 * Range: [0, 1]. Returns 0 for disjoint boxes, 1 for identical positive-area boxes.
 * For degenerate boxes (zero or negative area) the result is 0.
 */
export function iou(a: BBox, b: BBox): number {
  const [ax1, ay1, ax2, ay2] = a;
  const [bx1, by1, bx2, by2] = b;

  const iw = Math.min(ax2, bx2) - Math.max(ax1, bx1);
  const ih = Math.min(ay2, by2) - Math.max(ay1, by1);
  if (iw <= 0 || ih <= 0) return 0;

  const inter = iw * ih;
  const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
  const areaB = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);
  const union = areaA + areaB - inter;

  return union > 0 ? inter / union : 0;
}

/**
 * Generalized IoU (Rezatofighi et al., CVPR 2019).
 * Range: [-1, 1]. Equals IoU when one box contains the other; negative when boxes
 * are disjoint and the enclosing box is large relative to the union.
 */
export function giou(a: BBox, b: BBox): number {
  const [ax1, ay1, ax2, ay2] = a;
  const [bx1, by1, bx2, by2] = b;

  const iw = Math.min(ax2, bx2) - Math.max(ax1, bx1);
  const ih = Math.min(ay2, by2) - Math.max(ay1, by1);
  const inter = iw > 0 && ih > 0 ? iw * ih : 0;

  const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
  const areaB = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);
  const union = areaA + areaB - inter;
  if (union <= 0) return 0;

  const ew = Math.max(ax2, bx2) - Math.min(ax1, bx1);
  const eh = Math.max(ay2, by2) - Math.min(ay1, by1);
  const enclose = ew * eh;
  if (enclose <= 0) return 0;

  return inter / union - (enclose - union) / enclose;
}

/**
 * Distance-IoU (Zheng et al., AAAI 2020).
 * DIoU = IoU - ρ²(center_a, center_b) / c², where c is the diagonal of the
 * smallest enclosing box.
 */
export function diou(a: BBox, b: BBox): number {
  const [ax1, ay1, ax2, ay2] = a;
  const [bx1, by1, bx2, by2] = b;

  const iw = Math.min(ax2, bx2) - Math.max(ax1, bx1);
  const ih = Math.min(ay2, by2) - Math.max(ay1, by1);
  const inter = iw > 0 && ih > 0 ? iw * ih : 0;

  const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
  const areaB = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);
  const union = areaA + areaB - inter;
  if (union <= 0) return 0;

  const iouVal = inter / union;

  const dx = (ax1 + ax2 - bx1 - bx2) * 0.5;
  const dy = (ay1 + ay2 - by1 - by2) * 0.5;
  const rhoSq = dx * dx + dy * dy;

  const ew = Math.max(ax2, bx2) - Math.min(ax1, bx1);
  const eh = Math.max(ay2, by2) - Math.min(ay1, by1);
  const cSq = ew * ew + eh * eh;
  if (cSq <= 0) return iouVal;

  return iouVal - rhoSq / cSq;
}

/**
 * Complete-IoU (Zheng et al., AAAI 2020). Adds an aspect-ratio consistency
 * term on top of DIoU.
 */
export function ciou(a: BBox, b: BBox): number {
  const [ax1, ay1, ax2, ay2] = a;
  const [bx1, by1, bx2, by2] = b;

  const iw = Math.min(ax2, bx2) - Math.max(ax1, bx1);
  const ih = Math.min(ay2, by2) - Math.max(ay1, by1);
  const inter = iw > 0 && ih > 0 ? iw * ih : 0;

  const aw = ax2 - ax1;
  const ah = ay2 - ay1;
  const bw = bx2 - bx1;
  const bh = by2 - by1;
  const areaA = aw > 0 && ah > 0 ? aw * ah : 0;
  const areaB = bw > 0 && bh > 0 ? bw * bh : 0;
  const union = areaA + areaB - inter;
  if (union <= 0) return 0;

  const iouVal = inter / union;

  const dx = (ax1 + ax2 - bx1 - bx2) * 0.5;
  const dy = (ay1 + ay2 - by1 - by2) * 0.5;
  const rhoSq = dx * dx + dy * dy;

  const ew = Math.max(ax2, bx2) - Math.min(ax1, bx1);
  const eh = Math.max(ay2, by2) - Math.min(ay1, by1);
  const cSq = ew * ew + eh * eh;
  if (cSq <= 0) return iouVal;

  const distTerm = rhoSq / cSq;
  // Short-circuit before computing alpha: alpha = v / ((1 - iou) + v) is 0/0 when v = 0
  // AND iou = 1 (identical boxes). Falling through would NaN the result.
  if (ah <= 0 || bh <= 0) return iouVal - distTerm;
  const v = FOUR_OVER_PI_SQ * (Math.atan(aw / ah) - Math.atan(bw / bh)) ** 2;
  if (v === 0) return iouVal - distTerm;

  const alpha = v / (1 - iouVal + v);

  return iouVal - distTerm - alpha * v;
}

/**
 * Batched IoU. Returns an M*N row-major Float64Array where
 * `out[i * N + j] === iou(preds[i], dets[j])`.
 *
 * Layout chosen for cache locality during cost-matrix construction.
 */
export function iouMatrix(preds: ReadonlyArray<BBox>, dets: ReadonlyArray<BBox>): Float64Array {
  const N = dets.length;
  const out = new Float64Array(preds.length * N);

  let row = 0;
  for (const a of preds) {
    let col = 0;
    for (const b of dets) {
      out[row + col] = iou(a, b);
      col++;
    }
    row += N;
  }

  return out;
}

/**
 * Batched GIoU. Same layout as {@link iouMatrix}.
 */
export function giouMatrix(preds: ReadonlyArray<BBox>, dets: ReadonlyArray<BBox>): Float64Array {
  const N = dets.length;
  const out = new Float64Array(preds.length * N);

  let row = 0;
  for (const a of preds) {
    let col = 0;
    for (const b of dets) {
      out[row + col] = giou(a, b);
      col++;
    }
    row += N;
  }

  return out;
}
