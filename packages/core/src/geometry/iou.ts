import type { BBox } from '../types.js';

/**
 * Intersection-over-Union of two xyxy bboxes.
 * Range: [0, 1]. Returns 0 for disjoint boxes, 1 for identical positive-area boxes.
 * For degenerate boxes (zero or negative area) the result is 0.
 */
export function iou(_a: BBox, _b: BBox): number {
  throw new Error('not implemented');
}

/**
 * Generalized IoU (Rezatofighi et al., CVPR 2019).
 * Range: [-1, 1]. Equals IoU when one box contains the other; negative when boxes
 * are disjoint and the enclosing box is large relative to the union.
 */
export function giou(_a: BBox, _b: BBox): number {
  throw new Error('not implemented');
}

/**
 * Distance-IoU (Zheng et al., AAAI 2020).
 * DIoU = IoU - ρ²(center_a, center_b) / c², where c is the diagonal of the
 * smallest enclosing box.
 */
export function diou(_a: BBox, _b: BBox): number {
  throw new Error('not implemented');
}

/**
 * Complete-IoU (Zheng et al., AAAI 2020). Adds an aspect-ratio consistency
 * term on top of DIoU.
 */
export function ciou(_a: BBox, _b: BBox): number {
  throw new Error('not implemented');
}

/**
 * Batched IoU. Returns an M*N row-major Float64Array where
 * `out[i * N + j] === iou(preds[i], dets[j])`.
 *
 * Layout chosen for cache locality during cost-matrix construction.
 */
export function iouMatrix(_preds: ReadonlyArray<BBox>, _dets: ReadonlyArray<BBox>): Float64Array {
  throw new Error('not implemented');
}

/**
 * Batched GIoU. Same layout as {@link iouMatrix}.
 */
export function giouMatrix(_preds: ReadonlyArray<BBox>, _dets: ReadonlyArray<BBox>): Float64Array {
  throw new Error('not implemented');
}
