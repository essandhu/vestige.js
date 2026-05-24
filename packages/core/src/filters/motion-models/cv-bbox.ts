import type { KalmanState, MotionModel } from '../kalman.js';

/**
 * SORT-style constant-velocity motion model. State is 7-dimensional:
 *
 * ```
 *   [cx, cy, s, r, cẋ, cẏ, ṡ]
 * ```
 *
 * where `cx, cy` are bbox center coordinates, `s = w * h` is area, and
 * `r = w / h` is aspect ratio. Aspect ratio is treated as constant (no
 * velocity component for `r`), which matches the original SORT formulation
 * (Bewley et al., ICIP 2016) — the paper assumes pedestrian aspect ratios
 * change slowly enough that modelling their velocity isn't worth the noise.
 *
 * The measurement vector is the first four components of the state, i.e.
 * `[cx, cy, s, r]` — the projection of a bbox into the model's coordinate frame.
 * See {@link xyxyToXysr} for the canonical conversion from the public bbox
 * format.
 *
 * Process and measurement noise are state-independent constants matching the
 * abewley/sort reference (`sort.py:KalmanBoxTracker.__init__`):
 *
 * - `Q = diag(1, 1, 1, 1, 1e-2, 1e-2, 1e-4)`
 * - `R = diag(1, 1, 10, 10)`
 * - initial `P = diag(10, 10, 10, 10, 1e4, 1e4, 1e4)`
 */
export class CvBBoxMotionModel implements MotionModel {
  readonly stateDim = 7;
  readonly measDim = 4;
  readonly F: Float64Array;
  readonly H: Float64Array;

  constructor() {
    throw new Error('CvBBoxMotionModel: not implemented');
  }

  processNoise(_stateMean: Float64Array): Float64Array {
    throw new Error('CvBBoxMotionModel.processNoise: not implemented');
  }

  measurementNoise(_stateMean: Float64Array): Float64Array {
    throw new Error('CvBBoxMotionModel.measurementNoise: not implemented');
  }

  /**
   * Initialize a state from a `[cx, cy, s, r]` measurement. Velocities start
   * at zero; initial covariance puts large uncertainty on the velocity block
   * so the first few updates can move the estimate freely.
   */
  init(_measurement: Float64Array): KalmanState {
    throw new Error('CvBBoxMotionModel.init: not implemented');
  }
}

/**
 * Convert a `[x1, y1, x2, y2]` bbox into SORT's `[cx, cy, s, r]` measurement
 * frame, where `s = w * h` is area and `r = w / h` is aspect ratio.
 *
 * Behavior at `h == 0` is undefined (the result will contain `Infinity` for `r`);
 * callers filter degenerate boxes upstream, matching the rest of the geometry
 * module's contract (CONTRIBUTING.md §3.3).
 */
export function xyxyToXysr(_bbox: readonly [number, number, number, number]): Float64Array {
  throw new Error('xyxyToXysr: not implemented');
}

/**
 * Inverse of {@link xyxyToXysr}: convert `[cx, cy, s, r]` back into
 * `[x1, y1, x2, y2]`. Width is recovered as `sqrt(s * r)` and height as
 * `sqrt(s / r)`. Both `s` and `r` must be positive; negative `s` from a
 * runaway Kalman state is treated as zero (returns a degenerate box at the
 * center), matching abewley/sort's `convert_x_to_bbox`.
 */
export function xysrToXyxy(_state: Float64Array): [number, number, number, number] {
  throw new Error('xysrToXyxy: not implemented');
}
