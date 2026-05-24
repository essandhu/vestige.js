// biome-ignore-all lint/style/noNonNullAssertion: measurement / state buffers
// are bounded by the model's measDim / stateDim contract; non-null asserting
// the fixed-offset reads is cheaper than a guard on each access.

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

  private readonly _Q: Float64Array;
  private readonly _R: Float64Array;
  private readonly _initCov: Float64Array;

  constructor() {
    const F = new Float64Array(49);
    for (let i = 0; i < 7; i++) F[i * 7 + i] = 1;
    F[0 * 7 + 4] = 1;
    F[1 * 7 + 5] = 1;
    F[2 * 7 + 6] = 1;
    this.F = F;

    const H = new Float64Array(28);
    for (let i = 0; i < 4; i++) H[i * 7 + i] = 1;
    this.H = H;

    this._Q = diag7(1, 1, 1, 1, 1e-2, 1e-2, 1e-4);
    this._R = diag4(1, 1, 10, 10);
    this._initCov = diag7(10, 10, 10, 10, 1e4, 1e4, 1e4);
  }

  processNoise(_stateMean: Float64Array): Float64Array {
    return this._Q;
  }

  measurementNoise(_stateMean: Float64Array): Float64Array {
    return this._R;
  }

  /**
   * Initialize a state from a `[cx, cy, s, r]` measurement. Velocities start
   * at zero; initial covariance puts large uncertainty on the velocity block
   * so the first few updates can move the estimate freely.
   */
  init(measurement: Float64Array): KalmanState {
    const mean = new Float64Array(7);
    mean[0] = measurement[0]!;
    mean[1] = measurement[1]!;
    mean[2] = measurement[2]!;
    mean[3] = measurement[3]!;

    return { mean, covariance: new Float64Array(this._initCov) };
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
export function xyxyToXysr(bbox: readonly [number, number, number, number]): Float64Array {
  const [x1, y1, x2, y2] = bbox;
  const w = x2 - x1;
  const h = y2 - y1;

  return new Float64Array([x1 + w / 2, y1 + h / 2, w * h, w / h]);
}

/**
 * Inverse of {@link xyxyToXysr}: convert `[cx, cy, s, r]` back into
 * `[x1, y1, x2, y2]`. Width is recovered as `sqrt(s * r)` and height as
 * `s / w`. Negative or zero `s` from a runaway Kalman state collapses to a
 * degenerate box at the center — this is a defensive extension over
 * abewley/sort's `convert_x_to_bbox`, which propagates NaN for `s < 0`.
 */
export function xysrToXyxy(state: Float64Array): [number, number, number, number] {
  const cx = state[0]!;
  const cy = state[1]!;
  const s = state[2]!;
  const r = state[3]!;
  if (s <= 0) return [cx, cy, cx, cy];
  const w = Math.sqrt(s * r);
  const h = s / w;
  const halfW = w / 2;
  const halfH = h / 2;

  return [cx - halfW, cy - halfH, cx + halfW, cy + halfH];
}

function diag7(
  a: number,
  b: number,
  c: number,
  d: number,
  e: number,
  f: number,
  g: number,
): Float64Array {
  const m = new Float64Array(49);
  m[0] = a;
  m[8] = b;
  m[16] = c;
  m[24] = d;
  m[32] = e;
  m[40] = f;
  m[48] = g;

  return m;
}

function diag4(a: number, b: number, c: number, d: number): Float64Array {
  const m = new Float64Array(16);
  m[0] = a;
  m[5] = b;
  m[10] = c;
  m[15] = d;

  return m;
}
