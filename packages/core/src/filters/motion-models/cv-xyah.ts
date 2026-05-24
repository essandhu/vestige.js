// biome-ignore-all lint/style/noNonNullAssertion: measurement / state buffers
// are bounded by the model's measDim / stateDim contract; non-null asserting
// the fixed-offset reads is cheaper than a guard on each access.

import type { KalmanState, MotionModel } from '../kalman.js';

/**
 * Options for {@link CvXyahMotionModel}. Defaults match the DeepSORT reference
 * (`nwojke/deep_sort/kalman_filter.py`); ByteTrack and OC-SORT inherit the
 * same constants. Override only if you have a measured reason to.
 */
export interface CvXyahOptions {
  /** Position-noise weight; default `1 / 20`. */
  readonly stdWeightPosition?: number;
  /** Velocity-noise weight; default `1 / 160`. */
  readonly stdWeightVelocity?: number;
}

/**
 * DeepSORT-style constant-velocity motion model. State is 8-dimensional:
 *
 * ```
 *   [cx, cy, a, h, cẋ, cẏ, ȧ, ḣ]
 * ```
 *
 * where `cx, cy` are bbox center coordinates, `a = w / h` is aspect ratio, and
 * `h` is height. This is the canonical state vector for DeepSORT, ByteTrack,
 * and OC-SORT (ARCHITECTURE.md §5.5).
 *
 * The measurement vector is the first four components, i.e. `[cx, cy, a, h]`
 * (see {@link xyxyToXyah} in `geometry/bbox.ts`).
 *
 * Unlike SORT's cv-bbox model, both `Q` and `R` here are **functions of the
 * state**: their entries scale with the bbox height `h`. This reproduces
 * DeepSORT's scale-aware noise model — a tall bbox is more uncertain in
 * absolute pixel units than a short one. Specifically, with
 * `wp = stdWeightPosition` and `wv = stdWeightVelocity`:
 *
 * ```
 *   stdPos = [wp·h, wp·h, 1e-2, wp·h]
 *   stdVel = [wv·h, wv·h, 1e-5, wv·h]
 *   Q      = diag(stdPos² ++ stdVel²)
 *
 *   stdR   = [wp·h, wp·h, 1e-1, wp·h]
 *   R      = diag(stdR²)
 *
 *   stdInit = [2·wp·h, 2·wp·h, 1e-2, 2·wp·h, 10·wv·h, 10·wv·h, 1e-5, 10·wv·h]
 *   P_init  = diag(stdInit²)
 * ```
 *
 * The constants `1e-2`, `1e-5`, `1e-1` on the aspect-ratio dimension are
 * DeepSORT's "this is a unitless ratio, scale it differently" hack and are
 * faithfully reproduced.
 */
export class CvXyahMotionModel implements MotionModel {
  readonly stateDim = 8;
  readonly measDim = 4;
  readonly F: Float64Array;
  readonly H: Float64Array;

  readonly stdWeightPosition: number;
  readonly stdWeightVelocity: number;

  constructor(options?: CvXyahOptions) {
    this.stdWeightPosition = options?.stdWeightPosition ?? 1 / 20;
    this.stdWeightVelocity = options?.stdWeightVelocity ?? 1 / 160;

    const F = new Float64Array(64);
    for (let i = 0; i < 8; i++) F[i * 8 + i] = 1;
    for (let i = 0; i < 4; i++) F[i * 8 + (i + 4)] = 1;
    this.F = F;

    const H = new Float64Array(32);
    for (let i = 0; i < 4; i++) H[i * 8 + i] = 1;
    this.H = H;
  }

  processNoise(stateMean: Float64Array): Float64Array {
    const h = stateMean[3]!;
    const sp = this.stdWeightPosition * h;
    const sv = this.stdWeightVelocity * h;
    const m = new Float64Array(64);
    m[0] = sp * sp;
    m[9] = sp * sp;
    m[18] = 1e-4;
    m[27] = sp * sp;
    m[36] = sv * sv;
    m[45] = sv * sv;
    m[54] = 1e-10;
    m[63] = sv * sv;

    return m;
  }

  measurementNoise(stateMean: Float64Array): Float64Array {
    const h = stateMean[3]!;
    const sp = this.stdWeightPosition * h;
    const m = new Float64Array(16);
    m[0] = sp * sp;
    m[5] = sp * sp;
    m[10] = 1e-2;
    m[15] = sp * sp;

    return m;
  }

  /**
   * Initialize a state from a `[cx, cy, a, h]` measurement. Velocities start
   * at zero; initial covariance is the diagonal `diag(stdInit²)` from the
   * class docstring above.
   */
  init(measurement: Float64Array): KalmanState {
    const h = measurement[3]!;
    const sp = this.stdWeightPosition * h;
    const sv = this.stdWeightVelocity * h;

    const mean = new Float64Array(8);
    mean[0] = measurement[0]!;
    mean[1] = measurement[1]!;
    mean[2] = measurement[2]!;
    mean[3] = measurement[3]!;

    const covariance = new Float64Array(64);
    const stdPos = 2 * sp;
    const stdVel = 10 * sv;
    covariance[0] = stdPos * stdPos;
    covariance[9] = stdPos * stdPos;
    covariance[18] = 1e-4;
    covariance[27] = stdPos * stdPos;
    covariance[36] = stdVel * stdVel;
    covariance[45] = stdVel * stdVel;
    covariance[54] = 1e-10;
    covariance[63] = stdVel * stdVel;

    return { mean, covariance };
  }
}
