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

  constructor(_options?: CvXyahOptions) {
    throw new Error('CvXyahMotionModel: not implemented');
  }

  processNoise(_stateMean: Float64Array): Float64Array {
    throw new Error('CvXyahMotionModel.processNoise: not implemented');
  }

  measurementNoise(_stateMean: Float64Array): Float64Array {
    throw new Error('CvXyahMotionModel.measurementNoise: not implemented');
  }

  /**
   * Initialize a state from a `[cx, cy, a, h]` measurement. Velocities start
   * at zero; initial covariance is the diagonal `diag(stdInit²)` from the
   * class docstring above.
   */
  init(_measurement: Float64Array): KalmanState {
    throw new Error('CvXyahMotionModel.init: not implemented');
  }
}
