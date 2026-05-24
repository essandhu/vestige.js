/**
 * Generic linear Kalman filter over a pluggable {@link MotionModel}.
 * See ARCHITECTURE.md §5.5 for the design and §5.3 for the numerical-stability
 * rationale (Cholesky-based solve of the innovation covariance, not naive inversion).
 *
 * The filter itself is thin: predict and update are textbook linear-Gaussian
 * equations. All algorithm-specific behavior — the constant-velocity transition
 * matrix `F`, the measurement matrix `H`, and the state-dependent process /
 * measurement noise covariances `Q` and `R` — lives on the {@link MotionModel}.
 */

/**
 * Mean and covariance of a Kalman state.
 *
 * - `mean` is length `model.stateDim`.
 * - `covariance` is row-major (`stateDim × stateDim`).
 *
 * Both arrays are owned by the state and treated as read-only by the filter;
 * predict / update return fresh allocations rather than mutating in place.
 */
export interface KalmanState {
  readonly mean: Float64Array;
  readonly covariance: Float64Array;
}

/**
 * Projection of a state into measurement space: `H x` and `H P Hᵀ + R`.
 * Used both internally by {@link KalmanFilter.update} and externally for
 * association-time gating (e.g. Mahalanobis distance).
 */
export interface ProjectedState {
  readonly mean: Float64Array;
  readonly covariance: Float64Array;
}

/**
 * A linear-Gaussian motion model: state transition `F`, measurement matrix `H`,
 * process noise `Q(state)`, measurement noise `R(state)`, and a constructor
 * for a fresh state from a first measurement.
 *
 * Implementations are free to make `Q` and `R` state-dependent (DeepSORT scales
 * both by bbox height) or state-independent (SORT). The interface passes the
 * current state mean to both; state-independent implementations may ignore it.
 *
 * See `motion-models/cv-bbox.ts` for the SORT 7-d model and `motion-models/cv-xyah.ts`
 * for the DeepSORT 8-d model used by ByteTrack and OC-SORT.
 */
export interface MotionModel {
  /** Dimensionality of the state vector (e.g. 7 for SORT, 8 for DeepSORT). */
  readonly stateDim: number;
  /** Dimensionality of the measurement vector (4 for bbox trackers). */
  readonly measDim: number;
  /** State transition matrix, row-major (`stateDim × stateDim`). Constant. */
  readonly F: Float64Array;
  /** Measurement matrix, row-major (`measDim × stateDim`). Constant. */
  readonly H: Float64Array;

  /**
   * Process noise covariance `Q` for one predict step at the given state mean,
   * row-major (`stateDim × stateDim`). Implementations return a fresh allocation
   * the filter is free to consume; state-independent models may cache and return
   * the same buffer (the filter does not mutate it).
   */
  processNoise(stateMean: Float64Array): Float64Array;

  /**
   * Measurement noise covariance `R` for one update step at the given (predicted)
   * state mean, row-major (`measDim × measDim`). Same ownership convention as
   * {@link processNoise}.
   */
  measurementNoise(stateMean: Float64Array): Float64Array;

  /**
   * Initialize a fresh {@link KalmanState} from a first measurement of length
   * `measDim`. Position dimensions are set from the measurement; velocity
   * dimensions are zero. Initial covariance is large on velocities to reflect
   * "we have no idea how it's moving yet."
   */
  init(measurement: Float64Array): KalmanState;
}

/**
 * Linear Kalman filter parameterized by a {@link MotionModel}. The filter is
 * stateless across frames — the user passes the previous {@link KalmanState} in
 * and gets a new one out. This matches the "pure functional core, imperative
 * shell" principle (ARCHITECTURE.md §2.1): the trackers hold per-frame state,
 * the filter does not.
 *
 * Numerically, {@link update} uses a Cholesky-based solve of the innovation
 * covariance `S = H P Hᵀ + R` to compute the Kalman gain `K = P Hᵀ S⁻¹`,
 * rather than explicitly inverting `S`. This is the scipy / DeepSORT convention
 * and is what keeps the filter stable when detections are tightly clustered
 * (ARCHITECTURE.md §5.3).
 */
export class KalmanFilter {
  constructor(public readonly model: MotionModel) {}

  /**
   * One predict step: `x' = F x`, `P' = F P Fᵀ + Q(x)`.
   * Returns a fresh {@link KalmanState}; the input is not mutated.
   */
  predict(_state: KalmanState): KalmanState {
    throw new Error('KalmanFilter.predict: not implemented');
  }

  /**
   * One update step using the innovation `y = z - H x` and a Cholesky-based
   * solve for the Kalman gain. Returns a fresh {@link KalmanState}; the input
   * and the measurement are not mutated.
   *
   * @param state previous state (typically post-predict)
   * @param measurement length `model.measDim`
   */
  update(_state: KalmanState, _measurement: Float64Array): KalmanState {
    throw new Error('KalmanFilter.update: not implemented');
  }

  /**
   * Project a state into measurement space: returns `H x` and `S = H P Hᵀ + R(x)`.
   * Exposed publicly for association-time gating (Mahalanobis distance) and as
   * an introspection hook for tests.
   */
  project(_state: KalmanState): ProjectedState {
    throw new Error('KalmanFilter.project: not implemented');
  }
}
