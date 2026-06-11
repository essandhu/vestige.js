import type { EvalFrame } from './frames.js';

/**
 * HOTA results (Luiten et al., IJCV 2020), computed the way
 * `JonathonLuiten/TrackEval` (`metrics/hota.py`) computes them — that is the
 * §4.2 reference implementation for this module.
 *
 * Every scalar is the mean of the corresponding per-alpha value over the 19
 * localization thresholds `alphas = 0.05, 0.10, …, 0.95`. The per-alpha
 * arrays are exposed so benchmark reports can show the full curve.
 */
export interface HotaResult {
  /** Higher Order Tracking Accuracy: mean over alphas of `√(DetA·AssA)`. */
  readonly hota: number;
  /** Detection accuracy: mean over alphas of `tp / (tp + fn + fp)`. */
  readonly deta: number;
  /** Association accuracy: mean over alphas of the TP-averaged `A(c)` score. */
  readonly assa: number;
  /** Detection recall: mean over alphas of `tp / (tp + fn)`. */
  readonly detRe: number;
  /** Detection precision: mean over alphas of `tp / (tp + fp)`. */
  readonly detPr: number;
  /** Association recall: mean over alphas of the TP-averaged gt-side coverage. */
  readonly assRe: number;
  /** Association precision: mean over alphas of the TP-averaged tracker-side coverage. */
  readonly assPr: number;
  /**
   * Localization accuracy: mean over alphas of mean-IoU-over-TPs. Follows
   * TrackEval's convention that an alpha with zero TPs contributes 1 (the
   * `np.maximum(1e-10, …)` guard in `hota.py`), so LocA stays a pure
   * localization signal and never double-counts detection failures.
   */
  readonly locA: number;
  /** The 19 localization thresholds, `(k+1)/20` for `k = 0…18`. */
  readonly alphas: ReadonlyArray<number>;
  /** Per-alpha HOTA, parallel to `alphas`. */
  readonly hotaPerAlpha: Float64Array;
  /** Per-alpha DetA, parallel to `alphas`. */
  readonly detaPerAlpha: Float64Array;
  /** Per-alpha AssA, parallel to `alphas`. */
  readonly assaPerAlpha: Float64Array;
  /** Per-alpha LocA, parallel to `alphas`. */
  readonly locAPerAlpha: Float64Array;
}

/**
 * Compute HOTA and its decomposition over a sequence.
 *
 * Matching follows TrackEval `hota.py`: a first pass accumulates a global
 * alignment score per (gt id, tracker id) pair; a second pass solves one
 * per-frame assignment maximizing `globalAlignment · IoU` (so association
 * context breaks per-frame ties), then each alpha counts the matched pairs
 * with `IoU ≥ alpha` as TPs. Association scores `A(c)` are computed per TP
 * from the per-pair match counts.
 *
 * Throws on a sequence with zero ground-truth detections
 * (`/ground[\s-]?truth/i`); see {@link indexSequence} for the shared
 * degenerate-input contract.
 */
export function hota(_frames: ReadonlyArray<EvalFrame>): HotaResult {
  throw new Error('not implemented');
}
