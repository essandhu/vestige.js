import type { EvalFrame } from './frames.js';

/** Options for {@link identity}. */
export interface IdentityOptions {
  /**
   * Minimum IoU for a gt ↔ tracker pair to count as a per-frame potential
   * match. Default 0.5, matching TrackEval's `Identity` metric default.
   */
  readonly simThreshold?: number;
}

/**
 * ID metrics (Ristani et al., ECCV Workshops 2016), computed the way
 * `JonathonLuiten/TrackEval` (`metrics/identity.py`) computes them.
 */
export interface IdentityResult {
  /** `idtp / (idtp + 0.5·idfn + 0.5·idfp)`. */
  readonly idf1: number;
  /** ID recall: `idtp / (idtp + idfn)`. */
  readonly idr: number;
  /** ID precision: `idtp / (idtp + idfp)`. 0 when the tracker output is empty. */
  readonly idp: number;
  /** True positive ID associations under the optimal trajectory assignment. */
  readonly idtp: number;
  /** Ground-truth detections not covered by the assigned tracker identity. */
  readonly idfn: number;
  /** Tracker detections not covered by the assigned ground-truth identity. */
  readonly idfp: number;
}

/**
 * Compute IDF1 / IDP / IDR over a sequence.
 *
 * Unlike CLEAR-MOT's per-frame greedy continuity, ID metrics solve one global
 * bipartite assignment between gt and tracker *trajectories* (each side padded
 * with dummy nodes so unmatched trajectories pay their full FN/FP cost), then
 * count per-frame agreements of the assigned pairs. A pair's per-frame
 * agreement requires IoU ≥ `simThreshold`. Minimizing `IDFN + IDFP` is
 * equivalent to maximizing IDTP; the assignment is solved with the in-tree
 * Jonker-Volgenant solver.
 *
 * Throws on a sequence with zero ground-truth detections
 * (`/ground[\s-]?truth/i`); see {@link indexSequence} for the shared
 * degenerate-input contract.
 */
export function identity(
  _frames: ReadonlyArray<EvalFrame>,
  _options?: IdentityOptions,
): IdentityResult {
  throw new Error('not implemented');
}
