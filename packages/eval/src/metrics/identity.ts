import { solveLsap } from '../core.js';
import type { EvalFrame } from './frames.js';
import { indexSequence } from './frames.js';

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
  frames: ReadonlyArray<EvalFrame>,
  options?: IdentityOptions,
): IdentityResult {
  const threshold = options?.simThreshold ?? 0.5;
  const seq = indexSequence(frames);
  const numGt = seq.numGtIds;
  const numTr = seq.numTrackIds;

  // overlap[g·numTr + t] = frames where gt g and tracker t agree (IoU ≥ threshold).
  const overlap = new Float64Array(numGt * numTr);
  for (const frame of seq.frames) {
    const n = frame.track.length;
    for (let i = 0; i < frame.gt.length; i++) {
      const gtId = frame.gt[i] ?? 0;
      for (let j = 0; j < n; j++) {
        if ((frame.sim[i * n + j] ?? 0) >= threshold - Number.EPSILON) {
          const pair = gtId * numTr + (frame.track[j] ?? 0);
          overlap[pair] = (overlap[pair] ?? 0) + 1;
        }
      }
    }
  }

  // Trajectory-level bipartite assignment on a (numGt + numTr)² matrix:
  // rows = gt ids then per-tracker dummies, cols = tracker ids then per-gt
  // dummies. A real pair costs its joint FN + FP; a trajectory matched to its
  // own dummy pays its full det count; cross-dummy cells are forbidden so an
  // unmatched trajectory can't dodge its cost. Minimizing total cost is
  // equivalent to maximizing IDTP (Ristani et al. §3; TrackEval identity.py).
  const size = numGt + numTr;
  const cost = new Float64Array(size * size).fill(Number.POSITIVE_INFINITY);
  for (let g = 0; g < numGt; g++) {
    const gtCount = seq.gtIdCounts[g] ?? 0;
    for (let t = 0; t < numTr; t++) {
      const trCount = seq.trackIdCounts[t] ?? 0;
      cost[g * size + t] = gtCount + trCount - 2 * (overlap[g * numTr + t] ?? 0);
    }
    cost[g * size + numTr + g] = gtCount;
  }
  for (let t = 0; t < numTr; t++) {
    cost[(numGt + t) * size + t] = seq.trackIdCounts[t] ?? 0;
  }
  for (let t = 0; t < numTr; t++) {
    for (let g = 0; g < numGt; g++) {
      cost[(numGt + t) * size + numTr + g] = 0;
    }
  }

  const { rowToCol } = solveLsap(cost, size, size);
  let idtp = 0;
  for (let g = 0; g < numGt; g++) {
    const t = rowToCol[g] ?? -1;
    if (t >= 0 && t < numTr) idtp += overlap[g * numTr + t] ?? 0;
  }

  const idfn = seq.numGtDets - idtp;
  const idfp = seq.numTrackDets - idtp;
  return {
    idf1: (2 * idtp) / (2 * idtp + idfn + idfp),
    idr: idtp / seq.numGtDets,
    idp: seq.numTrackDets > 0 ? idtp / seq.numTrackDets : 0,
    idtp,
    idfn,
    idfp,
  };
}
