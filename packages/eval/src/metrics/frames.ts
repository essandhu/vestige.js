import type { BBox } from '../core.js';

/**
 * One frame of aligned ground truth and tracker output — the common input
 * shape consumed by every metric in this package (ARCHITECTURE.md §10.3).
 *
 * `gtIds`/`gtBoxes` and `trackIds`/`trackBoxes` are parallel arrays. Identity
 * values are opaque labels: they only need to be consistent within a sequence,
 * not dense or ordered. Frames with no ground truth or no tracker output use
 * empty arrays.
 */
export interface EvalFrame {
  /** Ground-truth object identities present in this frame. */
  readonly gtIds: ReadonlyArray<number>;
  /** Ground-truth boxes (xyxy), parallel to `gtIds`. */
  readonly gtBoxes: ReadonlyArray<BBox>;
  /** Tracker-assigned identities present in this frame. */
  readonly trackIds: ReadonlyArray<number>;
  /** Tracker boxes (xyxy), parallel to `trackIds`. */
  readonly trackBoxes: ReadonlyArray<BBox>;
}

/**
 * One frame after identity densification: ids remapped to `[0, numGtIds)` /
 * `[0, numTrackIds)`, plus the per-frame IoU similarity matrix.
 */
export interface IndexedFrame {
  /** Dense ground-truth ids present this frame. */
  readonly gt: Int32Array;
  /** Dense tracker ids present this frame. */
  readonly track: Int32Array;
  /** Row-major `gt.length × track.length` IoU matrix: `sim[i * track.length + j]`. */
  readonly sim: Float64Array;
}

/**
 * A whole sequence in the dense form the metric implementations operate on.
 * Mirrors the preprocessed representation used by `JonathonLuiten/TrackEval`
 * (`_BaseDataset.get_preprocessed_seq_data`), which is the canonical reference
 * for every metric in this package (CONTRIBUTING.md §4.2).
 */
export interface IndexedSequence {
  readonly frames: ReadonlyArray<IndexedFrame>;
  /** Number of distinct ground-truth identities in the sequence. */
  readonly numGtIds: number;
  /** Number of distinct tracker identities in the sequence. */
  readonly numTrackIds: number;
  /** Total ground-truth detections across all frames. */
  readonly numGtDets: number;
  /** Total tracker detections across all frames. */
  readonly numTrackDets: number;
  /** Per dense gt id: number of frames the id appears in. Length `numGtIds`. */
  readonly gtIdCounts: Float64Array;
  /** Per dense tracker id: number of frames the id appears in. Length `numTrackIds`. */
  readonly trackIdCounts: Float64Array;
}

/**
 * Densify a sequence of {@link EvalFrame}s: remap gt/tracker identities to
 * contiguous indices (in order of first appearance — deterministic, see
 * ARCHITECTURE.md §2.5) and precompute the per-frame IoU similarity matrices
 * shared by CLEAR-MOT, identity, and HOTA.
 *
 * Degenerate inputs:
 *
 * - A sequence with zero ground-truth detections throws
 *   (`/ground[\s-]?truth/i`) — evaluating against empty ground truth is a
 *   caller error, not a measurable score.
 * - A frame whose id and box arrays differ in length throws (`/parallel/i`).
 * - A repeated identity within a single frame throws (`/duplicate/i`).
 */
export function indexSequence(_frames: ReadonlyArray<EvalFrame>): IndexedSequence {
  throw new Error('not implemented');
}
