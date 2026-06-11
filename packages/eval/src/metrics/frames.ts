import type { BBox } from '../core.js';
import { iouMatrix } from '../core.js';

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
export function indexSequence(frames: ReadonlyArray<EvalFrame>): IndexedSequence {
  const gtDense = new Map<number, number>();
  const trackDense = new Map<number, number>();
  const gtCounts: number[] = [];
  const trackCounts: number[] = [];
  let numGtDets = 0;
  let numTrackDets = 0;

  const indexed: IndexedFrame[] = [];
  for (let f = 0; f < frames.length; f++) {
    const frame = frames[f];
    if (frame === undefined) continue;
    const gt = densifyIds(frame.gtIds, frame.gtBoxes.length, gtDense, gtCounts, 'gt', f + 1);
    const track = densifyIds(
      frame.trackIds,
      frame.trackBoxes.length,
      trackDense,
      trackCounts,
      'tracker',
      f + 1,
    );
    numGtDets += gt.length;
    numTrackDets += track.length;
    indexed.push({ gt, track, sim: iouMatrix(frame.gtBoxes, frame.trackBoxes) });
  }

  if (numGtDets === 0) {
    throw new Error('sequence contains no ground-truth detections');
  }

  return {
    frames: indexed,
    numGtIds: gtCounts.length,
    numTrackIds: trackCounts.length,
    numGtDets,
    numTrackDets,
    gtIdCounts: Float64Array.from(gtCounts),
    trackIdCounts: Float64Array.from(trackCounts),
  };
}

function densifyIds(
  ids: ReadonlyArray<number>,
  numBoxes: number,
  dense: Map<number, number>,
  counts: number[],
  side: 'gt' | 'tracker',
  frameNo: number,
): Int32Array {
  if (ids.length !== numBoxes) {
    throw new Error(
      `frame ${frameNo}: ${side} ids and boxes are not parallel arrays (${ids.length} ids, ${numBoxes} boxes)`,
    );
  }

  const out = new Int32Array(ids.length);
  const seen = new Set<number>();
  for (let i = 0; i < ids.length; i++) {
    const id = ids[i] ?? 0;
    if (seen.has(id)) {
      throw new Error(`frame ${frameNo}: duplicate ${side} id ${id}`);
    }
    seen.add(id);

    let denseId = dense.get(id);
    if (denseId === undefined) {
      denseId = counts.length;
      dense.set(id, denseId);
      counts.push(0);
    }
    counts[denseId] = (counts[denseId] ?? 0) + 1;
    out[i] = denseId;
  }
  return out;
}
