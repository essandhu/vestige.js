import type { Detection } from '../types.js';
import { type AssociationResult, BaseTracker, type InternalTrack } from './base.js';

/**
 * Construction options for {@link SortTracker}. Defaults match
 * abewley/sort (`sort.py:Sort.__init__`) and the original SORT paper
 * (Bewley et al., ICIP 2016).
 *
 * | Option | Default | Origin |
 * |---|---|---|
 * | `maxAge` | 1 | sort.py default; the original paper makes no occlusion handling promises |
 * | `minHits` | 3 | sort.py default; "three consecutive detections" confirmation criterion |
 * | `iouThreshold` | 0.3 | sort.py default; IoU below this is treated as a non-match |
 *
 * See ARCHITECTURE.md §6.2 for the per-option semantics.
 */
export interface SortTrackerOptions {
  /** Lost tracks survive `timeSinceUpdate <= maxAge`; removed once strictly greater. Default 1. */
  readonly maxAge?: number;
  /** Consecutive matches required for a tentative track to become confirmed. Default 3. */
  readonly minHits?: number;
  /** Minimum IoU for a track-detection pair to be considered for association. Default 0.3. */
  readonly iouThreshold?: number;
}

/**
 * Simple Online and Realtime Tracking (Bewley, Ge, Ott, Ramos, Upcroft —
 * ICIP 2016; arXiv:1602.00763). The single-pass IoU + Hungarian baseline.
 *
 * Reference implementation: abewley/sort (`sort.py`). The TypeScript port
 * preserves the algorithm verbatim:
 *
 * - 7-d constant-velocity Kalman filter on `[cx, cy, s, r, ċx, ċy, ṡ]`
 *   (see {@link CvBBoxMotionModel}).
 * - Aspect ratio `r` is treated as constant — no velocity component.
 * - Cost matrix is `1 − IoU(predicted, detection)`; pairs below `iouThreshold`
 *   are gated to `+Infinity` so the Hungarian solver never picks them.
 * - Linear sum assignment via Jonker-Volgenant ({@link import('../solvers/hungarian.js').solveLsap}).
 *
 * Two intentional deviations from sort.py:
 *
 * - `Track.id` is assigned starting at 1 (sort.py starts an internal counter
 *   at 0 and outputs `id + 1`; the visible ids are the same).
 * - Per the architecture's explicit lifecycle states (ARCHITECTURE.md §4.2),
 *   tracks pass through `tentative → confirmed → lost → removed` explicitly.
 *   sort.py has no `lost` state — a confirmed track that misses one frame
 *   has `time_since_update = 1` and gets removed only when `> max_age`.
 *   This implementation calls the same intermediate state `lost`. The
 *   export rule below preserves sort.py's observable output exactly.
 *
 * The export rule reproduces sort.py's
 * `(time_since_update < 1) and (hit_streak >= min_hits or frame_count <= min_hits)`:
 * during the first `minHits` frames any matched track is output, after that
 * only confirmed tracks are. This is what lets the very first frame's
 * detections appear in the output without waiting `minHits` frames.
 */
export class SortTracker<TPayload = unknown> extends BaseTracker<TPayload> {
  /** Resolved IoU threshold; cached so association doesn't read the options object per-pair. */
  readonly iouThreshold: number;

  constructor(options: SortTrackerOptions = {}) {
    super({
      minHits: options.minHits ?? 3,
      maxAge: options.maxAge ?? 1,
    });
    this.iouThreshold = options.iouThreshold ?? 0.3;
  }

  protected predictTrack(_track: InternalTrack<TPayload>): void {
    throw new Error('SortTracker.predictTrack: not implemented');
  }

  protected updateTrack(_track: InternalTrack<TPayload>, _detection: Detection<TPayload>): void {
    throw new Error('SortTracker.updateTrack: not implemented');
  }

  protected initTrack(_detection: Detection<TPayload>): InternalTrack<TPayload> {
    throw new Error('SortTracker.initTrack: not implemented');
  }

  protected associate(
    _detections: ReadonlyArray<Detection<TPayload>>,
    _associableTracks: ReadonlyArray<InternalTrack<TPayload>>,
  ): AssociationResult {
    throw new Error('SortTracker.associate: not implemented');
  }
}
