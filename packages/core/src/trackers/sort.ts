// biome-ignore-all lint/style/noNonNullAssertion: indices into the cost
// matrix are bounded by the M*N rectangular contract; non-null asserting
// the cost-matrix reads is cheaper than a guard on each access.

import { KalmanFilter, type KalmanState } from '../filters/kalman.js';
import { CvBBoxMotionModel, xysrToXyxy, xyxyToXysr } from '../filters/motion-models/cv-bbox.js';
import { iouMatrix } from '../geometry/iou.js';
import { solveLsap } from '../solvers/hungarian.js';
import type { Detection, Track } from '../types.js';
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
 *   are gated to `+Infinity` so the Hungarian solver never picks them. sort.py
 *   instead post-filters: it runs `linear_sum_assignment(-iou_matrix)` without
 *   gating and then drops matches whose IoU fell below the threshold. These
 *   strategies are observationally equivalent on the inputs that come up in
 *   practice (any track gated out of the optimal pre-gated assignment would
 *   also be post-filtered out of sort.py's optimal); the pre-gate convention
 *   is what ARCHITECTURE.md §5.6 mandates for the family.
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
  private readonly kalmanFilter: KalmanFilter;

  constructor(options: SortTrackerOptions = {}) {
    super({
      minHits: options.minHits ?? 3,
      maxAge: options.maxAge ?? 1,
    });
    this.iouThreshold = options.iouThreshold ?? 0.3;
    this.kalmanFilter = new KalmanFilter(new CvBBoxMotionModel());
  }

  protected predictTrack(track: InternalTrack<TPayload>): void {
    // abewley/sort safeguard (sort.py:KalmanBoxTracker.predict): if ṡ would
    // drive scale to ≤ 0 on the next step, zero ṡ first. Keeps xysrToXyxy
    // out of its degenerate-collapse branch under runaway velocity estimates.
    const oldMean = track.kalmanState.mean;
    let stateForPredict = track.kalmanState;
    if (oldMean[2]! + oldMean[6]! <= 0) {
      const corrected = new Float64Array(oldMean);
      corrected[6] = 0;
      stateForPredict = { mean: corrected, covariance: track.kalmanState.covariance };
    }
    const next = this.kalmanFilter.predict(stateForPredict);
    track.kalmanState = next;
    track.bbox = xysrToXyxy(next.mean);
  }

  protected updateTrack(track: InternalTrack<TPayload>, detection: Detection<TPayload>): void {
    const measurement = xyxyToXysr(detection.bbox);
    const updated = this.kalmanFilter.update(track.kalmanState, measurement);
    track.kalmanState = updated;
    track.bbox = xysrToXyxy(updated.mean);
    track.lastDetection = detection;
  }

  protected initTrack(detection: Detection<TPayload>): InternalTrack<TPayload> {
    // hits = 0, hitStreak = 0: the spawn detection is NOT a hit (matches
    // sort.py:KalmanBoxTracker.__init__). The first hit is recorded the next
    // time the track matches; without this, SortTracker would confirm a track
    // one frame earlier than sort.py.
    const measurement = xyxyToXysr(detection.bbox);
    const kalmanState: KalmanState = this.kalmanFilter.model.init(measurement);
    return {
      id: 0,
      state: 'tentative',
      age: 0,
      hits: 0,
      hitStreak: 0,
      timeSinceUpdate: 0,
      kalmanState,
      bbox: xysrToXyxy(kalmanState.mean),
      lastDetection: detection,
    };
  }

  protected associate(
    detections: ReadonlyArray<Detection<TPayload>>,
    associableTracks: ReadonlyArray<InternalTrack<TPayload>>,
  ): AssociationResult {
    const M = associableTracks.length;
    const N = detections.length;
    if (M === 0 || N === 0) {
      return {
        matched: [],
        unmatchedTracks: associableTracks.map((_, i) => i),
        unmatchedDetections: detections.map((_, i) => i),
      };
    }

    const preds = associableTracks.map((t) => t.bbox);
    const detBoxes = detections.map((d) => d.bbox);
    const iou = iouMatrix(preds, detBoxes);

    // Cost = 1 − IoU; gate pairs whose IoU is below threshold to +Infinity so
    // the Hungarian solver never selects them (ARCHITECTURE.md §5.6).
    const cost = new Float64Array(M * N);
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        const v = iou[i * N + j]!;
        cost[i * N + j] = v < this.iouThreshold ? Number.POSITIVE_INFINITY : 1 - v;
      }
    }

    const { rowToCol } = solveLsap(cost, M, N);
    const matched: Array<[number, number]> = [];
    const matchedTracks = new Uint8Array(M);
    const matchedDets = new Uint8Array(N);
    for (let i = 0; i < M; i++) {
      const j = rowToCol[i]!;
      if (j !== -1) {
        matched.push([i, j]);
        matchedTracks[i] = 1;
        matchedDets[j] = 1;
      }
    }
    const unmatchedTracks: number[] = [];
    for (let i = 0; i < M; i++) if (matchedTracks[i] === 0) unmatchedTracks.push(i);
    const unmatchedDetections: number[] = [];
    for (let j = 0; j < N; j++) if (matchedDets[j] === 0) unmatchedDetections.push(j);

    return { matched, unmatchedTracks, unmatchedDetections };
  }

  /**
   * sort.py's observable export rule:
   *
   * ```python
   * (time_since_update < 1) and (hit_streak >= min_hits or frame_count <= min_hits)
   * ```
   *
   * In our explicit lifecycle: confirmed tracks matched this frame are always
   * output; during the first `minHits` frames, *tentative* tracks matched
   * this frame are also output (the "warmup" clause). This is what lets a
   * detection appear in the very first frame's result without waiting
   * `minHits` consecutive frames for confirmation.
   */
  protected override exportConfirmed(): Track<TPayload>[] {
    const out: Track<TPayload>[] = [];
    const warmup = this._frameIndex <= this.options.minHits;
    for (const track of this.tracks.values()) {
      if (track.timeSinceUpdate !== 0) continue;
      if (track.state === 'confirmed' || (warmup && track.state === 'tentative')) {
        out.push(this.materializeTrack(track));
      }
    }
    return out;
  }
}
