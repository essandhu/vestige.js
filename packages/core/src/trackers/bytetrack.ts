import { KalmanFilter } from '../filters/kalman.js';
import { CvXyahMotionModel } from '../filters/motion-models/cv-xyah.js';
import type { Detection, Track } from '../types.js';
import { type AssociationResult, BaseTracker, type InternalTrack } from './base.js';

/**
 * Construction options for {@link ByteTracker}. Defaults match the official
 * FoundationVision/ByteTrack reference (`yolox/tracker/byte_tracker.py`) and
 * the ECCV 2022 paper (Zhang et al., arXiv:2110.06864).
 *
 * | Option | Default | Origin |
 * |---|---|---|
 * | `trackThresh` | 0.5 | `args.track_thresh`; high/low score split |
 * | `trackBuffer` | 30 | `args.track_buffer`; lost-track retention horizon |
 * | `matchThresh` | 0.8 | `args.match_thresh`; stage-1 IoU-distance cutoff |
 * | `frameRate` | 30 | `frame_rate`; scales `trackBuffer` to actual frames |
 * | `mot20` | false | `args.mot20`; when false, stage 1 & 3 apply `fuse_score` |
 *
 * See ARCHITECTURE.md §6.3 for the per-option semantics and §10.1 for the
 * acceptance window vs. published numbers.
 *
 * Three numeric constants are not exposed because the reference hard-codes
 * them and varying them would diverge from the published algorithm:
 *
 * - **Low-score floor** `0.1`: detections below this are discarded entirely.
 * - **Stage-2 IoU-distance cutoff** `0.5`: low-score association threshold.
 * - **Stage-3 IoU-distance cutoff** `0.7`: unconfirmed-track association threshold.
 * - **`det_thresh = trackThresh + 0.1`**: minimum score to spawn a new track.
 * - **Duplicate-IoU cutoff** `0.15` distance (≡ IoU > 0.85): tracked/lost dedup.
 */
export interface ByteTrackerOptions {
  /**
   * Detections strictly greater than this go into stage 1; detections in
   * `(0.1, trackThresh]` go into stage 2; detections at or below `0.1` are
   * dropped. Default 0.5.
   */
  readonly trackThresh?: number;
  /**
   * Lost-track retention horizon in **reference frames** (i.e. for `frameRate=30`).
   * The effective `maxAge` is `Math.floor(frameRate / 30 * trackBuffer)`.
   * Default 30.
   */
  readonly trackBuffer?: number;
  /**
   * Stage-1 IoU-distance cutoff: a high-score detection must have
   * `1 − IoU(track, det) ≤ matchThresh` (i.e. `IoU ≥ 1 − matchThresh`) for the
   * Hungarian solver to consider it. Default 0.8 (so IoU ≥ 0.2).
   */
  readonly matchThresh?: number;
  /**
   * Multiplier on `trackBuffer` to translate it to actual frames; useful when
   * running on >30 FPS sources where 30 reference frames is less wall-clock
   * time than at 30 FPS. `maxAge = Math.floor(frameRate / 30 * trackBuffer)`.
   * Default 30 (no scaling).
   */
  readonly frameRate?: number;
  /**
   * When `false` (default), stage-1 and stage-3 cost matrices are multiplied
   * by `(1 − det_score)` per detection column via `fuse_score`, biasing the
   * matcher toward higher-confidence detections. When `true` (MOT20 dataset),
   * `fuse_score` is skipped because MOT20's heavy crowding makes
   * confidence-weighting hurt more than it helps. Matches `args.mot20`.
   * Default false.
   */
  readonly mot20?: boolean;
}

/**
 * ByteTrack (Zhang, Sun, Jiang, Yu, Weng, Yuan, Luo, Liu, Wang —
 * ECCV 2022; arXiv:2110.06864). Three-stage association over the
 * DeepSORT-style cv-xyah Kalman filter.
 *
 * Reference implementation: FoundationVision/ByteTrack
 * (`yolox/tracker/byte_tracker.py`). The TypeScript port preserves the
 * algorithm's three-stage structure:
 *
 * 1. **Stage 1** — high-score detections (`score > trackThresh`) are matched
 *    against the union of currently-confirmed tracks and currently-lost tracks
 *    (`strack_pool`) using `1 − IoU` cost optionally weighted by
 *    `fuse_score`. Cutoff: `matchThresh` (default 0.8).
 * 2. **Stage 2** — low-score detections (`0.1 < score ≤ trackThresh`) are
 *    matched against stage-1-unmatched **confirmed** tracks only (lost tracks
 *    excluded). Cutoff: 0.5 (hard-coded per the reference). `fuse_score` is
 *    not applied here.
 * 3. **Stage 3** — unmatched high-score detections from stage 1 are matched
 *    against tentative ("unconfirmed") tracks. Cutoff: 0.7 (hard-coded).
 *    `fuse_score` applied unless `mot20 = true`.
 *
 * Lifecycle deviations from {@link BaseTracker}'s default transitions:
 *
 * - **Frame-1 spawn → immediately `confirmed`.** Mirrors
 *   `STrack.activate(frame_id=1)` setting `is_activated = True` on frame 1.
 *   Frames 2+ spawn `tentative`.
 * - **Tentative tracks get exactly one chance.** A tentative track that
 *   isn't matched in stage 3 of its second frame is removed immediately —
 *   not retained for `maxAge` frames. ByteTrack's `mark_removed()` rule.
 * - **`maxAge` applies only to lost tracks.** Computed as
 *   `Math.floor(frameRate / 30 * trackBuffer)` (matches the reference's
 *   `buffer_size = int(frame_rate / 30.0 * track_buffer)` and
 *   `max_time_lost = buffer_size`).
 *
 * For this reason {@link update} is overridden in full rather than relying on
 * {@link BaseTracker.update}'s shared lifecycle. The abstract hooks
 * ({@link predictTrack}, {@link updateTrack}, {@link initTrack}) are still
 * implemented for parity with the base class contract, but {@link associate}
 * is unused — three-stage matching is inlined into {@link update}.
 *
 * One intentional deviation from `byte_tracker.py`:
 *
 * - `Track.id` is assigned starting at 1 via the shared {@link BaseTracker}
 *   counter (`nextId`). The reference uses a class-level `BaseTrack._count`
 *   that persists across `BYTETracker` instances; vestige.js scopes the
 *   counter per-instance so {@link reset} returns to id=1.
 */
export class ByteTracker<TPayload = unknown> extends BaseTracker<TPayload> {
  /** Resolved `trackThresh`. Cached so the per-frame hot path doesn't read options. */
  readonly trackThresh: number;
  /** Resolved `trackBuffer`. */
  readonly trackBuffer: number;
  /** Resolved `matchThresh`. */
  readonly matchThresh: number;
  /** Resolved `frameRate`. */
  readonly frameRate: number;
  /** Resolved `mot20`. */
  readonly mot20: boolean;
  /**
   * Minimum detection score required to spawn a new track.
   * Equals `trackThresh + 0.1` (per `BYTETracker.__init__:
   * self.det_thresh = args.track_thresh + 0.1`).
   */
  readonly detThresh: number;
  /**
   * Lost-track retention in actual frames. Equals
   * `Math.floor(frameRate / 30 * trackBuffer)`. Surfaced as `maxAge` on the
   * shared {@link BaseTrackerOptions}.
   */
  readonly maxAge: number;

  protected readonly kalmanFilter: KalmanFilter;

  constructor(options: ByteTrackerOptions = {}) {
    const trackThresh = options.trackThresh ?? 0.5;
    const trackBuffer = options.trackBuffer ?? 30;
    const matchThresh = options.matchThresh ?? 0.8;
    const frameRate = options.frameRate ?? 30;
    const mot20 = options.mot20 ?? false;
    const maxAge = Math.floor((frameRate / 30) * trackBuffer);

    // ByteTrack confirms a track on its first match-after-spawn (frames > 1) or
    // immediately on frame 1; either way the "consecutive matches required"
    // count is 1 in BaseTracker terms. The lifecycle override below carries the
    // remaining ByteTrack-specific transitions.
    super({ minHits: 1, maxAge });

    this.trackThresh = trackThresh;
    this.trackBuffer = trackBuffer;
    this.matchThresh = matchThresh;
    this.frameRate = frameRate;
    this.mot20 = mot20;
    this.detThresh = trackThresh + 0.1;
    this.maxAge = maxAge;
    this.kalmanFilter = new KalmanFilter(new CvXyahMotionModel());
  }

  /**
   * Kalman predict for a single track. For lost tracks the reference zeros
   * the height-velocity component before predict
   * (`STrack.predict: if state != Tracked: mean_state[7] = 0`), which keeps
   * the box from drifting vertically while the track is unobserved.
   */
  protected predictTrack(_track: InternalTrack<TPayload>): void {
    throw new Error('ByteTracker.predictTrack not implemented');
  }

  /**
   * Kalman update for a single matched track. The measurement is the
   * `[cx, cy, w/h, h]` form of the detection bbox (DeepSORT cv-xyah).
   * `track.bbox`, `track.kalmanState`, and `track.lastDetection` are all
   * refreshed in place.
   */
  protected updateTrack(_track: InternalTrack<TPayload>, _detection: Detection<TPayload>): void {
    throw new Error('ByteTracker.updateTrack not implemented');
  }

  /**
   * Build a fresh tentative {@link InternalTrack} from a first observation.
   * `state = 'tentative'`, `hits = 0`, `hitStreak = 0` — the spawn detection
   * is **not** counted as a hit (matches the rest of the family — see
   * {@link BaseTracker.initTrack} JSDoc). Frame-1 promotion to `'confirmed'`
   * is handled in {@link update} after this method returns.
   */
  protected initTrack(_detection: Detection<TPayload>): InternalTrack<TPayload> {
    throw new Error('ByteTracker.initTrack not implemented');
  }

  /**
   * Unused: ByteTrack's three-stage matching is inlined into {@link update}.
   * The {@link BaseTracker} contract requires this method, so it stays as a
   * never-called stub.
   */
  protected associate(
    _detections: ReadonlyArray<Detection<TPayload>>,
    _associableTracks: ReadonlyArray<InternalTrack<TPayload>>,
  ): AssociationResult {
    throw new Error('ByteTracker.associate is not used — see update() override');
  }

  /**
   * Process one frame of detections via the three-stage ByteTrack pipeline.
   * Returns the publicly-visible tracks (those `confirmed` and matched this
   * frame). See class docstring for the stage ordering and the lifecycle
   * deviations from {@link BaseTracker.update}.
   */
  override update(_detections: ReadonlyArray<Detection<TPayload>>): Track<TPayload>[] {
    throw new Error('ByteTracker.update not implemented');
  }
}
