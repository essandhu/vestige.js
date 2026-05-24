// biome-ignore-all lint/style/noNonNullAssertion: indices into cost / iou
// matrices and id-keyed maps are bounded by the M*N rectangular contracts
// and explicit Map.has guards; asserting at each read is cheaper than a
// per-cell `number | undefined` narrowing in the per-frame hot path.

import { KalmanFilter } from '../filters/kalman.js';
import { CvXyahMotionModel } from '../filters/motion-models/cv-xyah.js';
import { xyahToXyxy, xyxyToXyah } from '../geometry/bbox.js';
import { iouMatrix, iou as iouScalar } from '../geometry/iou.js';
import { solveLsap } from '../solvers/hungarian.js';
import type { BBox, Detection, Track } from '../types.js';
import { BaseTracker, type InternalTrack } from './base.js';

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
 * Five numeric constants are not exposed because the reference hard-codes
 * them and varying them would diverge from the published algorithm:
 *
 * - **Low-score floor** `0.1`: detections with `score ≤ 0.1` are discarded.
 * - **Stage-2 IoU-distance cutoff** `0.5`: low-score association threshold.
 * - **Stage-3 IoU-distance cutoff** `0.7`: unconfirmed-track association threshold.
 * - **`det_thresh = trackThresh + 0.1`**: minimum score to spawn a new track.
 * - **Duplicate-IoU cutoff** `0.15` distance (≡ IoU > 0.85): tracked/lost dedup.
 */
export interface ByteTrackerOptions {
  /**
   * Detections strictly greater than this go into stage 1; detections in
   * `(0.1, trackThresh)` go into stage 2; detections at or below `0.1` are
   * dropped. Note: a detection with `score === trackThresh` exactly is
   * dropped — the reference uses strict `<` and `>` comparisons against
   * `trackThresh`, leaving the boundary uncovered. Default 0.5.
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

/** Hard-coded score floor; detections at or below this are discarded entirely. */
const LOW_SCORE_FLOOR = 0.1;
/** Hard-coded stage-2 IoU-distance cutoff (reference: `matching.linear_assignment(thresh=0.5)`). */
const STAGE2_CUTOFF = 0.5;
/** Hard-coded stage-3 IoU-distance cutoff (reference: `matching.linear_assignment(thresh=0.7)`). */
const STAGE3_CUTOFF = 0.7;
/** Hard-coded duplicate-IoU-distance cutoff (reference: `pdist < 0.15`). */
const DUP_IOU_DISTANCE_CUTOFF = 0.15;
/** Reference frame rate that `trackBuffer` is calibrated against. */
const REFERENCE_FRAME_RATE = 30;

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
 * 2. **Stage 2** — low-score detections (`0.1 < score < trackThresh`) are
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
 * For this reason {@link update} writes its own per-frame pipeline rather
 * than calling {@link BaseTracker.runStandardLifecycle}. Per-track bookkeeping
 * still uses the shared primitives ({@link BaseTracker.applyMatch},
 * {@link BaseTracker.applyMiss}, {@link BaseTracker.spawnTrack},
 * {@link BaseTracker.sweepRemoved}) so the counter / state-transition logic
 * isn't duplicated.
 *
 * Two intentional deviations from `byte_tracker.py`:
 *
 * - `Track.id` is assigned starting at 1 via the shared {@link BaseTracker}
 *   counter (`nextId`). The reference uses a class-level `BaseTrack._count`
 *   that persists across `BYTETracker` instances; vestige.js scopes the
 *   counter per-instance so {@link reset} returns to id=1.
 * - vestige.js's lifecycle states are `tentative | confirmed | lost | removed`
 *   (ARCHITECTURE.md §4.2); the reference splits this across
 *   `state ∈ {New, Tracked, Lost, Removed}` plus the `is_activated` boolean.
 *   The mapping is: `tentative ≡ Tracked && !is_activated`, `confirmed ≡
 *   Tracked && is_activated`. Observable output is identical.
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
    const frameRate = options.frameRate ?? REFERENCE_FRAME_RATE;
    const mot20 = options.mot20 ?? false;
    const maxAge = Math.floor((frameRate / REFERENCE_FRAME_RATE) * trackBuffer);

    super();

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
  protected predictTrack(track: InternalTrack<TPayload>): void {
    let stateForPredict = track.kalmanState;
    if (track.state === 'lost') {
      const corrected = new Float64Array(track.kalmanState.mean);
      corrected[7] = 0;
      stateForPredict = { mean: corrected, covariance: track.kalmanState.covariance };
    }
    const next = this.kalmanFilter.predict(stateForPredict);
    track.kalmanState = next;
    track.bbox = xyahToXyxy([next.mean[0]!, next.mean[1]!, next.mean[2]!, next.mean[3]!]);
  }

  /**
   * Kalman update for a single matched track. The measurement is the
   * `[cx, cy, w/h, h]` form of the detection bbox (DeepSORT cv-xyah).
   * `track.bbox`, `track.kalmanState`, and `track.lastDetection` are all
   * refreshed in place.
   */
  protected updateTrack(track: InternalTrack<TPayload>, detection: Detection<TPayload>): void {
    const measurement = Float64Array.from(xyxyToXyah(detection.bbox));
    const updated = this.kalmanFilter.update(track.kalmanState, measurement);
    track.kalmanState = updated;
    track.bbox = xyahToXyxy([
      updated.mean[0]!,
      updated.mean[1]!,
      updated.mean[2]!,
      updated.mean[3]!,
    ]);
    track.lastDetection = detection;
  }

  /**
   * Build a fresh tentative {@link InternalTrack} from a first observation.
   * `state = 'tentative'`, `hits = 0`, `hitStreak = 0` — the spawn detection
   * is **not** counted as a hit (matches the rest of the family — see
   * {@link BaseTracker.initTrack} JSDoc). Frame-1 promotion to `'confirmed'`
   * is handled in {@link update} after this method returns.
   */
  protected initTrack(detection: Detection<TPayload>): InternalTrack<TPayload> {
    const measurement = Float64Array.from(xyxyToXyah(detection.bbox));
    const kalmanState = this.kalmanFilter.model.init(measurement);
    return {
      id: 0,
      state: 'tentative',
      age: 0,
      hits: 0,
      hitStreak: 0,
      timeSinceUpdate: 0,
      kalmanState,
      bbox: xyahToXyxy([
        kalmanState.mean[0]!,
        kalmanState.mean[1]!,
        kalmanState.mean[2]!,
        kalmanState.mean[3]!,
      ]),
      lastDetection: detection,
    };
  }

  /**
   * Process one frame of detections via the three-stage ByteTrack pipeline.
   * Returns the publicly-visible tracks (those `confirmed` and matched this
   * frame). See class docstring for the stage ordering and the lifecycle
   * deviations from {@link BaseTracker.update}.
   */
  override update(detections: ReadonlyArray<Detection<TPayload>>): Track<TPayload>[] {
    this._frameIndex++;

    // 1. Partition existing tracks. tentatives stay aside for stage 3;
    //    strackPool (confirmed + lost) feeds stages 1 & 2 after a predict step.
    const tentatives: InternalTrack<TPayload>[] = [];
    const strackPool: InternalTrack<TPayload>[] = [];
    for (const track of this.tracks.values()) {
      if (track.state === 'tentative') tentatives.push(track);
      else strackPool.push(track);
    }

    // 2. Predict confirmed + lost; age all surviving tracks (the spawn frame
    //    is handled below — new tracks created this frame start at age 0).
    for (const track of strackPool) {
      this.predictTrack(track);
      track.age++;
    }
    for (const track of tentatives) {
      // Tentatives are not predicted (faithful to the reference's `multi_predict`
      // running only on `strack_pool`). They still age.
      track.age++;
    }

    // 3. Partition detections by score. Per the reference:
    //      remain (high) = scores > trackThresh
    //      second (low)  = (scores > LOW_SCORE_FLOOR) AND (scores < trackThresh)
    //    Detections at scores ≤ LOW_SCORE_FLOOR or exactly == trackThresh
    //    are discarded (the boundary is deliberately not covered by either bin).
    const high: Detection<TPayload>[] = [];
    const low: Detection<TPayload>[] = [];
    for (const det of detections) {
      if (det.score > this.trackThresh) high.push(det);
      else if (det.score > LOW_SCORE_FLOOR && det.score < this.trackThresh) low.push(det);
    }

    // 4. Stage 1: high-score dets ↔ strackPool, optionally fuse_score-weighted.
    const stage1 = matchByIou(
      strackPool.map((t) => t.bbox),
      high,
      this.matchThresh,
      !this.mot20,
    );
    for (const [ti, di] of stage1.matched) {
      // applyMatch handles `updateTrack` + counter bumps + lost→confirmed.
      this.applyMatch(strackPool[ti]!, high[di]!);
    }

    // 5. Stage 2 pool: stage-1-unmatched but state==confirmed
    //    (the reference excludes lost tracks from stage 2 explicitly).
    const stage2Tracks: InternalTrack<TPayload>[] = [];
    for (const ti of stage1.unmatchedTracks) {
      const track = strackPool[ti]!;
      if (track.state === 'confirmed') stage2Tracks.push(track);
    }
    const stage2 = matchByIou(
      stage2Tracks.map((t) => t.bbox),
      low,
      STAGE2_CUTOFF,
      false, // fuse_score not applied in stage 2 (matches the reference)
    );
    for (const [ti, di] of stage2.matched) {
      // Stage-2 pool is confirmed-only, so applyMatch's lost→confirmed branch
      // is inert here; we just want the counter bumps + updateTrack.
      this.applyMatch(stage2Tracks[ti]!, low[di]!);
    }
    // Stage-2-unmatched confirmed tracks → lost.
    for (const ti of stage2.unmatchedTracks) {
      const track = stage2Tracks[ti]!;
      track.state = 'lost';
      this.applyMiss(track);
    }
    // Stage-1-unmatched lost tracks stay lost; their tsu advances.
    for (const ti of stage1.unmatchedTracks) {
      const track = strackPool[ti]!;
      if (track.state === 'lost') this.applyMiss(track);
    }

    // 6. Stage 3: stage-1-unmatched HIGH dets ↔ tentative tracks.
    const stage3Dets: Detection<TPayload>[] = [];
    for (const j of stage1.unmatchedDetections) stage3Dets.push(high[j]!);
    const stage3 = matchByIou(
      tentatives.map((t) => t.bbox),
      stage3Dets,
      STAGE3_CUTOFF,
      !this.mot20,
    );
    for (const [ti, di] of stage3.matched) {
      const track = tentatives[ti]!;
      this.applyMatch(track, stage3Dets[di]!);
      // Tentative → confirmed (one-chance rule; first match after spawn confirms).
      // applyMatch's lost→confirmed branch is inert on a tentative track, so the
      // explicit transition here is what actually promotes the track.
      track.state = 'confirmed';
    }
    // Stage-3-unmatched tentatives → removed immediately (mark_removed in the reference).
    for (const ti of stage3.unmatchedTracks) {
      tentatives[ti]!.state = 'removed';
    }

    // 7. Spawn new tracks for stage-3-unmatched high-score dets above detThresh.
    for (const j of stage3.unmatchedDetections) {
      const det = stage3Dets[j]!;
      if (det.score < this.detThresh) continue;
      const fresh = this.spawnTrack(det);
      // Frame-1 deviation: STrack.activate(frame_id=1) sets is_activated=True
      // immediately, so the track is observable on its spawn frame. Frames 2+
      // require a stage-3 match next frame before becoming confirmed.
      if (this._frameIndex === 1) {
        fresh.state = 'confirmed';
        fresh.hits = 1;
        fresh.hitStreak = 1;
      }
    }

    // 8. Reap lost tracks past max_time_lost.
    for (const track of this.tracks.values()) {
      if (track.state === 'lost' && track.timeSinceUpdate > this.maxAge) {
        track.state = 'removed';
      }
    }

    // 9. Sweep removed tracks out of the map.
    this.sweepRemoved();

    // 10. Deduplicate near-identical confirmed/lost pairs. When the IoU between
    //     a confirmed and a lost track exceeds 0.85 (i.e. iou_distance < 0.15),
    //     remove the younger of the two — this prevents the lost retention from
    //     spawning a phantom duplicate after a re-association. See the
    //     reference's `remove_duplicate_stracks`.
    this.removeDuplicates();

    // 11. Export confirmed tracks matched this frame (default rule on
    //     BaseTracker honors `state === 'confirmed' && timeSinceUpdate === 0`,
    //     which matches ByteTrack's `is_activated` output filter).
    return this.exportConfirmed();
  }

  /**
   * After-association dedup. Confirmed and lost tracks whose IoU exceeds 0.85
   * are treated as duplicates; the younger one (smaller `age`) is removed.
   * Ties prefer keeping the confirmed track over the lost one (consistent
   * with the reference's `else: dupa.append(p)` branch, where `stracksa` is
   * the confirmed list and `stracksb` the lost list — equal-time pairs see
   * the confirmed entry dropped, but in practice equal-time pairs would mean
   * two tracks spawned on the same frame, which the spawn logic doesn't
   * produce).
   */
  private removeDuplicates(): void {
    const confirmed: InternalTrack<TPayload>[] = [];
    const lost: InternalTrack<TPayload>[] = [];
    for (const track of this.tracks.values()) {
      if (track.state === 'confirmed') confirmed.push(track);
      else if (track.state === 'lost') lost.push(track);
    }
    if (confirmed.length === 0 || lost.length === 0) return;
    for (const c of confirmed) {
      for (const l of lost) {
        // 1 − IoU < 0.15  ≡  IoU > 0.85.
        if (1 - iouScalar(c.bbox, l.bbox) < DUP_IOU_DISTANCE_CUTOFF) {
          // Younger = smaller age; in ties, drop the confirmed one (reference branch).
          if (c.age > l.age) this.tracks.delete(l.id);
          else this.tracks.delete(c.id);
        }
      }
    }
  }
}

/**
 * IoU-distance association between predicted-track bboxes and detections.
 * Builds an M×N cost matrix where `cost[i, j] = 1 − IoU(tracks[i], dets[j])`
 * (optionally weighted via `fuse_score`), gates cells whose cost exceeds
 * `cutoff` to `+Infinity`, and runs Jonker-Volgenant assignment. Empty inputs
 * return an all-unmatched result without invoking the solver.
 *
 * Hoisted to module scope to keep the per-frame hot path free of closures
 * (CONTRIBUTING.md §3.4).
 */
function matchByIou<TPayload>(
  trackBoxes: ReadonlyArray<BBox>,
  detections: ReadonlyArray<Detection<TPayload>>,
  cutoff: number,
  applyFuseScore: boolean,
): { matched: Array<[number, number]>; unmatchedTracks: number[]; unmatchedDetections: number[] } {
  const M = trackBoxes.length;
  const N = detections.length;
  if (M === 0 || N === 0) {
    return {
      matched: [],
      unmatchedTracks: trackBoxes.map((_, i) => i),
      unmatchedDetections: detections.map((_, i) => i),
    };
  }
  const detBoxes = detections.map((d) => d.bbox);
  const iou = iouMatrix(trackBoxes, detBoxes);
  const cost = new Float64Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let c = 1 - iou[i * N + j]!;
      // fuse_score: cost' = 1 − (1 − cost) * det_score. Preserves the
      // gating sentinel because `1 − Infinity * positive` underflows to
      // `−Infinity`, and `1 − (−Infinity)` is `+Infinity` again.
      if (applyFuseScore) c = 1 - (1 - c) * detections[j]!.score;
      cost[i * N + j] = c > cutoff ? Number.POSITIVE_INFINITY : c;
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
