// biome-ignore-all lint/style/noNonNullAssertion: indices into cost / iou
// matrices and id-keyed maps are bounded by the M*N rectangular contracts
// and explicit Map.has guards; asserting at each read is cheaper than a
// per-cell `number | undefined` narrowing in the per-frame hot path.

import { KalmanFilter, type KalmanState } from '../filters/kalman.js';
import { CvBBoxMotionModel, xysrToXyxy, xyxyToXysr } from '../filters/motion-models/cv-bbox.js';
import { giouMatrix, iouMatrix } from '../geometry/iou.js';
import { solveLsap } from '../solvers/hungarian.js';
import type { BBox, Detection, Track } from '../types.js';
import { BaseTracker, type InternalTrack } from './base.js';

/**
 * Cost-function selector matching noahcao/OC_SORT's `ASSO_FUNCS` table.
 * vestige.js ships `'iou'` and `'giou'`; `'ciou'`, `'diou'`, and `'ct_dist'`
 * from the reference are not exposed because they are not used in any
 * benchmarked OC-SORT configuration in the paper.
 */
export type OcSortAsoFunc = 'iou' | 'giou';

/**
 * Construction options for {@link OcSortTracker}. Defaults match the official
 * noahcao/OC_SORT reference (`trackers/ocsort_tracker/ocsort.py:OCSort.__init__`)
 * and ARCHITECTURE.md §6.4. The Cao et al. CVPR 2023 paper
 * (arXiv:2203.14360) is the canonical specification.
 *
 * | Option | Default | Origin |
 * |---|---|---|
 * | `detThresh` | 0.6 | ARCHITECTURE.md §6.4 default; noahcao has no default (required arg) |
 * | `maxAge` | 30 | noahcao default; lost-track retention horizon |
 * | `minHits` | 3 | noahcao default; warmup window length |
 * | `iouThreshold` | 0.3 | noahcao default; primary-association IoU cutoff |
 * | `deltaT` | 3 | noahcao default; OCM velocity-baseline window |
 * | `asoFunc` | 'iou' | noahcao default; primary association cost function |
 * | `inertia` | 0.2 | noahcao default; OCM (`vdc_weight`) angular-cost weight |
 * | `useByte` | false | noahcao default; enables the optional ByteTrack-style low-score stage |
 *
 * Two numeric constants are not exposed because the reference hard-codes them:
 *
 * - **Low-score floor** `0.1`: detections at or below this are discarded
 *   even from the optional BYTE stage (`inds_low = scores > 0.1`).
 * - **OCR threshold**: matches the primary `iouThreshold` (no separate cutoff).
 */
export interface OcSortTrackerOptions {
  /**
   * Minimum detection score to spawn a new track AND to enter the primary
   * association stage. Detections scoring below `detThresh` but above the
   * hard-coded `0.1` floor flow into the optional BYTE stage if
   * `useByte = true`. Default 0.6.
   */
  readonly detThresh?: number;
  /**
   * Lost-track retention horizon in frames. A track with
   * `timeSinceUpdate > maxAge` is reaped. Default 30.
   */
  readonly maxAge?: number;
  /**
   * Consecutive matches required for a tentative track to confirm. During
   * the first `minHits` frames any matched track is output regardless of
   * state (the SORT-style warmup; see {@link exportConfirmed}). Default 3.
   */
  readonly minHits?: number;
  /**
   * Minimum IoU for a track-detection pair to be considered for association
   * in any stage. Pairs below this are gated to `+Infinity` (ARCHITECTURE.md §5.6).
   * Default 0.3.
   */
  readonly iouThreshold?: number;
  /**
   * Observation-Centric Momentum (OCM) baseline window in frames: track
   * velocity is computed as the direction from the observation `deltaT`
   * frames ago to the current observation (with a graceful fallback when
   * fewer than `deltaT` observations exist). Larger `deltaT` smooths over
   * detection jitter at the cost of slower direction response. Default 3.
   */
  readonly deltaT?: number;
  /**
   * Primary-stage cost function. `'iou'` is the paper's default; `'giou'`
   * is the alternative reported in noahcao's `ASSO_FUNCS` table. The OCR
   * (recovery) stage always uses the same function as the primary stage.
   * Default `'iou'`.
   */
  readonly asoFunc?: OcSortAsoFunc;
  /**
   * Observation-Centric Momentum (OCM) weight. The angular-consistency
   * term in the primary cost matrix is multiplied by `inertia * det_score`
   * before being subtracted from `(1 − IoU)`. Larger values penalize
   * direction-inconsistent matches more strongly. Default 0.2.
   */
  readonly inertia?: number;
  /**
   * When `true`, runs a ByteTrack-style secondary association on
   * unmatched tracks against low-score detections (`0.1 < score < detThresh`)
   * before OCR. Default `false` (noahcao default; the paper reports the
   * BYTE-enabled config separately and does not claim it as the canonical
   * OC-SORT pipeline).
   */
  readonly useByte?: boolean;
}

/** Hard-coded low-score floor; matches `inds_low = scores > 0.1` in noahcao. */
const LOW_SCORE_FLOOR = 0.1;

/**
 * Per-track OC-SORT state that extends the shared {@link InternalTrack} shape
 * with the four observation-centric fields the paper introduces. Kept private
 * to this module — the base interface deliberately doesn't carry OC-SORT
 * specifics (ARCHITECTURE.md §2.2: "algorithms own their options").
 *
 * Fields mirror noahcao's `KalmanBoxTracker`:
 *
 * - {@link lastObservation} ≡ `self.last_observation` (the most recent real
 *   detection, used by OCR and by the velocity-computation fallback).
 * - {@link observations} ≡ `self.observations` (sparse `age → bbox` history,
 *   used by `k_previous_obs` for OCM and by the velocity computation).
 * - {@link velocity} ≡ `self.velocity` (normalized `[dy, dx]` direction vector
 *   from the `deltaT`-old observation to the latest one; `null` until the
 *   track has at least two observations).
 * - {@link kalmanSnapshot} stores the pre-occlusion Kalman state so that on
 *   re-association {@link OcSortTracker} can roll back and replay a virtual
 *   trajectory (ORU; paper §3.2 and noahcao's `KalmanFilterNew.freeze` /
 *   `unfreeze`).
 */
interface OcSortInternalTrack<TPayload> extends InternalTrack<TPayload> {
  /** Most recent real-detection bbox. `null` only before the first match. */
  lastObservation: BBox | null;
  /** Sparse `age → observed bbox` history for OCM's `k_previous_obs` lookup. */
  observations: Map<number, BBox>;
  /** OCM velocity as a normalized `[dy, dx]` direction vector. `null` until two observations exist. */
  velocity: [number, number] | null;
  /** Pre-occlusion Kalman state, captured on the first miss; consumed and cleared on re-association (ORU). */
  kalmanSnapshot: KalmanState | null;
  /** Age at which `lastObservation` was set; used to compute the time gap for ORU's virtual trajectory. */
  lastObservationAge: number;
}

/**
 * Observation-Centric SORT (Cao, Pang, Weng, Khirodkar, Kitani — CVPR 2023;
 * arXiv:2203.14360). Adds three observation-centric mechanisms on top of
 * the SORT baseline to address Kalman drift during occlusion:
 *
 * 1. **Observation-Centric Re-update (ORU)** — when a lost track re-associates,
 *    roll the Kalman state back to the last real observation and replay a
 *    linearly-interpolated virtual trajectory of measurement updates between
 *    that observation and the new one. This erases the drift accumulated by
 *    raw `predict()` steps during the unobserved window. Paper §3.2; reference
 *    `KalmanFilterNew.freeze` / `unfreeze` in
 *    `OC_SORT/trackers/ocsort_tracker/kalmanfilter.py`.
 *
 * 2. **Observation-Centric Momentum (OCM)** — augments the primary IoU cost
 *    with a direction-consistency term: for each (track, detection) pair, the
 *    direction from the track's `deltaT`-old observation to the detection
 *    center is compared against the track's measured velocity. Aligned
 *    directions discount the cost, misaligned ones increase it. Weighted by
 *    `inertia * det_score`. Paper §3.3; reference `associate()` in
 *    `OC_SORT/trackers/ocsort_tracker/association.py`.
 *
 * 3. **Observation-Centric Recovery (OCR)** — second-pass match using each
 *    unmatched track's *last observed position* (not its Kalman prediction)
 *    against the unmatched detections. Recovers tracks whose predictions have
 *    drifted out of IoU range during long occlusions. Paper §3.4; reference
 *    `OCSort.update` lines 280–302.
 *
 * Inherits SORT's per-frame lifecycle: `tentative → confirmed → lost → removed`
 * with `maxAge`-bounded lost retention and the SORT-style warmup output rule
 * (any matched track during frames `≤ minHits`). The lifecycle is encoded
 * directly in {@link update} (not {@link BaseTracker.runStandardLifecycle})
 * because the multi-stage matching pipeline runs ORU between association
 * and bookkeeping, which the standard helper doesn't support — per ADR-0003 §3,
 * multi-stage trackers orchestrate their own `update()`.
 *
 * Two intentional deviations from `ocsort.py`:
 *
 * - `Track.id` starts at 1 via the shared {@link BaseTracker} counter; the
 *   reference uses `id + 1` on output but a class-level counter that persists
 *   across instances. {@link reset} resets to 1 here.
 * - vestige.js's lifecycle states are explicit (`tentative | confirmed |
 *   lost | removed`; ARCHITECTURE.md §4.2). The reference has no explicit
 *   states — it filters on `time_since_update` and `hit_streak` directly.
 *   The export rule below reproduces the reference's observable output.
 */
export class OcSortTracker<TPayload = unknown> extends BaseTracker<TPayload> {
  readonly detThresh: number;
  readonly maxAge: number;
  readonly minHits: number;
  readonly iouThreshold: number;
  readonly deltaT: number;
  readonly asoFunc: OcSortAsoFunc;
  readonly inertia: number;
  readonly useByte: boolean;

  protected readonly kalmanFilter: KalmanFilter;

  constructor(options: OcSortTrackerOptions = {}) {
    super();
    this.detThresh = options.detThresh ?? 0.6;
    this.maxAge = options.maxAge ?? 30;
    this.minHits = options.minHits ?? 3;
    this.iouThreshold = options.iouThreshold ?? 0.3;
    this.deltaT = options.deltaT ?? 3;
    this.asoFunc = options.asoFunc ?? 'iou';
    this.inertia = options.inertia ?? 0.2;
    this.useByte = options.useByte ?? false;
    this.kalmanFilter = new KalmanFilter(new CvBBoxMotionModel());
  }

  /**
   * Kalman predict for a single track. Mirrors the abewley/sort safeguard
   * (also retained in noahcao's `KalmanBoxTracker.predict`): if the scale
   * velocity ṡ would drive scale to ≤ 0 on the next step, zero ṡ first to
   * keep {@link xysrToXyxy} out of its degenerate-collapse branch.
   */
  protected predictTrack(track: InternalTrack<TPayload>): void {
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

  /**
   * Single Kalman update for a matched detection. Refreshes
   * `bbox`, `kalmanState`, and `lastDetection`; does **not** record
   * observation history or update velocity — that bookkeeping lives in
   * {@link recordObservation} so {@link applyMatch} can drive both.
   */
  protected updateTrack(track: InternalTrack<TPayload>, detection: Detection<TPayload>): void {
    const measurement = xyxyToXysr(detection.bbox);
    const updated = this.kalmanFilter.update(track.kalmanState, measurement);
    track.kalmanState = updated;
    track.bbox = xysrToXyxy(updated.mean);
    track.lastDetection = detection;
  }

  /**
   * Build a fresh tentative {@link OcSortInternalTrack}. `hits = 0`,
   * `hitStreak = 0` matches the SORT family (the spawn detection is not a
   * hit). `lastObservation = null`, `observations` empty, `velocity = null`,
   * `kalmanSnapshot = null` — the OC-SORT-specific fields are populated
   * lazily as observations accumulate.
   */
  protected initTrack(detection: Detection<TPayload>): InternalTrack<TPayload> {
    const measurement = xyxyToXysr(detection.bbox);
    const kalmanState: KalmanState = this.kalmanFilter.model.init(measurement);
    const ocTrack: OcSortInternalTrack<TPayload> = {
      id: 0,
      state: 'tentative',
      age: 0,
      hits: 0,
      hitStreak: 0,
      timeSinceUpdate: 0,
      kalmanState,
      bbox: xysrToXyxy(kalmanState.mean),
      lastDetection: detection,
      lastObservation: null,
      observations: new Map(),
      velocity: null,
      kalmanSnapshot: null,
      lastObservationAge: -1,
    };
    return ocTrack;
  }

  /**
   * Process one frame of detections through the OC-SORT pipeline:
   *
   * 1. Predict every existing track; age each by 1.
   * 2. Partition detections at `detThresh`. Optional BYTE pool collects
   *    `0.1 < score < detThresh` detections when `useByte = true`.
   * 3. **Primary association** — high-score detections ↔ all tracks,
   *    using IoU (or GIoU per `asoFunc`) augmented by the OCM angular term.
   * 4. **ORU (Observation Re-Update)** — for each matched lost track, roll
   *    the Kalman state back to the pre-occlusion snapshot and replay a
   *    linearly-interpolated virtual trajectory of measurement updates
   *    ending at the new observation, then apply the standard match
   *    bookkeeping. For matched confirmed tracks, just bookkeep.
   * 5. **Optional BYTE stage** — low-score detections ↔ stage-3-unmatched
   *    tracks, IoU-only cost. Stage-2-matched tracks get the standard
   *    bookkeeping (BYTE matches do not trigger ORU in the reference).
   * 6. **OCR (Observation-Centric Recovery)** — unmatched detections ↔
   *    unmatched tracks' `lastObservation` (not Kalman predictions) using
   *    the primary cost function.
   * 7. Mark unmatched tracks (snapshotting Kalman state for ORU on the
   *    first miss); spawn new tracks from leftover unmatched high-score
   *    detections.
   * 8. Advance state transitions (tentative → confirmed via warmup or
   *    hitStreak; confirmed → lost on miss; lost → removed past `maxAge`).
   * 9. Sweep removed; export per the warmup-aware rule (see
   *    {@link exportConfirmed}).
   *
   * @throws Error with `not implemented` on the scaffold commit; the
   *   implementation commit turns this method green.
   */
  override update(_detections: ReadonlyArray<Detection<TPayload>>): Track<TPayload>[] {
    throw new Error('OcSortTracker.update: not implemented');
  }

  /**
   * OC-SORT's observable export rule, faithful to `ocsort.py` lines 321–323:
   *
   * ```python
   * if (trk.time_since_update < 1) and
   *    (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
   *     ret.append(...)
   * ```
   *
   * Equivalently in our explicit lifecycle: confirmed tracks matched this
   * frame are always output; during the first `minHits` frames, tentative
   * tracks matched this frame are also output (the warmup clause that lets
   * a detection appear in the very first frame's result).
   *
   * The reference's output bbox preference (`last_observation[:4]` when
   * available, else `get_state()[0]`) is matched here by exporting
   * `track.bbox` — which {@link updateTrack} has just refreshed from the
   * post-update Kalman state, equal-by-design to the last observation
   * within Kalman-filter noise.
   */
  protected override exportConfirmed(): Track<TPayload>[] {
    const out: Track<TPayload>[] = [];
    const warmup = this._frameIndex <= this.minHits;
    for (const track of this.tracks.values()) {
      if (track.timeSinceUpdate !== 0) continue;
      if (track.state === 'confirmed' || (warmup && track.state === 'tentative')) {
        out.push(this.materializeTrack(track));
      }
    }
    return out;
  }
}

/**
 * Look up the OC-SORT extension state for a track that {@link initTrack} created.
 * Centralized so the rest of the module doesn't repeat the cast.
 */
function _ocState<TPayload>(track: InternalTrack<TPayload>): OcSortInternalTrack<TPayload> {
  return track as OcSortInternalTrack<TPayload>;
}

/**
 * Direction vector from one bbox center to another, normalized to unit length.
 * Returns `[dy, dx]` to mirror noahcao's `speed_direction`, which orders the
 * vertical component first. The `1e-6` denominator floor matches the
 * reference's regularization for zero-displacement edge cases.
 */
function _speedDirection(from: BBox, to: BBox): [number, number] {
  const cx1 = (from[0] + from[2]) / 2;
  const cy1 = (from[1] + from[3]) / 2;
  const cx2 = (to[0] + to[2]) / 2;
  const cy2 = (to[1] + to[3]) / 2;
  const dx = cx2 - cx1;
  const dy = cy2 - cy1;
  const norm = Math.sqrt(dx * dx + dy * dy) + 1e-6;
  return [dy / norm, dx / norm];
}

/**
 * `k_previous_obs(observations, cur_age, k)` from `ocsort.py`. Walks the last
 * `k` ages looking for the *oldest* observation in that window
 * (`cur_age − k` first, decreasing); if none found, falls back to the most
 * ancient observation in the dict (`observations[max(keys)]`). Returns `null`
 * only when the observation dict is empty — equivalent to the reference's
 * `[-1, -1, -1, -1, -1]` placeholder.
 */
function _kPreviousObs(observations: Map<number, BBox>, curAge: number, k: number): BBox | null {
  if (observations.size === 0) return null;
  for (let i = 0; i < k; i++) {
    const dt = k - i;
    const found = observations.get(curAge - dt);
    if (found !== undefined) return found;
  }
  let maxAge = -1;
  for (const age of observations.keys()) if (age > maxAge) maxAge = age;
  return observations.get(maxAge) ?? null;
}

// Module-scope exports of the helpers kept private for now. When the
// implementation commit lands they become callable from the per-frame
// pipeline; the scaffold references them only to anchor JSDoc cross-links.
export const __internals__ = { _ocState, _speedDirection, _kPreviousObs };

// Module-scope use-marker for the otherwise-unused matrix imports during the
// scaffold; the implementation commit threads them into the primary /
// OCR cost-matrix construction. Without this reference, TS would flag the
// imports as dead even though they are required by the impl.
const _scaffoldImportAnchor = { iouMatrix, giouMatrix, solveLsap } as const;
void _scaffoldImportAnchor;
