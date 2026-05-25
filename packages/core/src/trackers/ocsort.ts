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
 *
 * Note: per `ocsort.py:OCSort.update`, the asoFunc selector is consulted
 * **only** by the BYTE stage (when `useByte = true`) and by OCR. The
 * primary stage's IoU cost is hard-coded to `iou_batch` in the reference;
 * this port preserves that.
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
 * | `asoFunc` | 'iou' | noahcao default; cost function for the BYTE and OCR stages |
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
   * Cost function for the BYTE and OCR stages. `'iou'` is the paper's
   * default; `'giou'` is the alternative reported in noahcao's `ASSO_FUNCS`
   * table. The primary stage always uses IoU (the reference's `associate()`
   * hard-codes `iou_batch`). Default `'iou'`.
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
   * `bbox`, `kalmanState`, and `lastDetection`. Does **not** record
   * observation history or update velocity — that bookkeeping lives in
   * {@link applyOcSortMatch} so {@link runORU} can be inserted between
   * the rollback and the final update.
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
   * Process one frame of detections through the OC-SORT pipeline. See the
   * class JSDoc for the algorithmic shape; numbered steps in the body
   * correspond to the per-frame stages described there.
   */
  override update(detections: ReadonlyArray<Detection<TPayload>>): Track<TPayload>[] {
    this._frameIndex++;

    // 1. Predict every existing track and age it. Snapshot must be taken
    //    BEFORE applyOcSortMiss (in step 7) to capture the post-predict state
    //    matching noahcao's `freeze()` semantics; that's handled there.
    const trackArr: OcSortInternalTrack<TPayload>[] = [];
    for (const t of this.tracks.values()) trackArr.push(t as OcSortInternalTrack<TPayload>);
    for (const t of trackArr) {
      this.predictTrack(t);
      t.age++;
    }

    // 2. Partition detections. The reference's three buckets:
    //      remain_inds = scores > det_thresh        → primary association
    //      inds_second = (scores > 0.1) AND (scores < det_thresh) → BYTE only
    //      everything else (≤ 0.1, or exactly == det_thresh) is discarded.
    //    Strict inequalities mirror noahcao verbatim — boundary detections
    //    are deliberately uncovered.
    const high: Detection<TPayload>[] = [];
    const low: Detection<TPayload>[] = [];
    for (const d of detections) {
      if (d.score > this.detThresh) high.push(d);
      else if (this.useByte && d.score > LOW_SCORE_FLOOR && d.score < this.detThresh) low.push(d);
    }

    // 3. Stage 1 — OCM-augmented IoU association over (all tracks × high dets).
    const stage1 = associatePrimary(trackArr, high, this.iouThreshold, this.inertia, this.deltaT);

    // 4. Apply matches. `applyOcSortMatch` handles ORU (for lost tracks with
    //    a snapshot), velocity update, and the standard match bookkeeping.
    for (const [ti, di] of stage1.matched) {
      this.applyOcSortMatch(trackArr[ti]!, high[di]!);
    }

    let stillUnmatchedTrackIdx = stage1.unmatchedTracks.slice();
    let stillUnmatchedDetIdx = stage1.unmatchedDetections.slice();

    // 5. (Optional) BYTE stage — low-score dets ↔ stage-1-unmatched tracks.
    //    No OCM, no fuse_score; just IoU/asoFunc-distance gated at iouThreshold.
    if (this.useByte && low.length > 0 && stillUnmatchedTrackIdx.length > 0) {
      const candTracks: OcSortInternalTrack<TPayload>[] = [];
      for (const ti of stillUnmatchedTrackIdx) candTracks.push(trackArr[ti]!);
      const stage2 = associateAsoFunc(
        candTracks.map((t) => t.bbox),
        low,
        this.iouThreshold,
        this.asoFunc,
      );
      const matchedTrackPositions = new Set<number>();
      for (const [ci, di] of stage2.matched) {
        const ti = stillUnmatchedTrackIdx[ci]!;
        this.applyOcSortMatch(trackArr[ti]!, low[di]!);
        matchedTrackPositions.add(ci);
      }
      const next: number[] = [];
      for (let i = 0; i < stillUnmatchedTrackIdx.length; i++) {
        if (!matchedTrackPositions.has(i)) next.push(stillUnmatchedTrackIdx[i]!);
      }
      stillUnmatchedTrackIdx = next;
    }

    // 6. OCR stage — match remaining unmatched HIGH dets against remaining
    //    unmatched tracks' *last observation* (not Kalman prediction). Recovers
    //    tracks whose Kalman state drifted out of IoU range during occlusion.
    if (stillUnmatchedDetIdx.length > 0 && stillUnmatchedTrackIdx.length > 0) {
      const eligible: { ti: number; lastObs: BBox }[] = [];
      for (const ti of stillUnmatchedTrackIdx) {
        const lo = trackArr[ti]!.lastObservation;
        if (lo !== null) eligible.push({ ti, lastObs: lo });
      }
      if (eligible.length > 0) {
        const ocrDets: Detection<TPayload>[] = [];
        for (const di of stillUnmatchedDetIdx) ocrDets.push(high[di]!);
        const stage3 = associateAsoFunc(
          eligible.map((e) => e.lastObs),
          ocrDets,
          this.iouThreshold,
          this.asoFunc,
        );
        const matchedTrackIds = new Set<number>();
        const matchedDetPositions = new Set<number>();
        for (const [ei, oi] of stage3.matched) {
          const ti = eligible[ei]!.ti;
          const di = stillUnmatchedDetIdx[oi]!;
          this.applyOcSortMatch(trackArr[ti]!, high[di]!);
          matchedTrackIds.add(ti);
          matchedDetPositions.add(oi);
        }
        stillUnmatchedTrackIdx = stillUnmatchedTrackIdx.filter((ti) => !matchedTrackIds.has(ti));
        const nextDets: number[] = [];
        for (let i = 0; i < stillUnmatchedDetIdx.length; i++) {
          if (!matchedDetPositions.has(i)) nextDets.push(stillUnmatchedDetIdx[i]!);
        }
        stillUnmatchedDetIdx = nextDets;
      }
    }

    // 7. Mark remaining unmatched tracks as missed. `applyOcSortMiss` captures
    //    the post-predict Kalman state as the ORU snapshot on the first miss
    //    of an unobserved run (when `tsu === 0` going into this miss and a
    //    `lastObservation` exists to roll back to).
    for (const ti of stillUnmatchedTrackIdx) this.applyOcSortMiss(trackArr[ti]!);

    // 8. Spawn fresh tracks for leftover unmatched HIGH-score detections.
    //    BYTE-pool leftovers do not spawn (noahcao convention; low-score dets
    //    are association-only). Score is already > detThresh by partition.
    for (const di of stillUnmatchedDetIdx) this.spawnTrack(high[di]!);

    // 9. Advance lifecycle. Cascade is intentional (each `if`, not `else if`):
    //    a tentative track that misses past `maxAge` reaches `removed` in a
    //    single pass, and a confirmed track that missed this frame falls to
    //    `lost` and then potentially to `removed` if `maxAge === 0`.
    for (const track of this.tracks.values()) {
      if (track.state === 'tentative') {
        if (track.timeSinceUpdate > this.maxAge) track.state = 'removed';
        else if (track.hitStreak >= this.minHits) track.state = 'confirmed';
      }
      if (track.state === 'confirmed' && track.timeSinceUpdate > 0) {
        track.state = 'lost';
      }
      if (track.state === 'lost' && track.timeSinceUpdate > this.maxAge) {
        track.state = 'removed';
      }
    }

    // 10. Sweep removed and export (warmup-aware; see {@link exportConfirmed}).
    this.sweepRemoved();
    return this.exportConfirmed();
  }

  /**
   * Apply a successful (track, detection) match with OC-SORT-specific bookkeeping:
   *
   * 1. If the track was previously unmatched and we have an ORU snapshot,
   *    roll the Kalman state back and replay a linearly-interpolated virtual
   *    trajectory ending at the new detection ({@link runORU}). This happens
   *    BEFORE the standard match-update so the post-update state has the
   *    drift correction baked in.
   * 2. Compute the new OCM velocity using the {@link _lookupVelocityPrev}
   *    lookup, which mirrors noahcao's "oldest observation in the last
   *    `deltaT` frames, falling back to `last_observation`" rule. Skipped on
   *    the first match (when `lastObservation` is still `null`).
   * 3. Standard match bookkeeping via {@link applyMatch}: KF update, counter
   *    bumps, `lost → confirmed` transition.
   * 4. Record the new observation in {@link observations} (keyed by current
   *    age) and refresh {@link lastObservation} / {@link lastObservationAge}.
   */
  private applyOcSortMatch(
    track: OcSortInternalTrack<TPayload>,
    detection: Detection<TPayload>,
  ): void {
    if (
      track.kalmanSnapshot !== null &&
      track.lastObservation !== null &&
      track.lastObservationAge >= 0
    ) {
      this.runORU(track, detection);
    }

    if (track.lastObservation !== null) {
      const prevObs = this._lookupVelocityPrev(track);
      track.velocity = _speedDirection(prevObs, detection.bbox);
    }

    this.applyMatch(track, detection);

    track.lastObservation = detection.bbox;
    track.observations.set(track.age, detection.bbox);
    track.lastObservationAge = track.age;
  }

  /**
   * Observation-Centric Re-update. Roll the Kalman state back to the
   * pre-occlusion snapshot and replay a linearly-interpolated virtual
   * trajectory of measurement updates between {@link lastObservation} and
   * the new detection. Mirrors `KalmanFilterNew.unfreeze` in
   * `OC_SORT/trackers/ocsort_tracker/kalmanfilter.py`:
   *
   * - The virtual trajectory has `timeGap = age - lastObservationAge` steps.
   * - At each step `i ∈ [0, timeGap)`, a virtual box is computed by linear
   *   interpolation from `lastObservation` (step 0) to `newDet.bbox`
   *   (step `timeGap`).
   * - The KF runs `update(virtualBox)` then `predict()`, except after the
   *   last step where `predict()` is skipped (control returns to
   *   {@link applyMatch}'s {@link updateTrack} call, which performs the final
   *   update on the same `newDet.bbox`; this re-applies the latest virtual
   *   update with the actual measurement, matching noahcao's behavior where
   *   the recursive `update(new_box)` call leaves `observed = True` and the
   *   outer `update(z)` call then proceeds with the standard KF math).
   * - The snapshot is consumed and cleared after rollback.
   */
  private runORU(track: OcSortInternalTrack<TPayload>, newDet: Detection<TPayload>): void {
    const snap = track.kalmanSnapshot!;
    track.kalmanSnapshot = null;

    const timeGap = track.age - track.lastObservationAge;
    if (timeGap <= 0) {
      // Safety: same-frame re-match shouldn't be reachable but if it is,
      // just restore the snapshot and let the regular update math run.
      track.kalmanState = snap;
      return;
    }

    track.kalmanState = snap;

    const [cx1, cy1, w1, h1] = _bboxCenterWH(track.lastObservation!);
    const [cx2, cy2, w2, h2] = _bboxCenterWH(newDet.bbox);
    const dcx = (cx2 - cx1) / timeGap;
    const dcy = (cy2 - cy1) / timeGap;
    const dw = (w2 - w1) / timeGap;
    const dh = (h2 - h1) / timeGap;

    for (let i = 0; i < timeGap; i++) {
      const cx = cx1 + (i + 1) * dcx;
      const cy = cy1 + (i + 1) * dcy;
      const w = w1 + (i + 1) * dw;
      const h = h1 + (i + 1) * dh;
      const hw = w / 2;
      const hh = h / 2;
      const virtualBox: BBox = [cx - hw, cy - hh, cx + hw, cy + hh];
      const measurement = xyxyToXysr(virtualBox);
      track.kalmanState = this.kalmanFilter.update(track.kalmanState, measurement);
      if (i !== timeGap - 1) {
        track.kalmanState = this.kalmanFilter.predict(track.kalmanState);
      }
    }
    track.bbox = xysrToXyxy(track.kalmanState.mean);
  }

  /**
   * Mirror of noahcao's velocity-computation lookup
   * (`KalmanBoxTracker.update`): walk `dt = deltaT, deltaT-1, ..., 1`,
   * returning the FIRST observation found at `age - dt` (i.e. the oldest
   * within the last `deltaT` frames). Falls back to `lastObservation` only
   * when no observation exists in any age in `[age - deltaT, age - 1]`.
   *
   * Caller guarantees `lastObservation !== null`.
   */
  private _lookupVelocityPrev(track: OcSortInternalTrack<TPayload>): BBox {
    for (let i = 0; i < this.deltaT; i++) {
      const dt = this.deltaT - i;
      const found = track.observations.get(track.age - dt);
      if (found !== undefined) return found;
    }
    return track.lastObservation!;
  }

  /**
   * Apply an unsuccessful match. On the FIRST miss of an unobserved run
   * (when `tsu === 0` going in and a `lastObservation` exists), capture the
   * current Kalman state — the post-predict state from this frame — as the
   * ORU snapshot. Mirrors noahcao's `freeze()` being called once on the
   * `observed → not observed` transition inside `KalmanFilterNew.update`.
   */
  private applyOcSortMiss(track: OcSortInternalTrack<TPayload>): void {
    if (
      track.timeSinceUpdate === 0 &&
      track.lastObservation !== null &&
      track.kalmanSnapshot === null
    ) {
      track.kalmanSnapshot = track.kalmanState;
    }
    this.applyMiss(track);
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

// ─────────────────────────────────────────────────────────────────────────────
// Module-scope helpers. Hoisted out of the class so they don't allocate a
// closure per frame (CONTRIBUTING.md §3.4: no closures inside per-frame loops).
// ─────────────────────────────────────────────────────────────────────────────

interface StageAssociationResult {
  readonly matched: ReadonlyArray<readonly [number, number]>;
  readonly unmatchedTracks: ReadonlyArray<number>;
  readonly unmatchedDetections: ReadonlyArray<number>;
}

/**
 * Primary OC-SORT association: IoU cost augmented by the OCM angular term.
 * The reference (`association.py:associate`) hard-codes IoU here even when
 * `asoFunc='giou'`; this port preserves that asymmetry.
 *
 * For each (track, detection) pair:
 *
 * - Gate IoU < `iouThreshold` to `+Infinity` (ARCHITECTURE.md §5.6).
 * - Otherwise `cost = (1 - IoU) - inertia * det.score * angularConsistency`,
 *   where `angularConsistency ∈ [-0.5, +0.5]` is `(π/2 − |Δangle|) / π` of the
 *   angle between the track's stored velocity and the direction from the
 *   track's `kPreviousObs` observation to the detection center. The
 *   contribution is zero when either the track has no velocity yet or the
 *   `kPreviousObs` lookup returns null (`valid_mask = 0` in the reference).
 */
function associatePrimary<TPayload>(
  tracks: ReadonlyArray<OcSortInternalTrack<TPayload>>,
  dets: ReadonlyArray<Detection<TPayload>>,
  iouThresh: number,
  inertia: number,
  deltaT: number,
): StageAssociationResult {
  const M = tracks.length;
  const N = dets.length;
  if (M === 0 || N === 0) return _emptyAssociation(M, N);

  const trackBoxes: BBox[] = new Array(M);
  for (let i = 0; i < M; i++) trackBoxes[i] = tracks[i]!.bbox;
  const detBoxes: BBox[] = new Array(N);
  for (let j = 0; j < N; j++) detBoxes[j] = dets[j]!.bbox;

  const iou = iouMatrix(trackBoxes, detBoxes);
  const cost = new Float64Array(M * N);

  for (let i = 0; i < M; i++) {
    const track = tracks[i]!;
    const kPrev = _kPreviousObs(track.observations, track.age, deltaT);
    const vel = track.velocity; // may be null
    const ocmActive = inertia > 0 && vel !== null && kPrev !== null;
    for (let j = 0; j < N; j++) {
      const iouVal = iou[i * N + j]!;
      if (iouVal < iouThresh) {
        cost[i * N + j] = Number.POSITIVE_INFINITY;
        continue;
      }
      let c = 1 - iouVal;
      if (ocmActive) {
        const dir = _speedDirection(kPrev!, dets[j]!.bbox);
        // vel and dir are both [dy, dx]; their dot product is cos(Δangle).
        let cosAng = vel![0] * dir[0] + vel![1] * dir[1];
        if (cosAng > 1) cosAng = 1;
        else if (cosAng < -1) cosAng = -1;
        const ang = Math.acos(cosAng);
        const diffAng = (Math.PI / 2 - Math.abs(ang)) / Math.PI;
        c -= inertia * dets[j]!.score * diffAng;
      }
      cost[i * N + j] = c;
    }
  }

  return _solveAndPackage(cost, M, N);
}

/**
 * BYTE / OCR association: pure `asoFunc`-distance cost, gated at `iouThresh`.
 * Mirrors `OCSort.update`'s post-primary stages, where `self.asso_func` is
 * applied without the OCM term.
 *
 * The gating threshold is the `asoFunc` value (not raw IoU) — for `'giou'`,
 * `giouMatrix` returns the un-normalized GIoU in `[-1, +1]` (see
 * `geometry/iou.ts`), and this stage normalizes it to `[0, 1]` first
 * (`(giou + 1) / 2`, matching `association.py:giou_batch`) so that
 * `iouThresh = 0.3` continues to mean "30% normalized score."
 */
function associateAsoFunc<TPayload>(
  trackBoxes: ReadonlyArray<BBox>,
  dets: ReadonlyArray<Detection<TPayload>>,
  iouThresh: number,
  asoFunc: OcSortAsoFunc,
): StageAssociationResult {
  const M = trackBoxes.length;
  const N = dets.length;
  if (M === 0 || N === 0) return _emptyAssociation(M, N);

  const detBoxes: BBox[] = new Array(N);
  for (let j = 0; j < N; j++) detBoxes[j] = dets[j]!.bbox;

  let score: Float64Array;
  if (asoFunc === 'giou') {
    const raw = giouMatrix(trackBoxes, detBoxes);
    score = new Float64Array(raw.length);
    for (let i = 0; i < raw.length; i++) score[i] = (raw[i]! + 1) / 2;
  } else {
    score = iouMatrix(trackBoxes, detBoxes);
  }

  const cost = new Float64Array(M * N);
  for (let i = 0; i < M * N; i++) {
    const s = score[i]!;
    cost[i] = s < iouThresh ? Number.POSITIVE_INFINITY : 1 - s;
  }

  return _solveAndPackage(cost, M, N);
}

function _solveAndPackage(cost: Float64Array, M: number, N: number): StageAssociationResult {
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

function _emptyAssociation(M: number, N: number): StageAssociationResult {
  const ut: number[] = new Array(M);
  for (let i = 0; i < M; i++) ut[i] = i;
  const ud: number[] = new Array(N);
  for (let j = 0; j < N; j++) ud[j] = j;
  return { matched: [], unmatchedTracks: ut, unmatchedDetections: ud };
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

/** Convert a `[x1, y1, x2, y2]` bbox to `[cx, cy, w, h]`. */
function _bboxCenterWH(bbox: BBox): [number, number, number, number] {
  const w = bbox[2] - bbox[0];
  const h = bbox[3] - bbox[1];
  return [bbox[0] + w / 2, bbox[1] + h / 2, w, h];
}
