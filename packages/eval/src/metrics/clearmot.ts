import { solveLsap } from '../core.js';
import type { EvalFrame } from './frames.js';
import { indexSequence } from './frames.js';

/** Options for {@link clearMot}. */
export interface ClearMotOptions {
  /**
   * Minimum IoU for a gt ↔ tracker pair to count as a match.
   * Default 0.5, the standard CLEAR-MOT threshold (Bernardin & Stiefelhagen 2008).
   */
  readonly simThreshold?: number;
}

/**
 * CLEAR-MOT results (Bernardin & Stiefelhagen, EURASIP 2008), computed the way
 * `JonathonLuiten/TrackEval` (`metrics/clear.py`) computes them — that is the
 * §4.2 reference implementation for this module.
 */
export interface ClearMotResult {
  /**
   * Multiple Object Tracking Accuracy: `1 - (fn + fp + idsw) / numGtDets`.
   * Can be negative when errors outnumber ground-truth detections.
   */
  readonly mota: number;
  /**
   * Multiple Object Tracking Precision: mean IoU over matched pairs
   * (TrackEval's similarity convention, not the legacy distance form).
   * 0 when there are no matches.
   */
  readonly motp: number;
  /** Matched gt ↔ tracker pairs summed over frames. */
  readonly tp: number;
  /** Ground-truth detections left unmatched. */
  readonly fn: number;
  /** Tracker detections left unmatched. */
  readonly fp: number;
  /**
   * Identity switches: matches where the gt id's matched tracker id differs
   * from the tracker id it was last matched to at any earlier frame.
   */
  readonly idsw: number;
  /**
   * Fragmentations: matched→unmatched transitions of a gt id that are followed
   * by a later re-match (equivalently, matched-segment count − 1 per gt id
   * with at least one match).
   */
  readonly frag: number;
  /** Mostly-tracked gt identities (matched ratio > 0.8). */
  readonly mt: number;
  /** Partially-tracked gt identities (0.2 ≤ matched ratio ≤ 0.8). */
  readonly pt: number;
  /** Mostly-lost gt identities (matched ratio < 0.2). */
  readonly ml: number;
  /** Total ground-truth detections in the sequence. */
  readonly numGtDets: number;
  /** Distinct ground-truth identities in the sequence. */
  readonly numGtIds: number;
  /**
   * Sum of the IoU over every true positive — i.e. `motp × tp`, kept un-divided.
   *
   * Exposed because **MOTP does not aggregate as a mean of per-sequence MOTPs.**
   * TrackEval carries `MOTP_sum` as a *summed* field across sequences and only then
   * divides by the summed TP (`clear.py:_compute_final_fields`:
   * `MOTP = MOTP_sum / max(1, CLR_TP)`). Averaging the per-sequence MOTPs instead would
   * weight a 20-frame sequence the same as a 1000-frame one, and would not be the number
   * MOTChallenge publishes. See {@link import('../aggregate.js').combineClearMot}.
   */
  readonly motpSum: number;
}

/**
 * Compute the CLEAR-MOT metric family over a sequence.
 *
 * Per-frame matching follows TrackEval `clear.py`: maximize
 * `1000 · [tracker id matched this gt last frame] + IoU`, with pairs below
 * `simThreshold` forbidden. The continuity bonus keeps an existing gt ↔ tracker
 * pairing alive even when a different tracker box overlaps slightly better,
 * which is what makes IDSW counts meaningful.
 *
 * Throws on a sequence with zero ground-truth detections
 * (`/ground[\s-]?truth/i`); see {@link indexSequence} for the shared
 * degenerate-input contract.
 */
export function clearMot(
  frames: ReadonlyArray<EvalFrame>,
  options?: ClearMotOptions,
): ClearMotResult {
  const threshold = options?.simThreshold ?? 0.5;
  const seq = indexSequence(frames);

  let tp = 0;
  let fn = 0;
  let fp = 0;
  let idsw = 0;
  let motpSum = 0;

  // Per dense gt id: the tracker id it was last matched to at ANY earlier
  // frame (drives IDSW — gaps do not reset it), the tracker id it was matched
  // to in the immediately previous frame (drives the continuity bonus), the
  // total matched-frame count (drives MT/PT/ML), and the matched-segment
  // count (drives Frag). -1 = none.
  const lastMatchedTrack = new Int32Array(seq.numGtIds).fill(-1);
  const prevFrameTrack = new Int32Array(seq.numGtIds).fill(-1);
  const matchedCounts = new Float64Array(seq.numGtIds);
  const segments = new Int32Array(seq.numGtIds);

  const currFrameTrack = new Int32Array(seq.numGtIds);

  for (const frame of seq.frames) {
    const m = frame.gt.length;
    const n = frame.track.length;

    currFrameTrack.fill(-1);
    if (m > 0 && n > 0) {
      // TrackEval clear.py matching: maximize 1000·continuity + IoU with
      // sub-threshold pairs zeroed, then drop selected zero-score pairs.
      // Phrased as a minimization with cost = BIG − score so the JV solver
      // sees non-negative finite costs.
      const big = CONTINUITY_BONUS + 1;
      const cost = new Float64Array(m * n);
      for (let i = 0; i < m; i++) {
        const gtId = frame.gt[i] ?? 0;
        for (let j = 0; j < n; j++) {
          const sim = frame.sim[i * n + j] ?? 0;
          let score = 0;
          if (sim >= threshold - Number.EPSILON) {
            score = sim + (prevFrameTrack[gtId] === frame.track[j] ? CONTINUITY_BONUS : 0);
          }
          cost[i * n + j] = big - score;
        }
      }

      const { rowToCol } = solveLsap(cost, m, n);
      for (let i = 0; i < m; i++) {
        const j = rowToCol[i] ?? -1;
        if (j === -1) continue;
        const score = big - (cost[i * n + j] ?? big);
        if (score <= Number.EPSILON) continue;

        const gtId = frame.gt[i] ?? 0;
        const trackId = frame.track[j] ?? 0;
        tp++;
        motpSum += frame.sim[i * n + j] ?? 0;
        if (lastMatchedTrack[gtId] !== -1 && lastMatchedTrack[gtId] !== trackId) idsw++;
        lastMatchedTrack[gtId] = trackId;
        matchedCounts[gtId] = (matchedCounts[gtId] ?? 0) + 1;
        if (prevFrameTrack[gtId] === -1) segments[gtId] = (segments[gtId] ?? 0) + 1;
        currFrameTrack[gtId] = trackId;
      }

      // Commit the continuity memory ONLY on a frame that actually ran a match.
      //
      // This assignment must stay inside the `m > 0 && n > 0` guard. TrackEval
      // `clear.py:70-76` hits `continue` when a timestep has zero gt or zero tracker
      // detections, which skips the `prev_timestep_tracker_id[:] = np.nan` reset
      // further down — so the continuity memory SURVIVES the hole. Hoisting this line
      // out of the guard (as vestige originally had it) blanks the memory on every
      // empty frame, with two consequences on real data, where empty frames are
      // routine (a detector drops out, or preprocessing filters every gt in a frame
      // to a distractor):
      //
      //   * the +1000 continuity bonus stops protecting the existing gt↔tracker
      //     pairing on the frame after the hole, so the matcher is free to pick a
      //     different tracker id — a SPURIOUS ID SWITCH; and
      //   * `prevFrameTrack[gtId] === -1` reads as "was not tracked", so the resumed
      //     track is counted as a new segment — a SPURIOUS FRAG.
      //
      // Unmatched gt ids DO get cleared here (via `currFrameTrack`'s -1 fill), which
      // is the `prev_timestep_tracker_id[:] = np.nan` half of the reference's reset.
      prevFrameTrack.set(currFrameTrack);
    }
  }

  // FN/FP are global complements of TP against the detection totals.
  fn = seq.numGtDets - tp;
  fp = seq.numTrackDets - tp;

  let mt = 0;
  let pt = 0;
  let ml = 0;
  let frag = 0;
  for (let g = 0; g < seq.numGtIds; g++) {
    const ratio = (matchedCounts[g] ?? 0) / (seq.gtIdCounts[g] ?? 1);
    if (ratio > 0.8) mt++;
    else if (ratio < 0.2) ml++;
    else pt++;
    const segs = segments[g] ?? 0;
    if (segs > 0) frag += segs - 1;
  }

  return {
    mota: 1 - (fn + fp + idsw) / seq.numGtDets,
    motp: tp > 0 ? motpSum / tp : 0,
    tp,
    fn,
    fp,
    idsw,
    frag,
    mt,
    pt,
    ml,
    numGtDets: seq.numGtDets,
    numGtIds: seq.numGtIds,
    motpSum,
  };
}

/**
 * TrackEval's continuity bonus: a previous-frame pairing outscores any pure
 * IoU advantage (IoU ≤ 1 ≪ 1000), so matches persist until broken by the
 * threshold, not by jitter.
 */
const CONTINUITY_BONUS = 1000;
