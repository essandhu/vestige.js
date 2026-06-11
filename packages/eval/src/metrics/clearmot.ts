import type { EvalFrame } from './frames.js';

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
  _frames: ReadonlyArray<EvalFrame>,
  _options?: ClearMotOptions,
): ClearMotResult {
  throw new Error('not implemented');
}
