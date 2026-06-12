import type { EvalFrame } from '../metrics/frames.js';
import type { MotEntry } from './parse.js';

/**
 * MOT17/MOT20 distractor classes (TrackEval `mot_challenge_2d_box.py:325`):
 * `person_on_vehicle` (2), `static_person` (7), `distractor` (8),
 * `reflection` (12); MOT20 additionally treats `non_mot_vehicle` (6) as a
 * distractor. Tracker detections matched to gt of these classes are removed
 * before scoring.
 */
export const MOT17_DISTRACTOR_CLASS_IDS: ReadonlyArray<number> = [2, 7, 8, 12];
export const MOT20_DISTRACTOR_CLASS_IDS: ReadonlyArray<number> = [2, 6, 7, 8, 12];

/**
 * The full MOT16/17/20 gt class taxonomy (TrackEval's
 * `class_name_to_class_id`): pedestrian 1, person_on_vehicle 2, car 3,
 * bicycle 4, motorbike 5, non_mot_vehicle 6, static_person 7, distractor 8,
 * occluder 9, occluder_on_ground 10, occluder_full 11, reflection 12,
 * crowd 13. A gt class outside this list throws during preprocessing,
 * mirroring TrackEval's invalid-class guard.
 */
export const MOT_VALID_CLASS_IDS: ReadonlyArray<number> = [
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
];

/** Options for {@link preprocessMotSequence}. Defaults are the MOT17 setting. */
export interface MotPreprocessOptions {
  /**
   * Frames `1…numFrames` are materialized. Defaults to the maximum frame
   * number present on either side; pass the sequence's `seqLength` when
   * trailing empty frames matter.
   */
  readonly numFrames?: number;
  /** The evaluated class. Default 1 (pedestrian — the only MOT17 eval class). */
  readonly classId?: number;
  /** Default {@link MOT17_DISTRACTOR_CLASS_IDS}; pass the MOT20 list for MOT20. */
  readonly distractorClassIds?: ReadonlyArray<number>;
  /** Default {@link MOT_VALID_CLASS_IDS}. */
  readonly validClassIds?: ReadonlyArray<number>;
  /** Minimum IoU for the distractor-matching pass. Default 0.5. */
  readonly simThreshold?: number;
}

/**
 * TrackEval's MOT17/MOT20 preprocessing
 * (`mot_challenge_2d_box.py:get_preprocessed_seq_data`), turning raw gt and
 * tracker entries into the {@link EvalFrame}s the metrics consume. Per frame:
 *
 * 1. When both sides are non-empty, gt classes are validated against
 *    `validClassIds` (throws `/invalid.*class/i` otherwise — same guard, same
 *    branch as TrackEval: frames with an empty side are not validated).
 * 2. Tracker detections are matched against **all** gt boxes (any class) by
 *    Hungarian assignment on IoU, with pairs below `simThreshold − ε` zeroed.
 *    Tracker detections matched to a gt of a distractor class are removed —
 *    the tracker is not penalized for correctly detecting objects the
 *    benchmark annotates but does not evaluate.
 * 3. Gt entries are kept only if their consider flag (`score` column of gt
 *    files, truncated to integer like TrackEval's `astype(int)`) is non-zero
 *    AND their class is `classId`.
 *
 * Entry order within a frame is preserved on both sides. Tracker entries'
 * `classId` is ignored (MOT result files carry `-1`; TrackEval treats all
 * tracker detections as pedestrian).
 *
 * The output feeds directly into `clearMot` / `identity` / `hota`; the
 * `trackeval-preproc` fixture pins this whole pipeline against TrackEval's
 * output on a synthetic sequence.
 */
export function preprocessMotSequence(
  _gt: ReadonlyArray<MotEntry>,
  _tracker: ReadonlyArray<MotEntry>,
  _options?: MotPreprocessOptions,
): EvalFrame[] {
  throw new Error('not implemented');
}
