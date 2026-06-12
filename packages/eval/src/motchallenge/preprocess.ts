import { iouMatrix, solveLsap } from '../core.js';
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
  gt: ReadonlyArray<MotEntry>,
  tracker: ReadonlyArray<MotEntry>,
  options?: MotPreprocessOptions,
): EvalFrame[] {
  const classId = options?.classId ?? 1;
  const distractorClassIds = options?.distractorClassIds ?? MOT17_DISTRACTOR_CLASS_IDS;
  const validClassIds = options?.validClassIds ?? MOT_VALID_CLASS_IDS;
  const threshold = options?.simThreshold ?? 0.5;

  let maxFrame = 0;
  for (const e of gt) maxFrame = Math.max(maxFrame, e.frame);
  for (const e of tracker) maxFrame = Math.max(maxFrame, e.frame);
  const numFrames = options?.numFrames ?? maxFrame;

  const gtByFrame = groupByFrame(gt, numFrames);
  const trackByFrame = groupByFrame(tracker, numFrames);

  const out: EvalFrame[] = [];
  for (let f = 0; f < numFrames; f++) {
    const gtFrame = gtByFrame[f] ?? [];
    const trackFrame = trackByFrame[f] ?? [];

    // Step 2 (mot_challenge_2d_box.py:358-381): match tracker dets against
    // ALL gt boxes regardless of class; tracker dets matched (IoU ≥ threshold)
    // to a distractor-class gt are removed. TrackEval only runs this — and
    // the gt class validation guarding it — when both sides are non-empty.
    const removed = new Set<number>();
    if (gtFrame.length > 0 && trackFrame.length > 0) {
      const invalid = [
        ...new Set(gtFrame.map((e) => e.classId).filter((c) => !validClassIds.includes(c))),
      ];
      if (invalid.length > 0) {
        throw new Error(
          `frame ${f + 1}: invalid gt classes for MOT preprocessing: ${invalid.join(', ')}`,
        );
      }

      const m = gtFrame.length;
      const n = trackFrame.length;
      const sim = iouMatrix(
        gtFrame.map((e) => e.bbox),
        trackFrame.map((e) => e.bbox),
      );
      // Sub-threshold pairs are zeroed (not forbidden) exactly like
      // TrackEval's `matching_scores`; the maximization is phrased as a
      // minimization over cost = 1 − score for the JV solver, and selected
      // zero-score pairs are discarded afterwards.
      const cost = new Float64Array(m * n);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          const s = sim[i * n + j] ?? 0;
          cost[i * n + j] = 1 - (s >= threshold - Number.EPSILON ? s : 0);
        }
      }
      const { rowToCol } = solveLsap(cost, m, n);
      for (let i = 0; i < m; i++) {
        const j = rowToCol[i] ?? -1;
        if (j === -1) continue;
        if (1 - (cost[i * n + j] ?? 1) <= Number.EPSILON) continue;
        if (distractorClassIds.includes(gtFrame[i]?.classId ?? -1)) removed.add(j);
      }
    }

    // Step 4: keep only considered (`zero_marked != 0`, integer-truncated
    // like TrackEval's astype(int)) gt rows of the evaluated class.
    const gtIds: number[] = [];
    const gtBoxes: EvalFrame['gtBoxes'][number][] = [];
    for (const e of gtFrame) {
      if (Math.trunc(e.score) !== 0 && e.classId === classId) {
        gtIds.push(e.id);
        gtBoxes.push(e.bbox);
      }
    }
    const trackIds: number[] = [];
    const trackBoxes: EvalFrame['trackBoxes'][number][] = [];
    for (let j = 0; j < trackFrame.length; j++) {
      const e = trackFrame[j];
      if (e !== undefined && !removed.has(j)) {
        trackIds.push(e.id);
        trackBoxes.push(e.bbox);
      }
    }
    out.push({ gtIds, gtBoxes, trackIds, trackBoxes });
  }
  return out;
}

function groupByFrame(entries: ReadonlyArray<MotEntry>, numFrames: number): MotEntry[][] {
  const out: MotEntry[][] = Array.from({ length: numFrames }, () => []);
  for (const e of entries) {
    out[e.frame - 1]?.push(e);
  }
  return out;
}
