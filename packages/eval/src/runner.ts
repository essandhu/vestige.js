import type { BBox, Detection, Track, Tracker } from './core.js';
import type { EvalFrame } from './metrics/frames.js';
import type { MotEntry } from './motchallenge/parse.js';

/**
 * Drive a tracker through a whole sequence, one `update()` per frame, and
 * collect the per-frame confirmed-track outputs.
 *
 * The tracker is consumed as-is: it is **not** `reset()` first, so the caller
 * controls whether a run continues prior state (explicit cost over magic —
 * ARCHITECTURE.md §2.6). Pass a freshly constructed tracker for a normal
 * evaluation run.
 */
export function runTracker<TPayload>(
  tracker: Tracker<TPayload>,
  detectionsPerFrame: ReadonlyArray<ReadonlyArray<Detection<TPayload>>>,
): Track<TPayload>[][] {
  const out: Track<TPayload>[][] = [];
  for (const detections of detectionsPerFrame) {
    out.push(tracker.update(detections));
  }
  return out;
}

/**
 * Flatten per-frame tracker output into MOTChallenge entries (frame index `i`
 * becomes the 1-based frame number `i + 1`), ready for
 * {@link import('./motchallenge/parse.js').formatMotChallenge} or
 * {@link evalFramesFromEntries}. `classId` carries through when present;
 * `visibility` is the absent-marker `-1`.
 */
export function tracksToMotEntries(
  tracksPerFrame: ReadonlyArray<ReadonlyArray<Track<unknown>>>,
): MotEntry[] {
  const entries: MotEntry[] = [];
  for (let f = 0; f < tracksPerFrame.length; f++) {
    for (const track of tracksPerFrame[f] ?? []) {
      entries.push({
        frame: f + 1,
        id: track.id,
        bbox: track.bbox,
        score: track.score,
        classId: track.classId ?? -1,
        visibility: -1,
      });
    }
  }
  return entries;
}

/**
 * Join ground-truth entries and tracker entries into the per-frame
 * {@link EvalFrame} shape the metrics consume. Frames `1…numFrames` are
 * materialized (default: the maximum frame number on either side), so frames
 * where one side is absent still contribute their FNs / FPs. Within a frame,
 * entries keep their input order.
 */
export function evalFramesFromEntries(
  gt: ReadonlyArray<MotEntry>,
  tracker: ReadonlyArray<MotEntry>,
  numFrames?: number,
): EvalFrame[] {
  let maxFrame = 0;
  for (const e of gt) maxFrame = Math.max(maxFrame, e.frame);
  for (const e of tracker) maxFrame = Math.max(maxFrame, e.frame);
  const n = numFrames ?? maxFrame;

  interface MutableFrame {
    gtIds: number[];
    gtBoxes: BBox[];
    trackIds: number[];
    trackBoxes: BBox[];
  }
  const frames: MutableFrame[] = Array.from({ length: n }, () => ({
    gtIds: [],
    gtBoxes: [],
    trackIds: [],
    trackBoxes: [],
  }));

  for (const e of gt) {
    const frame = frames[e.frame - 1];
    if (frame === undefined) continue;
    frame.gtIds.push(e.id);
    frame.gtBoxes.push(e.bbox);
  }
  for (const e of tracker) {
    const frame = frames[e.frame - 1];
    if (frame === undefined) continue;
    frame.trackIds.push(e.id);
    frame.trackBoxes.push(e.bbox);
  }
  return frames;
}
