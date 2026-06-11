import type { Detection, Track, Tracker } from './core.js';
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
  _tracker: Tracker<TPayload>,
  _detectionsPerFrame: ReadonlyArray<ReadonlyArray<Detection<TPayload>>>,
): Track<TPayload>[][] {
  throw new Error('not implemented');
}

/**
 * Flatten per-frame tracker output into MOTChallenge entries (frame index `i`
 * becomes the 1-based frame number `i + 1`), ready for
 * {@link import('./motchallenge/parse.js').formatMotChallenge} or
 * {@link evalFramesFromEntries}. `classId` carries through when present;
 * `visibility` is the absent-marker `-1`.
 */
export function tracksToMotEntries(
  _tracksPerFrame: ReadonlyArray<ReadonlyArray<Track<unknown>>>,
): MotEntry[] {
  throw new Error('not implemented');
}

/**
 * Join ground-truth entries and tracker entries into the per-frame
 * {@link EvalFrame} shape the metrics consume. Frames `1…numFrames` are
 * materialized (default: the maximum frame number on either side), so frames
 * where one side is absent still contribute their FNs / FPs. Within a frame,
 * entries keep their input order.
 */
export function evalFramesFromEntries(
  _gt: ReadonlyArray<MotEntry>,
  _tracker: ReadonlyArray<MotEntry>,
  _numFrames?: number,
): EvalFrame[] {
  throw new Error('not implemented');
}
