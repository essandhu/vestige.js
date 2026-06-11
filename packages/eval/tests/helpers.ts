import type { BBox } from '../src/core.js';
import type { EvalFrame } from '../src/metrics/frames.js';

/** Build an xyxy box from MOT-style left/top/width/height. */
export function box(x: number, y: number, w: number, h: number): BBox {
  return [x, y, x + w, y + h];
}

/** Build an {@link EvalFrame} from `[id, bbox]` pair lists. */
export function frame(
  gt: ReadonlyArray<readonly [number, BBox]>,
  tracker: ReadonlyArray<readonly [number, BBox]>,
): EvalFrame {
  return {
    gtIds: gt.map(([id]) => id),
    gtBoxes: gt.map(([, b]) => b),
    trackIds: tracker.map(([id]) => id),
    trackBoxes: tracker.map(([, b]) => b),
  };
}

/**
 * A sequence where one ground-truth object (id 7, stationary at
 * `box(0, 0, 10, 10)`) exists for `n` frames and the tracker reports the
 * exact same box with per-frame tracker ids taken from `trackerIds`
 * (`null` = no tracker output that frame).
 */
export function singleObjectSequence(trackerIds: ReadonlyArray<number | null>): EvalFrame[] {
  const b = box(0, 0, 10, 10);
  return trackerIds.map((id) => frame([[7, b]], id === null ? [] : [[id, b]]));
}
