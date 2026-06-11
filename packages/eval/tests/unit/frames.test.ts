import { describe, expect, it } from 'vitest';
import { indexSequence } from '../../src/metrics/frames.js';
import { box, frame } from '../helpers.js';

describe('indexSequence', () => {
  it('remaps ids densely in order of first appearance', () => {
    const seq = indexSequence([
      frame(
        [
          [42, box(0, 0, 10, 10)],
          [7, box(20, 0, 10, 10)],
        ],
        [[900, box(0, 0, 10, 10)]],
      ),
      frame([[7, box(20, 0, 10, 10)]], [[5, box(20, 0, 10, 10)]]),
    ]);

    expect(seq.numGtIds).toBe(2);
    expect(seq.numTrackIds).toBe(2);
    // gt 42 → 0, gt 7 → 1; tracker 900 → 0, tracker 5 → 1.
    expect(Array.from(seq.frames[0]?.gt ?? [])).toEqual([0, 1]);
    expect(Array.from(seq.frames[1]?.gt ?? [])).toEqual([1]);
    expect(Array.from(seq.frames[0]?.track ?? [])).toEqual([0]);
    expect(Array.from(seq.frames[1]?.track ?? [])).toEqual([1]);
  });

  it('counts detections and per-id frame counts', () => {
    const b = box(0, 0, 10, 10);
    const seq = indexSequence([
      frame([[1, b]], [[1, b]]),
      frame(
        [
          [1, b],
          [2, box(20, 0, 10, 10)],
        ],
        [],
      ),
    ]);

    expect(seq.numGtDets).toBe(3);
    expect(seq.numTrackDets).toBe(1);
    expect(Array.from(seq.gtIdCounts)).toEqual([2, 1]);
    expect(Array.from(seq.trackIdCounts)).toEqual([1]);
  });

  it('computes the per-frame IoU similarity matrix row-major', () => {
    // gt [0,0,10,10] vs tracker [0,5,10,15]: inter 50, union 150 → 1/3.
    //                tracker [0,0,10,10]: identical → 1.
    const seq = indexSequence([
      frame(
        [[1, box(0, 0, 10, 10)]],
        [
          [8, box(0, 5, 10, 10)],
          [9, box(0, 0, 10, 10)],
        ],
      ),
    ]);

    const sim = seq.frames[0]?.sim ?? new Float64Array(0);
    expect(sim).toHaveLength(2);
    expect(sim[0]).toBeCloseTo(1 / 3, 12);
    expect(sim[1]).toBe(1);
  });

  it('throws when a sequence has no ground-truth detections', () => {
    expect(() => indexSequence([])).toThrow(/ground[\s-]?truth/i);
    expect(() => indexSequence([frame([], [[1, box(0, 0, 10, 10)]])])).toThrow(
      /ground[\s-]?truth/i,
    );
  });

  it('throws on id/box arrays of different lengths', () => {
    const bad = {
      gtIds: [1, 2],
      gtBoxes: [box(0, 0, 10, 10)],
      trackIds: [],
      trackBoxes: [],
    };
    expect(() => indexSequence([bad])).toThrow(/parallel/i);
  });

  it('throws on a duplicate identity within one frame', () => {
    const b = box(0, 0, 10, 10);
    expect(() =>
      indexSequence([
        frame(
          [
            [1, b],
            [1, box(20, 0, 10, 10)],
          ],
          [],
        ),
      ]),
    ).toThrow(/duplicate/i);
    expect(() =>
      indexSequence([
        frame(
          [[1, b]],
          [
            [3, b],
            [3, box(20, 0, 10, 10)],
          ],
        ),
      ]),
    ).toThrow(/duplicate/i);
  });
});
