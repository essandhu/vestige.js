import { describe, expect, it } from 'vitest';
import { clearMot } from '../../src/metrics/clearmot.js';
import { box, frame, singleObjectSequence } from '../helpers.js';

// Oracle values hand-traced against the TrackEval CLEAR semantics
// (JonathonLuiten/TrackEval, metrics/clear.py) — the CONTRIBUTING.md §4.2
// reference for this module.

describe('clearMot', () => {
  it('scores perfect tracking as MOTA 1 / MOTP 1', () => {
    // 2 gt objects × 5 frames, tracker output identical with stable ids.
    // TP = 10, FN = FP = IDSW = Frag = 0; both ids matched 5/5 > 0.8 → MT 2.
    const a = box(0, 0, 10, 10);
    const b = box(50, 0, 10, 10);
    const frames = Array.from({ length: 5 }, () =>
      frame(
        [
          [1, a],
          [2, b],
        ],
        [
          [11, a],
          [12, b],
        ],
      ),
    );

    const r = clearMot(frames);
    expect(r.mota).toBe(1);
    expect(r.motp).toBe(1);
    expect(r.tp).toBe(10);
    expect(r.fn).toBe(0);
    expect(r.fp).toBe(0);
    expect(r.idsw).toBe(0);
    expect(r.frag).toBe(0);
    expect(r.mt).toBe(2);
    expect(r.pt).toBe(0);
    expect(r.ml).toBe(0);
    expect(r.numGtDets).toBe(10);
    expect(r.numGtIds).toBe(2);
  });

  it('counts an identity switch mid-track', () => {
    // 1 gt × 4 frames, perfect boxes; tracker id 1 on frames 1–2, id 2 on 3–4.
    // TP = 4, IDSW = 1 (frame 3: last-matched id was 1), Frag = 0 (no gap).
    // MOTA = 1 - (0 + 0 + 1)/4 = 0.75. Matched ratio 1 > 0.8 → MT.
    const r = clearMot(singleObjectSequence([1, 1, 2, 2]));
    expect(r.tp).toBe(4);
    expect(r.idsw).toBe(1);
    expect(r.frag).toBe(0);
    expect(r.mota).toBeCloseTo(0.75, 12);
    expect(r.mt).toBe(1);
  });

  it('counts misses, false positives, and fragmentations', () => {
    // 1 gt × 4 frames. Tracker: match f1, nothing f2, match f3 + a far
    // spurious box, match f4. TP = 3, FN = 1 (f2), FP = 1 (spurious),
    // IDSW = 0 (same id resumes), Frag = 1 (gap f2 with later re-match).
    // MOTA = 1 - (1 + 1 + 0)/4 = 0.5. Ratio 3/4 = 0.75 → PT.
    const b = box(0, 0, 10, 10);
    const far = box(500, 500, 10, 10);
    const frames = [
      frame([[7, b]], [[1, b]]),
      frame([[7, b]], []),
      frame(
        [[7, b]],
        [
          [1, b],
          [99, far],
        ],
      ),
      frame([[7, b]], [[1, b]]),
    ];

    const r = clearMot(frames);
    expect(r.tp).toBe(3);
    expect(r.fn).toBe(1);
    expect(r.fp).toBe(1);
    expect(r.idsw).toBe(0);
    expect(r.frag).toBe(1);
    expect(r.mota).toBeCloseTo(0.5, 12);
    expect(r.motp).toBe(1);
    expect(r.mt).toBe(0);
    expect(r.pt).toBe(1);
    expect(r.ml).toBe(0);
  });

  it('treats sub-threshold overlap as FN + FP (and MOTA can go negative)', () => {
    // Single frame: gt [0,0,10,10] vs tracker [4,0,14,10] → inter 60,
    // union 140, IoU = 3/7 ≈ 0.4286 < 0.5 → unmatched.
    // TP = 0, FN = 1, FP = 1 → MOTA = 1 - 2/1 = -1; MOTP over 0 TPs = 0.
    const r = clearMot([frame([[1, box(0, 0, 10, 10)]], [[1, box(4, 0, 10, 10)]])]);
    expect(r.tp).toBe(0);
    expect(r.fn).toBe(1);
    expect(r.fp).toBe(1);
    expect(r.mota).toBe(-1);
    expect(r.motp).toBe(0);
  });

  it('honors a custom simThreshold', () => {
    // Same geometry as above, threshold 0.3 → IoU 3/7 ≥ 0.3 is now a match.
    const r = clearMot([frame([[1, box(0, 0, 10, 10)]], [[1, box(4, 0, 10, 10)]])], {
      simThreshold: 0.3,
    });
    expect(r.tp).toBe(1);
    expect(r.mota).toBe(1);
    expect(r.motp).toBeCloseTo(3 / 7, 12);
  });

  it('keeps the previous-frame pairing over a higher-IoU competitor', () => {
    // Frame 1: gt G ↔ T1 (IoU = 75/125 = 0.6), T1 only.
    // Frame 2: T1 overlaps G at 70/130 = 7/13 ≈ 0.538; a new box T2 overlaps
    // at 95/105 = 19/21 ≈ 0.905. TrackEval's 1000·continuity bonus keeps
    // G ↔ T1, leaving T2 as FP and IDSW = 0.
    // MOTP = (0.6 + 7/13)/2; MOTA = 1 - (0 + 1 + 0)/2 = 0.5.
    const g = box(0, 0, 10, 10);
    const frames = [
      frame([[1, g]], [[1, box(0, 2.5, 10, 10)]]),
      frame(
        [[1, g]],
        [
          [1, box(0, 3, 10, 10)],
          [2, box(0, 0.5, 10, 10)],
        ],
      ),
    ];

    const r = clearMot(frames);
    expect(r.tp).toBe(2);
    expect(r.fp).toBe(1);
    expect(r.idsw).toBe(0);
    expect(r.motp).toBeCloseTo((0.6 + 7 / 13) / 2, 12);
    expect(r.mota).toBeCloseTo(0.5, 12);
  });

  it('counts an identity switch across an unmatched gap', () => {
    // 1 gt × 3 frames: id 1 at f1, nothing at f2, id 2 at f3. The last-ever
    // matched id (1) persists across the gap, so the f3 match is a switch.
    // TP = 2, FN = 1, IDSW = 1, Frag = 1; MOTA = 1 - (1 + 0 + 1)/3 = 1/3.
    const r = clearMot(singleObjectSequence([1, null, 2]));
    expect(r.tp).toBe(2);
    expect(r.fn).toBe(1);
    expect(r.idsw).toBe(1);
    expect(r.frag).toBe(1);
    expect(r.mota).toBeCloseTo(1 / 3, 12);
  });

  it('classifies MT / PT / ML by matched ratio', () => {
    // gt 7 present in all 10 frames; matched in exactly 1 (ratio 0.1 < 0.2 → ML).
    const ml = clearMot(
      singleObjectSequence([1, null, null, null, null, null, null, null, null, null]),
    );
    expect(ml.ml).toBe(1);
    expect(ml.mt).toBe(0);
    expect(ml.pt).toBe(0);

    // Matched in 8 of 10 (ratio 0.8, not > 0.8) → PT, per TrackEval's strict
    // `np.greater(ratio, 0.8)`.
    const pt = clearMot(singleObjectSequence([1, 1, 1, 1, 1, 1, 1, 1, null, null]));
    expect(pt.pt).toBe(1);
    expect(pt.mt).toBe(0);
  });

  it('throws on a sequence with no ground-truth detections', () => {
    expect(() => clearMot([frame([], [[1, box(0, 0, 10, 10)]])])).toThrow(/ground[\s-]?truth/i);
  });
});
