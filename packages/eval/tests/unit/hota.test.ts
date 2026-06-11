import { describe, expect, it } from 'vitest';
import { hota } from '../../src/metrics/hota.js';
import { box, frame, singleObjectSequence } from '../helpers.js';

// Oracle values hand-traced from the HOTA definition (Luiten et al., IJCV
// 2020, eqs. 5–13) as implemented by TrackEval metrics/hota.py. The split-
// track and constant-IoU scenarios are small enough that DetA / AssA / the
// alpha integral can be carried out by hand.

const everyAlpha = (arr: Float64Array, expected: number): void => {
  for (const v of arr) expect(v).toBeCloseTo(expected, 12);
};

describe('hota', () => {
  it('exposes the 19 standard alphas (k+1)/20', () => {
    const r = hota(singleObjectSequence([1]));
    expect(r.alphas).toHaveLength(19);
    expect(r.alphas[0]).toBeCloseTo(0.05, 12);
    expect(r.alphas[9]).toBeCloseTo(0.5, 12);
    expect(r.alphas[18]).toBeCloseTo(0.95, 12);
    expect(r.hotaPerAlpha).toHaveLength(19);
    expect(r.detaPerAlpha).toHaveLength(19);
    expect(r.assaPerAlpha).toHaveLength(19);
    expect(r.locAPerAlpha).toHaveLength(19);
  });

  it('scores perfect tracking as 1 across the board', () => {
    const r = hota(singleObjectSequence([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]));
    expect(r.hota).toBe(1);
    expect(r.deta).toBe(1);
    expect(r.assa).toBe(1);
    expect(r.detRe).toBe(1);
    expect(r.detPr).toBe(1);
    expect(r.assRe).toBe(1);
    expect(r.assPr).toBe(1);
    expect(r.locA).toBe(1);
    everyAlpha(r.hotaPerAlpha, 1);
  });

  it('scores a split track as DetA 1, AssA 0.5, HOTA √0.5', () => {
    // 1 gt × 10 frames, perfect boxes; tracker id 1 (f1–5), id 2 (f6–10).
    // Every det matches at every alpha → DetA = 10/10 = 1.
    // matches(gt, 1) = 5, matches(gt, 2) = 5; gtCount = 10, trCount = 5 each.
    // A(c) = 5/(10 + 5 − 5) = 0.5 for every TP → AssA = 0.5.
    // AssRe per TP = 5/10 = 0.5; AssPr per TP = 5/5 = 1. HOTA = √0.5.
    const r = hota(singleObjectSequence([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]));
    expect(r.deta).toBe(1);
    expect(r.assa).toBeCloseTo(0.5, 12);
    expect(r.hota).toBeCloseTo(Math.SQRT1_2, 12);
    expect(r.assRe).toBeCloseTo(0.5, 12);
    expect(r.assPr).toBe(1);
    expect(r.locA).toBe(1);
    everyAlpha(r.assaPerAlpha, 0.5);
    everyAlpha(r.hotaPerAlpha, Math.SQRT1_2);
  });

  it('integrates over alphas for a constant localization quality of 0.6', () => {
    // 1 gt × 10 frames; one stable tracker id whose box overlaps at exactly
    // IoU 0.6 every frame ([0,0,10,10] vs [0,2.5,10,12.5]: 75/125).
    // Alphas 0.05…0.60 (12 of 19) match: DetA = AssA = 1, LocA = 0.6.
    // Alphas 0.65…0.95 (7 of 19) have no TPs: DetA = AssA = HOTA = 0 and
    // LocA = 1 by the TrackEval zero-TP convention.
    // Overall: HOTA = DetA = AssA = 12/19; LocA = (12·0.6 + 7·1)/19.
    const g = box(0, 0, 10, 10);
    const t = box(0, 2.5, 10, 10);
    const frames = Array.from({ length: 10 }, () => frame([[1, g]], [[3, t]]));

    const r = hota(frames);
    for (let k = 0; k < 19; k++) {
      const matched = (k + 1) / 20 <= 0.6;
      expect(r.hotaPerAlpha[k]).toBeCloseTo(matched ? 1 : 0, 12);
      expect(r.detaPerAlpha[k]).toBeCloseTo(matched ? 1 : 0, 12);
      expect(r.assaPerAlpha[k]).toBeCloseTo(matched ? 1 : 0, 12);
      expect(r.locAPerAlpha[k]).toBeCloseTo(matched ? 0.6 : 1, 12);
    }
    expect(r.hota).toBeCloseTo(12 / 19, 12);
    expect(r.deta).toBeCloseTo(12 / 19, 12);
    expect(r.assa).toBeCloseTo(12 / 19, 12);
    expect(r.locA).toBeCloseTo((12 * 0.6 + 7) / 19, 12);
  });

  it('decomposes pure detection loss: 8 of 10 frames covered → HOTA 0.8', () => {
    // 1 gt × 10 frames; tracker id 1 matches perfectly f1–8, silent f9–10.
    // TP = 8, FN = 2, FP = 0 → DetA = 0.8, DetRe = 0.8, DetPr = 1.
    // matches(gt,1) = 8; gtCount = 10, trCount = 8 → A(c) = 8/(10+8−8) = 0.8
    // → AssA = 0.8, AssRe = 0.8, AssPr = 1. HOTA = √(0.8·0.8) = 0.8.
    const r = hota(singleObjectSequence([1, 1, 1, 1, 1, 1, 1, 1, null, null]));
    expect(r.deta).toBeCloseTo(0.8, 12);
    expect(r.detRe).toBeCloseTo(0.8, 12);
    expect(r.detPr).toBe(1);
    expect(r.assa).toBeCloseTo(0.8, 12);
    expect(r.assRe).toBeCloseTo(0.8, 12);
    expect(r.assPr).toBe(1);
    expect(r.hota).toBeCloseTo(0.8, 12);
  });

  it('penalizes a clean two-object id swap through the association term', () => {
    // gt 1 at x=0 and gt 2 at x=100, 10 frames each, disjoint. Tracker A sits
    // on gt 1 for f1–5 then on gt 2 for f6–10; tracker B vice versa. All sims
    // are exactly 1 or 0, so DetA = 1 at every alpha.
    // matches(i, c) = 5 for all four pairs; gtCount = trCount = 10.
    // A(c) = 5/(10 + 10 − 5) = 1/3 → AssA = 1/3, HOTA = √(1/3).
    const a = box(0, 0, 10, 10);
    const b = box(100, 0, 10, 10);
    const frames = [
      ...Array.from({ length: 5 }, () =>
        frame(
          [
            [1, a],
            [2, b],
          ],
          [
            [101, a],
            [102, b],
          ],
        ),
      ),
      ...Array.from({ length: 5 }, () =>
        frame(
          [
            [1, a],
            [2, b],
          ],
          [
            [101, b],
            [102, a],
          ],
        ),
      ),
    ];

    const r = hota(frames);
    expect(r.deta).toBe(1);
    expect(r.assa).toBeCloseTo(1 / 3, 12);
    expect(r.hota).toBeCloseTo(Math.sqrt(1 / 3), 12);
  });

  it('throws on a sequence with no ground-truth detections', () => {
    expect(() => hota([frame([], [[1, box(0, 0, 10, 10)]])])).toThrow(/ground[\s-]?truth/i);
  });
});
