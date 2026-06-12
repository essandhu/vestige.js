import { describe, expect, it } from 'vitest';
import { identity } from '../../src/metrics/identity.js';
import { box, frame, singleObjectSequence } from '../helpers.js';

// Oracle values hand-traced from the ID-metrics definition (Ristani et al.,
// ECCV Workshops 2016, §3) as implemented by TrackEval metrics/identity.py.

describe('identity', () => {
  it('scores perfect tracking as IDF1 1', () => {
    const r = identity(singleObjectSequence([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]));
    expect(r.idtp).toBe(10);
    expect(r.idfn).toBe(0);
    expect(r.idfp).toBe(0);
    expect(r.idf1).toBe(1);
    expect(r.idr).toBe(1);
    expect(r.idp).toBe(1);
  });

  it('halves IDF1 when one gt trajectory is split into two equal tracker ids', () => {
    // gt (10 frames) vs tracker id 1 (f1–5) + id 2 (f6–10). The optimal
    // trajectory assignment picks one tracker id: IDTP = 5, IDFN = 5 (frames
    // covered by the other id), IDFP = 5 (the other id's detections).
    // IDF1 = 2·5 / (2·5 + 5 + 5) = 0.5.
    const r = identity(singleObjectSequence([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]));
    expect(r.idtp).toBe(5);
    expect(r.idfn).toBe(5);
    expect(r.idfp).toBe(5);
    expect(r.idf1).toBeCloseTo(0.5, 12);
    expect(r.idr).toBeCloseTo(0.5, 12);
    expect(r.idp).toBeCloseTo(0.5, 12);
  });

  it('scores partial coverage with no FPs as pure recall loss', () => {
    // gt 10 frames, tracker covers 6 with one id, silent otherwise.
    // IDTP = 6, IDFN = 4, IDFP = 0 → IDF1 = 12/16 = 0.75, IDR = 0.6, IDP = 1.
    const r = identity(singleObjectSequence([1, 1, 1, 1, 1, 1, null, null, null, null]));
    expect(r.idtp).toBe(6);
    expect(r.idfn).toBe(4);
    expect(r.idfp).toBe(0);
    expect(r.idf1).toBeCloseTo(0.75, 12);
    expect(r.idr).toBeCloseTo(0.6, 12);
    expect(r.idp).toBe(1);
  });

  it('halves IDF1 when one tracker id spans two gt trajectories', () => {
    // gt 1 lives f1–5 at x=0; gt 2 lives f6–10 at x=100. Tracker id 5 covers
    // both perfectly. The assignment gives id 5 to one gt: IDTP = 5,
    // IDFN = 5 (the other gt), IDFP = 5 (id 5's frames on the other gt).
    const a = box(0, 0, 10, 10);
    const b = box(100, 0, 10, 10);
    const frames = [
      ...Array.from({ length: 5 }, () => frame([[1, a]], [[5, a]])),
      ...Array.from({ length: 5 }, () => frame([[2, b]], [[5, b]])),
    ];

    const r = identity(frames);
    expect(r.idtp).toBe(5);
    expect(r.idfn).toBe(5);
    expect(r.idfp).toBe(5);
    expect(r.idf1).toBeCloseTo(0.5, 12);
  });

  it('scores an empty tracker output as IDF1 0', () => {
    const r = identity(singleObjectSequence([null, null, null]));
    expect(r.idtp).toBe(0);
    expect(r.idfn).toBe(3);
    expect(r.idfp).toBe(0);
    expect(r.idf1).toBe(0);
    expect(r.idr).toBe(0);
    expect(r.idp).toBe(0);
  });

  it('requires per-frame IoU at or above simThreshold for agreement', () => {
    // Tracker box [4,0,14,10] vs gt [0,0,10,10]: IoU 3/7 ≈ 0.4286.
    // Below the 0.5 default → no agreement; at 0.3 → full agreement.
    const frames = Array.from({ length: 4 }, () =>
      frame([[1, box(0, 0, 10, 10)]], [[1, box(4, 0, 10, 10)]]),
    );
    expect(identity(frames).idtp).toBe(0);
    expect(identity(frames, { simThreshold: 0.3 }).idtp).toBe(4);
  });

  it('throws on a sequence with no ground-truth detections', () => {
    expect(() => identity([frame([], [[1, box(0, 0, 10, 10)]])])).toThrow(/ground[\s-]?truth/i);
  });
});
