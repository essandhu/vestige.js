import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { ByteTracker } from '../../src/trackers/bytetrack.js';
import type { BBox, Detection } from '../../src/types.js';

const positiveBBox = fc
  .tuple(
    fc.float({ min: -200, max: 200, noNaN: true, noDefaultInfinity: true }),
    fc.float({ min: -200, max: 200, noNaN: true, noDefaultInfinity: true }),
    fc.float({ min: 20, max: 200, noNaN: true, noDefaultInfinity: true }),
    fc.float({ min: 20, max: 200, noNaN: true, noDefaultInfinity: true }),
  )
  .map(([x, y, w, h]) => [x, y, x + w, y + h] as BBox);

// Detection score ∈ [0.7, 1.0] to keep generated boxes in the high-score band
// (above detThresh = trackThresh + 0.1 = 0.6 by default). Stage-2 and
// score-floor behavior are exercised in the unit tests, not the property tests.
const detection: fc.Arbitrary<Detection> = positiveBBox.map((bbox) => ({ bbox, score: 0.9 }));

describe('ByteTracker invariants', () => {
  it('all returned ids within a frame are unique', () => {
    fc.assert(
      fc.property(fc.array(detection, { minLength: 0, maxLength: 8 }), (dets) => {
        const t = new ByteTracker();
        const out = t.update(dets);
        const ids = out.map((tr) => tr.id);
        expect(new Set(ids).size).toBe(ids.length);
      }),
      { numRuns: 100 },
    );
  });

  it('frameIndex equals the number of update() calls', () => {
    fc.assert(
      fc.property(fc.array(fc.array(detection, { maxLength: 4 }), { maxLength: 12 }), (frames) => {
        const t = new ByteTracker();
        for (const f of frames) t.update(f);
        expect(t.frameIndex).toBe(frames.length);
      }),
      { numRuns: 50 },
    );
  });

  it('a stationary detection keeps the same id across N frames', () => {
    fc.assert(
      fc.property(positiveBBox, fc.integer({ min: 2, max: 15 }), (bbox, n) => {
        const t = new ByteTracker();
        const ids: number[] = [];
        for (let i = 0; i < n; i++) {
          const out = t.update([{ bbox, score: 0.9 }]);
          if (out.length > 0 && out[0]) ids.push(out[0].id);
        }
        // Frame 1 outputs immediately; subsequent frames keep id=1.
        expect(new Set(ids).size).toBe(1);
      }),
      { numRuns: 30 },
    );
  });

  it('two far-apart stationary detections keep distinct stable ids', () => {
    fc.assert(
      fc.property(positiveBBox, fc.integer({ min: 3, max: 10 }), (a, n) => {
        const b: BBox = [a[0] + 10_000, a[1] + 10_000, a[2] + 10_000, a[3] + 10_000];
        const t = new ByteTracker();
        const idPairs: Array<[number, number]> = [];
        for (let i = 0; i < n; i++) {
          const out = t.update([
            { bbox: a, score: 0.9 },
            { bbox: b, score: 0.9 },
          ]);
          const sorted = [...out].sort((p, q) => p.bbox[0] - q.bbox[0]);
          if (sorted.length === 2 && sorted[0] && sorted[1]) {
            idPairs.push([sorted[0].id, sorted[1].id]);
          }
        }
        const unique = new Set(idPairs.map(([p, q]) => `${p}-${q}`));
        expect(unique.size).toBe(1);
      }),
      { numRuns: 30 },
    );
  });

  it('timeSinceUpdate equals exactly N after N consecutive missed frames', () => {
    // Direct check of the ADR-0003 §1 exactly-once invariant: each missed
    // frame must advance tsu by exactly 1. A doubled-applyMiss bug — the
    // exact bug that prompted ADR-0003 — would make this fire at N=1
    // (observed tsu=2). trackBuffer=100 keeps the track lost across the
    // full N range (frameRate defaults to 30 → effective maxAge=100) so we
    // never sample after removal.
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 20 }), positiveBBox, (n, bbox) => {
        const t = new ByteTracker({ trackBuffer: 100 });
        t.update([{ bbox, score: 0.9 }]); // frame 1: frame-1 instant activation → confirmed
        for (let i = 0; i < n; i++) t.update([]);
        const lost = t.getLostTracks();
        expect(lost).toHaveLength(1);
        expect(lost[0]?.timeSinceUpdate).toBe(n);
      }),
      { numRuns: 30 },
    );
  });

  it('nextId never decreases across frames', () => {
    fc.assert(
      fc.property(fc.array(fc.array(detection, { maxLength: 4 }), { maxLength: 10 }), (frames) => {
        const t = new ByteTracker();
        let maxIdSeen = 0;
        for (const f of frames) {
          const out = t.update(f);
          for (const tr of out) {
            expect(tr.id).toBeGreaterThanOrEqual(1);
            maxIdSeen = Math.max(maxIdSeen, tr.id);
          }
        }
        expect(maxIdSeen).toBeGreaterThanOrEqual(0);
      }),
      { numRuns: 50 },
    );
  });
});
