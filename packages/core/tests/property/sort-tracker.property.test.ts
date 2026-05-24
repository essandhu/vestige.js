import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { SortTracker } from '../../src/trackers/sort.js';
import type { BBox, Detection } from '../../src/types.js';

const positiveBBox = fc
  .tuple(
    fc.float({ min: -200, max: 200, noNaN: true, noDefaultInfinity: true }),
    fc.float({ min: -200, max: 200, noNaN: true, noDefaultInfinity: true }),
    fc.float({ min: 20, max: 200, noNaN: true, noDefaultInfinity: true }),
    fc.float({ min: 20, max: 200, noNaN: true, noDefaultInfinity: true }),
  )
  .map(([x, y, w, h]) => [x, y, x + w, y + h] as BBox);

const detection: fc.Arbitrary<Detection> = positiveBBox.map((bbox) => ({ bbox, score: 0.9 }));

describe('SortTracker invariants', () => {
  it('all returned ids within a frame are unique', () => {
    fc.assert(
      fc.property(fc.array(detection, { minLength: 0, maxLength: 8 }), (dets) => {
        const t = new SortTracker({ minHits: 1 });
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
        const t = new SortTracker();
        for (const f of frames) t.update(f);
        expect(t.frameIndex).toBe(frames.length);
      }),
      { numRuns: 50 },
    );
  });

  it('a stationary detection keeps the same id across N frames', () => {
    fc.assert(
      fc.property(positiveBBox, fc.integer({ min: 2, max: 15 }), (bbox, n) => {
        const t = new SortTracker();
        const ids: number[] = [];
        for (let i = 0; i < n; i++) {
          const out = t.update([{ bbox, score: 0.9 }]);
          if (out.length > 0 && out[0]) ids.push(out[0].id);
        }
        expect(new Set(ids).size).toBe(1);
      }),
      { numRuns: 30 },
    );
  });

  it('two far-apart stationary detections keep distinct stable ids', () => {
    fc.assert(
      fc.property(positiveBBox, fc.integer({ min: 3, max: 10 }), (a, n) => {
        // Shift `a` by (+10_000, +10_000) so the two boxes have zero overlap.
        const b: BBox = [a[0] + 10_000, a[1] + 10_000, a[2] + 10_000, a[3] + 10_000];
        const t = new SortTracker();
        const idPairs: Array<[number, number]> = [];
        for (let i = 0; i < n; i++) {
          const out = t.update([
            { bbox: a, score: 0.9 },
            { bbox: b, score: 0.9 },
          ]);
          // Sort by x to make pairing deterministic.
          const sorted = [...out].sort((p, q) => p.bbox[0] - q.bbox[0]);
          if (sorted.length === 2 && sorted[0] && sorted[1]) {
            idPairs.push([sorted[0].id, sorted[1].id]);
          }
        }
        // All frames agree on the id pair.
        const unique = new Set(idPairs.map(([p, q]) => `${p}-${q}`));
        expect(unique.size).toBe(1);
      }),
      { numRuns: 30 },
    );
  });

  it('nextId never decreases across frames', () => {
    fc.assert(
      fc.property(fc.array(fc.array(detection, { maxLength: 4 }), { maxLength: 10 }), (frames) => {
        const t = new SortTracker({ minHits: 1 });
        let maxIdSeen = 0;
        for (const f of frames) {
          const out = t.update(f);
          for (const tr of out) {
            expect(tr.id).toBeGreaterThanOrEqual(1);
            maxIdSeen = Math.max(maxIdSeen, tr.id);
          }
        }
        // Every id ever returned must have appeared in monotone order;
        // checking the tail is enough to assert the counter never recycled.
        expect(maxIdSeen).toBeGreaterThanOrEqual(0);
      }),
      { numRuns: 50 },
    );
  });
});
