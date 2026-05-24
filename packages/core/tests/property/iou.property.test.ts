import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { giou, iou, iouMatrix } from '../../src/geometry/iou.js';
import type { BBox } from '../../src/types.js';

const positiveBBox = fc
  .tuple(
    fc.float({ min: -500, max: 500, noNaN: true, noDefaultInfinity: true }),
    fc.float({ min: -500, max: 500, noNaN: true, noDefaultInfinity: true }),
    fc.float({ min: 0.5, max: 200, noNaN: true, noDefaultInfinity: true }),
    fc.float({ min: 0.5, max: 200, noNaN: true, noDefaultInfinity: true }),
  )
  .map(([x, y, w, h]) => [x, y, x + w, y + h] as BBox);

describe('IoU invariants', () => {
  it('is in [0, 1]', () => {
    fc.assert(
      fc.property(positiveBBox, positiveBBox, (a, b) => {
        const v = iou(a, b);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }),
    );
  });

  it('is symmetric', () => {
    fc.assert(
      fc.property(positiveBBox, positiveBBox, (a, b) => {
        expect(iou(a, b)).toBeCloseTo(iou(b, a), 9);
      }),
    );
  });

  it('iou(a, a) === 1 for positive-area boxes', () => {
    fc.assert(
      fc.property(positiveBBox, (a) => {
        expect(iou(a, a)).toBeCloseTo(1, 9);
      }),
    );
  });
});

describe('GIoU invariants', () => {
  it('is in [-1, 1]', () => {
    fc.assert(
      fc.property(positiveBBox, positiveBBox, (a, b) => {
        const v = giou(a, b);
        expect(v).toBeGreaterThanOrEqual(-1 - 1e-9);
        expect(v).toBeLessThanOrEqual(1 + 1e-9);
      }),
    );
  });

  it('giou <= iou for all box pairs', () => {
    fc.assert(
      fc.property(positiveBBox, positiveBBox, (a, b) => {
        expect(giou(a, b)).toBeLessThanOrEqual(iou(a, b) + 1e-9);
      }),
    );
  });
});

describe('iouMatrix matches scalar iou', () => {
  it('cell (i, j) equals iou(preds[i], dets[j])', () => {
    fc.assert(
      fc.property(
        fc.array(positiveBBox, { minLength: 1, maxLength: 8 }),
        fc.array(positiveBBox, { minLength: 1, maxLength: 8 }),
        (preds, dets) => {
          const m = iouMatrix(preds, dets);
          for (let i = 0; i < preds.length; i++) {
            for (let j = 0; j < dets.length; j++) {
              expect(m[i * dets.length + j]).toBeCloseTo(iou(preds[i]!, dets[j]!), 9);
            }
          }
        },
      ),
    );
  });
});
