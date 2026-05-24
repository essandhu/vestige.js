import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { ciou, diou, giou, iou, iouMatrix } from '../../src/geometry/iou.js';
import type { BBox } from '../../src/types.js';

const positiveBBox = fc
  .tuple(
    fc.float({ min: -500, max: 500, noNaN: true, noDefaultInfinity: true }),
    fc.float({ min: -500, max: 500, noNaN: true, noDefaultInfinity: true }),
    fc.float({ min: 0.5, max: 200, noNaN: true, noDefaultInfinity: true }),
    fc.float({ min: 0.5, max: 200, noNaN: true, noDefaultInfinity: true }),
  )
  .map(([x, y, w, h]) => [x, y, x + w, y + h] as BBox);

// Suites are skipped until iou.ts is implemented on `feature/iou`. Un-skip them as the
// first commit of that branch so the impl PR shows the red → green arc.

describe.skip('IoU invariants', () => {
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

describe.skip('GIoU invariants', () => {
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

describe.skip('DIoU invariants', () => {
  it('is in [-1, 1]', () => {
    fc.assert(
      fc.property(positiveBBox, positiveBBox, (a, b) => {
        const v = diou(a, b);
        expect(v).toBeGreaterThanOrEqual(-1 - 1e-9);
        expect(v).toBeLessThanOrEqual(1 + 1e-9);
      }),
    );
  });

  it('diou <= iou for all box pairs', () => {
    fc.assert(
      fc.property(positiveBBox, positiveBBox, (a, b) => {
        expect(diou(a, b)).toBeLessThanOrEqual(iou(a, b) + 1e-9);
      }),
    );
  });

  it('diou(a, a) === 1 for positive-area boxes', () => {
    fc.assert(
      fc.property(positiveBBox, (a) => {
        expect(diou(a, a)).toBeCloseTo(1, 9);
      }),
    );
  });

  it('is symmetric', () => {
    fc.assert(
      fc.property(positiveBBox, positiveBBox, (a, b) => {
        expect(diou(a, b)).toBeCloseTo(diou(b, a), 9);
      }),
    );
  });
});

describe.skip('CIoU invariants', () => {
  it('is bounded above by 1 (aspect term is non-negative)', () => {
    fc.assert(
      fc.property(positiveBBox, positiveBBox, (a, b) => {
        expect(ciou(a, b)).toBeLessThanOrEqual(1 + 1e-9);
      }),
    );
  });

  it('ciou <= diou for all box pairs', () => {
    fc.assert(
      fc.property(positiveBBox, positiveBBox, (a, b) => {
        expect(ciou(a, b)).toBeLessThanOrEqual(diou(a, b) + 1e-9);
      }),
    );
  });

  it('ciou(a, a) === 1 for positive-area boxes', () => {
    fc.assert(
      fc.property(positiveBBox, (a) => {
        expect(ciou(a, a)).toBeCloseTo(1, 9);
      }),
    );
  });

  it('is symmetric', () => {
    fc.assert(
      fc.property(positiveBBox, positiveBBox, (a, b) => {
        expect(ciou(a, b)).toBeCloseTo(ciou(b, a), 9);
      }),
    );
  });
});

describe.skip('iouMatrix matches scalar iou', () => {
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
