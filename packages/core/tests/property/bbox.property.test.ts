import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import {
  xyxyToXywh,
  xywhToXyxy,
  xyxyToCxcywh,
  cxcywhToXyxy,
  xyxyToXyah,
  xyahToXyxy,
  bboxArea,
  clipBBox,
} from '../../src/geometry/bbox.js';

/**
 * Arbitrary that yields a positive-area xyxy bbox with bounded coordinates and
 * a height strictly greater than zero (required for xyah conversions).
 */
const positiveBBox = fc
  .tuple(
    fc.float({ min: -1000, max: 1000, noNaN: true, noDefaultInfinity: true }),
    fc.float({ min: -1000, max: 1000, noNaN: true, noDefaultInfinity: true }),
    fc.float({ min: 0.5, max: 500, noNaN: true, noDefaultInfinity: true }),
    fc.float({ min: 0.5, max: 500, noNaN: true, noDefaultInfinity: true }),
  )
  .map(([x, y, w, h]) => [x, y, x + w, y + h] as const);

const EPS = 1e-6;

function approxTuple(a: readonly number[], b: readonly number[]) {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) {
    expect(Math.abs((a[i] as number) - (b[i] as number))).toBeLessThan(EPS);
  }
}

describe('bbox conversions are bijective', () => {
  it('xyxy -> xywh -> xyxy is identity', () => {
    fc.assert(
      fc.property(positiveBBox, (b) => {
        approxTuple(xywhToXyxy(xyxyToXywh(b)), Array.from(b));
      }),
    );
  });

  it('xyxy -> cxcywh -> xyxy is identity', () => {
    fc.assert(
      fc.property(positiveBBox, (b) => {
        approxTuple(cxcywhToXyxy(xyxyToCxcywh(b)), Array.from(b));
      }),
    );
  });

  it('xyxy -> xyah -> xyxy is identity for positive-height boxes', () => {
    fc.assert(
      fc.property(positiveBBox, (b) => {
        approxTuple(xyahToXyxy(xyxyToXyah(b)), Array.from(b));
      }),
    );
  });
});

describe('bboxArea invariants', () => {
  it('is non-negative for any input', () => {
    fc.assert(
      fc.property(
        fc.tuple(
          fc.float({ noNaN: true, noDefaultInfinity: true }),
          fc.float({ noNaN: true, noDefaultInfinity: true }),
          fc.float({ noNaN: true, noDefaultInfinity: true }),
          fc.float({ noNaN: true, noDefaultInfinity: true }),
        ),
        (b) => {
          expect(bboxArea(b as unknown as readonly [number, number, number, number])).toBeGreaterThanOrEqual(0);
        },
      ),
    );
  });

  it('matches width * height for positive-area boxes', () => {
    fc.assert(
      fc.property(positiveBBox, (b) => {
        const expected = (b[2] - b[0]) * (b[3] - b[1]);
        expect(bboxArea(b)).toBeCloseTo(expected, 6);
      }),
    );
  });
});

describe('clipBBox stays within image bounds', () => {
  it('result corners are within [0, width] x [0, height]', () => {
    fc.assert(
      fc.property(
        positiveBBox,
        fc.integer({ min: 1, max: 4096 }),
        fc.integer({ min: 1, max: 4096 }),
        (b, w, h) => {
          const [x1, y1, x2, y2] = clipBBox(b, w, h);
          expect(x1).toBeGreaterThanOrEqual(0);
          expect(y1).toBeGreaterThanOrEqual(0);
          expect(x2).toBeLessThanOrEqual(w);
          expect(y2).toBeLessThanOrEqual(h);
          expect(x1).toBeLessThanOrEqual(x2);
          expect(y1).toBeLessThanOrEqual(y2);
        },
      ),
    );
  });
});
