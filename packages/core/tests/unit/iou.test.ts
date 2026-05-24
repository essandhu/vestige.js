import { describe, expect, it } from 'vitest';
import { ciou, diou, giou, giouMatrix, iou, iouMatrix } from '../../src/geometry/iou.js';
import type { BBox } from '../../src/types.js';

describe('iou', () => {
  it('returns 1 for identical boxes', () => {
    expect(iou([0, 0, 10, 10], [0, 0, 10, 10])).toBe(1);
  });

  it('returns 0 for disjoint boxes', () => {
    expect(iou([0, 0, 10, 10], [20, 20, 30, 30])).toBe(0);
    expect(iou([0, 0, 10, 10], [11, 0, 20, 10])).toBe(0);
  });

  it('computes a half-overlap case correctly', () => {
    // a area 100, b area 100, intersect = [0,5,10,10] area 50, union = 150
    expect(iou([0, 0, 10, 10], [0, 5, 10, 15])).toBeCloseTo(50 / 150, 12);
  });

  it('handles one box fully inside another', () => {
    // a area 100, b area 36, intersect = b, union = 100
    expect(iou([0, 0, 10, 10], [2, 2, 8, 8])).toBeCloseTo(36 / 100, 12);
  });

  it('is symmetric', () => {
    const a: BBox = [3, 4, 13, 14];
    const b: BBox = [7, 8, 20, 25];
    expect(iou(a, b)).toBeCloseTo(iou(b, a), 12);
  });

  it('returns 0 when either box has zero area', () => {
    expect(iou([5, 5, 5, 5], [0, 0, 10, 10])).toBe(0);
    expect(iou([0, 0, 10, 10], [5, 5, 5, 5])).toBe(0);
  });
});

describe('giou', () => {
  it('equals IoU when one box contains the other', () => {
    // smallest enclosing box equals the union of the two boxes => giou == iou
    const a: BBox = [0, 0, 10, 10];
    const b: BBox = [2, 2, 8, 8];
    expect(giou(a, b)).toBeCloseTo(iou(a, b), 12);
  });

  it('is negative for disjoint boxes', () => {
    // a, b each area 100, disjoint, enclosing = [0,0,30,10] area 300, union 200
    // giou = 0 - (300 - 200) / 300 = -1/3
    expect(giou([0, 0, 10, 10], [20, 0, 30, 10])).toBeCloseTo(-1 / 3, 12);
  });

  it('is bounded by [-1, 1]', () => {
    expect(giou([0, 0, 10, 10], [0, 0, 10, 10])).toBeCloseTo(1, 12);
    expect(giou([0, 0, 1, 1], [1000, 1000, 1001, 1001])).toBeGreaterThanOrEqual(-1);
  });

  it('is never greater than IoU', () => {
    const cases: Array<[BBox, BBox]> = [
      [
        [0, 0, 10, 10],
        [5, 5, 15, 15],
      ],
      [
        [0, 0, 10, 10],
        [20, 0, 30, 10],
      ],
      [
        [0, 0, 10, 10],
        [2, 2, 8, 8],
      ],
      [
        [0, 0, 100, 50],
        [10, 10, 20, 40],
      ],
    ];
    for (const [a, b] of cases) {
      expect(giou(a, b)).toBeLessThanOrEqual(iou(a, b) + 1e-12);
    }
  });
});

describe('diou', () => {
  it('returns 1 for identical boxes', () => {
    expect(diou([0, 0, 10, 10], [0, 0, 10, 10])).toBe(1);
  });

  it('equals IoU when centers coincide', () => {
    // b is fully inside a, both centered at (5, 5)  =>  rho^2 = 0  =>  diou = iou.
    const a: BBox = [0, 0, 10, 10];
    const b: BBox = [2, 2, 8, 8];
    expect(diou(a, b)).toBeCloseTo(iou(a, b), 12);
  });

  it('matches the Zheng et al. AAAI 2020 formula on a half-overlap case', () => {
    // a = [0,0,10,10], b = [0,5,10,15]:
    //   IoU = 50/150 = 1/3; centers (5,5) and (5,10) => rho^2 = 25;
    //   enclosing [0,0,10,15] => c^2 = 10^2 + 15^2 = 325.
    //   DIoU = 1/3 - 25/325 = 10/39.
    expect(diou([0, 0, 10, 10], [0, 5, 10, 15])).toBeCloseTo(10 / 39, 12);
  });

  it('is negative for disjoint boxes', () => {
    // a, b each 10x10, horizontally separated by a 10-px gap:
    //   IoU = 0; rho^2 = (25 - 5)^2 = 400; enclosing [0,0,30,10] => c^2 = 1000.
    //   DIoU = 0 - 400/1000 = -0.4.
    expect(diou([0, 0, 10, 10], [20, 0, 30, 10])).toBeCloseTo(-0.4, 12);
  });

  it('is symmetric', () => {
    const a: BBox = [3, 4, 13, 14];
    const b: BBox = [7, 8, 20, 25];
    expect(diou(a, b)).toBeCloseTo(diou(b, a), 12);
  });
});

describe('ciou', () => {
  it('returns 1 for identical boxes', () => {
    expect(ciou([0, 0, 10, 10], [0, 0, 10, 10])).toBe(1);
  });

  it('equals DIoU when aspect ratios match (v = 0)', () => {
    // Both boxes are 10x10 squares  =>  atan(w/h) terms cancel  =>  v = 0.
    const a: BBox = [0, 0, 10, 10];
    const b: BBox = [0, 5, 10, 15];
    expect(ciou(a, b)).toBeCloseTo(diou(a, b), 12);
  });

  it('matches the Zheng et al. AAAI 2020 formula on an aspect-mismatch case', () => {
    // a = [0,0,10,10]  (w/h = 1), b = [0,0,20,10]  (w/h = 2), top-aligned.
    //   IoU = 100/200 = 0.5; centers (5,5) and (10,5) => rho^2 = 25;
    //   enclosing [0,0,20,10] => c^2 = 500.
    // CIoU = IoU - rho^2/c^2 - alpha*v, with v = (4/pi^2)*(atan(1) - atan(2))^2
    // and alpha = v / ((1 - IoU) + v).
    const a: BBox = [0, 0, 10, 10];
    const b: BBox = [0, 0, 20, 10];
    const iouVal = 0.5;
    const rhoSq = 25;
    const cSq = 500;
    const v = (4 / Math.PI ** 2) * (Math.atan(1) - Math.atan(2)) ** 2;
    const alpha = v / (1 - iouVal + v);
    const expected = iouVal - rhoSq / cSq - alpha * v;
    expect(ciou(a, b)).toBeCloseTo(expected, 12);
  });

  it('is at most DIoU (aspect term is non-negative)', () => {
    const cases: Array<[BBox, BBox]> = [
      [
        [0, 0, 10, 10],
        [2, 3, 8, 7],
      ],
      [
        [0, 0, 100, 50],
        [10, 10, 60, 40],
      ],
      [
        [0, 0, 10, 10],
        [0, 5, 10, 15],
      ],
    ];
    for (const [a, b] of cases) {
      expect(ciou(a, b)).toBeLessThanOrEqual(diou(a, b) + 1e-12);
    }
  });

  it('is symmetric', () => {
    const a: BBox = [3, 4, 13, 14];
    const b: BBox = [7, 8, 20, 25];
    expect(ciou(a, b)).toBeCloseTo(ciou(b, a), 12);
  });
});

describe('iouMatrix', () => {
  it('lays out values in row-major order matching scalar iou', () => {
    const preds: BBox[] = [
      [0, 0, 10, 10],
      [20, 20, 30, 30],
    ];
    const dets: BBox[] = [
      [0, 0, 10, 10],
      [5, 5, 15, 15],
      [25, 25, 35, 35],
    ];
    const m = iouMatrix(preds, dets);
    expect(m).toBeInstanceOf(Float64Array);
    expect(m.length).toBe(preds.length * dets.length);
    for (let i = 0; i < preds.length; i++) {
      for (let j = 0; j < dets.length; j++) {
        expect(m[i * dets.length + j]).toBeCloseTo(iou(preds[i]!, dets[j]!), 12);
      }
    }
  });

  it('returns an empty matrix when either side is empty', () => {
    expect(iouMatrix([], [[0, 0, 10, 10]]).length).toBe(0);
    expect(iouMatrix([[0, 0, 10, 10]], []).length).toBe(0);
  });
});

describe('giouMatrix', () => {
  it('matches scalar giou cell-by-cell', () => {
    const preds: BBox[] = [
      [0, 0, 10, 10],
      [50, 50, 60, 60],
    ];
    const dets: BBox[] = [
      [2, 2, 8, 8],
      [20, 0, 30, 10],
    ];
    const m = giouMatrix(preds, dets);
    for (let i = 0; i < preds.length; i++) {
      for (let j = 0; j < dets.length; j++) {
        expect(m[i * dets.length + j]).toBeCloseTo(giou(preds[i]!, dets[j]!), 12);
      }
    }
  });
});
