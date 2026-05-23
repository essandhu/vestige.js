import { describe, it, expect } from 'vitest';
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

const APPROX = 1e-9;

describe('xyxyToXywh', () => {
  it('converts to top-left + width/height', () => {
    expect(xyxyToXywh([10, 20, 30, 50])).toEqual([10, 20, 20, 30]);
  });

  it('handles origin-anchored box', () => {
    expect(xyxyToXywh([0, 0, 100, 50])).toEqual([0, 0, 100, 50]);
  });
});

describe('xywhToXyxy', () => {
  it('is the inverse of xyxyToXywh', () => {
    expect(xywhToXyxy([10, 20, 20, 30])).toEqual([10, 20, 30, 50]);
  });
});

describe('xyxyToCxcywh', () => {
  it('returns center coordinates and size', () => {
    expect(xyxyToCxcywh([10, 20, 30, 50])).toEqual([20, 35, 20, 30]);
  });

  it('handles centered-on-origin box', () => {
    expect(xyxyToCxcywh([-5, -10, 5, 10])).toEqual([0, 0, 10, 20]);
  });
});

describe('cxcywhToXyxy', () => {
  it('is the inverse of xyxyToCxcywh', () => {
    expect(cxcywhToXyxy([20, 35, 20, 30])).toEqual([10, 20, 30, 50]);
  });
});

describe('xyxyToXyah', () => {
  it('returns center, aspect ratio (w/h), and height', () => {
    // [10, 20, 30, 50]  =>  w=20, h=30, cx=20, cy=35, a=20/30
    const [cx, cy, a, h] = xyxyToXyah([10, 20, 30, 50]);
    expect(cx).toBe(20);
    expect(cy).toBe(35);
    expect(a).toBeCloseTo(20 / 30, 12);
    expect(h).toBe(30);
  });

  it('aspect ratio is 1 for square boxes', () => {
    const [, , a] = xyxyToXyah([0, 0, 10, 10]);
    expect(a).toBe(1);
  });
});

describe('xyahToXyxy', () => {
  it('round-trips with xyxyToXyah', () => {
    const original: [number, number, number, number] = [10, 20, 30, 50];
    const xyah = xyxyToXyah(original);
    const back = xyahToXyxy(xyah);
    expect(back[0]).toBeCloseTo(original[0], 9);
    expect(back[1]).toBeCloseTo(original[1], 9);
    expect(back[2]).toBeCloseTo(original[2], 9);
    expect(back[3]).toBeCloseTo(original[3], 9);
  });
});

describe('bboxArea', () => {
  it('computes width times height for positive boxes', () => {
    expect(bboxArea([10, 20, 30, 50])).toBe(600);
    expect(bboxArea([0, 0, 100, 100])).toBe(10000);
  });

  it('returns 0 for zero-area boxes', () => {
    expect(bboxArea([5, 5, 5, 5])).toBe(0);
    expect(bboxArea([5, 5, 5, 10])).toBe(0);
    expect(bboxArea([5, 5, 10, 5])).toBe(0);
  });

  it('returns 0 for negative-area (inverted) boxes', () => {
    expect(bboxArea([30, 50, 10, 20])).toBe(0);
  });
});

describe('clipBBox', () => {
  it('leaves in-bounds boxes unchanged', () => {
    expect(clipBBox([10, 10, 40, 40], 100, 100)).toEqual([10, 10, 40, 40]);
  });

  it('clamps out-of-bounds corners to the image rectangle', () => {
    expect(clipBBox([-10, -10, 50, 50], 100, 100)).toEqual([0, 0, 50, 50]);
    expect(clipBBox([50, 50, 150, 150], 100, 100)).toEqual([50, 50, 100, 100]);
  });

  it('clamps fully out-of-bounds boxes to a degenerate edge', () => {
    const clipped = clipBBox([200, 200, 300, 300], 100, 100);
    // x1 and x2 both >= width  =>  both clamp to width
    expect(clipped[0]).toBe(100);
    expect(clipped[2]).toBe(100);
    expect(bboxArea(clipped)).toBeLessThanOrEqual(APPROX);
  });
});
