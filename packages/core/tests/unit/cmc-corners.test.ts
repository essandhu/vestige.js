import { describe, expect, it } from 'vitest';
import type { GrayFrame } from '../../src/plugins/cmc.js';
import { sampleBilinear, shiTomasiCorners } from '../../src/plugins/cmc.js';

const grayFrame = (width: number, height: number, fill: (x: number, y: number) => number) => {
  const data = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) data[y * width + x] = fill(x, y);
  }
  return { data, width, height } satisfies GrayFrame;
};

/** 60×60 black frame with a white 20×20 square spanning x,y ∈ [20, 40). */
const squareFrame = () =>
  grayFrame(60, 60, (x, y) => (x >= 20 && x < 40 && y >= 20 && y < 40 ? 255 : 0));

describe('sampleBilinear', () => {
  // 2×2 image [[0, 10], [20, 30]] — every oracle value is exact by hand.
  const img = { data: Float32Array.from([0, 10, 20, 30]), width: 2, height: 2 };

  it('returns exact values at integer positions', () => {
    expect(sampleBilinear(img, 0, 0)).toBe(0);
    expect(sampleBilinear(img, 1, 0)).toBe(10);
    expect(sampleBilinear(img, 0, 1)).toBe(20);
    expect(sampleBilinear(img, 1, 1)).toBe(30);
  });

  it('interpolates linearly between samples', () => {
    // The image is exactly the plane 10x + 20y, which bilinear reproduces.
    expect(sampleBilinear(img, 0.5, 0)).toBeCloseTo(5, 12);
    expect(sampleBilinear(img, 0, 0.5)).toBeCloseTo(10, 12);
    expect(sampleBilinear(img, 0.5, 0.5)).toBeCloseTo(15, 12);
    expect(sampleBilinear(img, 0.25, 0.75)).toBeCloseTo(17.5, 12);
  });

  it('clamps out-of-range coordinates to the edge', () => {
    expect(sampleBilinear(img, -3, 0)).toBe(0);
    expect(sampleBilinear(img, 5, 5)).toBe(30);
  });
});

describe('shiTomasiCorners', () => {
  it('finds the four corners of a high-contrast square', () => {
    const corners = shiTomasiCorners(squareFrame(), { maxCorners: 8, minDistance: 5 });
    // The square's true corners (inclusive pixel bounds [20, 39]).
    const truth: Array<[number, number]> = [
      [20, 20],
      [39, 20],
      [20, 39],
      [39, 39],
    ];
    for (const [tx, ty] of truth) {
      const hit = corners.some(([x, y]) => Math.hypot(x - tx, y - ty) <= 3);
      expect(hit, `no corner detected near (${tx}, ${ty})`).toBe(true);
    }
  });

  it('returns an empty array for a flat frame', () => {
    expect(shiTomasiCorners(grayFrame(40, 40, () => 128))).toEqual([]);
  });

  it('caps the number of returned corners at maxCorners', () => {
    expect(shiTomasiCorners(squareFrame(), { maxCorners: 2 }).length).toBeLessThanOrEqual(2);
  });

  it('suppresses neighbors within minDistance', () => {
    // All four square corners are within hypot(19,19) ≈ 26.9 px of each
    // other, so a 50 px minimum distance leaves exactly the strongest one.
    const corners = shiTomasiCorners(squareFrame(), { maxCorners: 10, minDistance: 50 });
    expect(corners).toHaveLength(1);
  });

  it('is deterministic', () => {
    const a = shiTomasiCorners(squareFrame(), { maxCorners: 8 });
    const b = shiTomasiCorners(squareFrame(), { maxCorners: 8 });
    expect(a).toEqual(b);
  });
});
