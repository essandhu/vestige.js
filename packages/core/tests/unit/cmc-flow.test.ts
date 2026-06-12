import { describe, expect, it } from 'vitest';
import type { GrayFrame } from '../../src/plugins/cmc.js';
import { estimateSimilarity, trackPointsLk } from '../../src/plugins/cmc.js';

/**
 * Smooth analytic texture with gradients everywhere — periods of 45–90 px,
 * well inside the default 21 px LK window's linearization range. Sampling
 * the function at warped coordinates produces an exactly-warped frame with
 * no resampling error, so the expected flow is the warp itself.
 */
const texture = (x: number, y: number): number =>
  128 +
  40 * Math.sin(x * 0.12) * Math.cos(y * 0.1) +
  30 * Math.sin(x * 0.07 + y * 0.05 + 1) +
  25 * Math.cos(x * 0.04 - y * 0.09 + 2);

const render = (
  width: number,
  height: number,
  warp: (x: number, y: number) => readonly [number, number] = (x, y) => [x, y],
): GrayFrame => {
  const data = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [sx, sy] = warp(x, y);
      data[y * width + x] = texture(sx, sy);
    }
  }
  return { data, width, height };
};

describe('trackPointsLk', () => {
  it('recovers a pure translation to sub-pixel accuracy', () => {
    // Features move by t = (2.7, −1.3): curr(q) = texture(q − t).
    const tx = 2.7;
    const ty = -1.3;
    const prev = render(128, 96);
    const curr = render(128, 96, (x, y) => [x - tx, y - ty]);

    const points: Array<readonly [number, number]> = [
      [40, 40],
      [64, 48],
      [88, 56],
    ];
    const tracked = trackPointsLk(prev, curr, points);

    expect(tracked).toHaveLength(3);
    for (let i = 0; i < points.length; i++) {
      const p = points[i] ?? [0, 0];
      const q = tracked[i] ?? null;
      expect(q, `point ${i} lost`).not.toBeNull();
      if (q === null) continue;
      expect(q[0]).toBeCloseTo(p[0] + tx, 1);
      expect(q[1]).toBeCloseTo(p[1] + ty, 1);
    }
  });

  it('returns null for points whose window leaves the frame', () => {
    const prev = render(128, 96);
    const curr = render(128, 96, (x, y) => [x - 1, y]);
    const tracked = trackPointsLk(prev, curr, [[2, 2]]);
    expect(tracked).toEqual([null]);
  });

  it('returns null on gradient-free (flat) regions', () => {
    const flat: GrayFrame = { data: new Float32Array(128 * 96).fill(50), width: 128, height: 96 };
    const tracked = trackPointsLk(flat, flat, [[64, 48]]);
    expect(tracked).toEqual([null]);
  });
});

describe('estimateSimilarity', () => {
  it('recovers an exact translation', () => {
    const src: Array<readonly [number, number]> = [
      [10, 10],
      [60, 12],
      [35, 70],
      [80, 55],
    ];
    const dst = src.map(([x, y]) => [x + 5, y - 3] as const);

    const m = estimateSimilarity(src, dst);
    expect(m).not.toBeNull();
    if (m === null) return;
    const expected = [1, 0, 5, 0, 1, -3];
    for (let i = 0; i < 6; i++) expect(m[i]).toBeCloseTo(expected[i] ?? 0, 9);
  });

  it('recovers rotation + uniform scale + translation', () => {
    // s = 1.5, θ = 30°: a = s·cosθ, b = s·sinθ, t = (2, 1).
    const a = 1.5 * Math.cos(Math.PI / 6);
    const b = 1.5 * Math.sin(Math.PI / 6);
    const src: Array<readonly [number, number]> = [
      [0, 0],
      [40, 5],
      [10, 50],
      [60, 60],
      [25, 20],
      [55, 35],
    ];
    const dst = src.map(([x, y]) => [a * x - b * y + 2, b * x + a * y + 1] as const);

    const m = estimateSimilarity(src, dst);
    expect(m).not.toBeNull();
    if (m === null) return;
    const expected = [a, -b, 2, b, a, 1];
    for (let i = 0; i < 6; i++) expect(m[i]).toBeCloseTo(expected[i] ?? 0, 9);
  });

  it('rejects gross outliers via RANSAC', () => {
    const src: Array<readonly [number, number]> = [
      [10, 10],
      [60, 12],
      [35, 70],
      [80, 55],
      [20, 40],
      [70, 30],
      [45, 25],
      [15, 65],
      // outlier pair sources
      [50, 50],
      [30, 30],
    ];
    const dst = src.map(([x, y]) => [x + 5, y - 3] as const);
    dst[8] = [120, 7];
    dst[9] = [0, 90];

    const m = estimateSimilarity(src, dst);
    expect(m).not.toBeNull();
    if (m === null) return;
    const expected = [1, 0, 5, 0, 1, -3];
    for (let i = 0; i < 6; i++) expect(m[i]).toBeCloseTo(expected[i] ?? 0, 6);
  });

  it('returns null below minPoints', () => {
    const src: Array<readonly [number, number]> = [
      [0, 0],
      [10, 0],
      [0, 10],
    ];
    const dst = src.map(([x, y]) => [x + 1, y] as const);
    expect(estimateSimilarity(src, dst)).toBeNull();
  });

  it('returns null for degenerate (coincident) points', () => {
    const src: Array<readonly [number, number]> = [
      [5, 5],
      [5, 5],
      [5, 5],
      [5, 5],
    ];
    const dst: Array<readonly [number, number]> = [
      [7, 5],
      [7, 5],
      [7, 5],
      [7, 5],
    ];
    expect(estimateSimilarity(src, dst)).toBeNull();
  });

  it('is deterministic for a fixed seed', () => {
    const src: Array<readonly [number, number]> = [
      [10, 10],
      [60, 12],
      [35, 70],
      [80, 55],
      [20, 40],
      [70, 30],
    ];
    const dst = src.map(([x, y]) => [x + 2.5, y + 4.25] as const);
    dst[5] = [200, 200];

    const m1 = estimateSimilarity(src, dst);
    const m2 = estimateSimilarity(src, dst);
    expect(m1).not.toBeNull();
    expect(m1).toEqual(m2);
  });
});
