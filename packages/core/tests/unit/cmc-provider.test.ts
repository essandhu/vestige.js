import { describe, expect, it } from 'vitest';
import type { GrayFrame } from '../../src/plugins/cmc.js';
import { SparseOpticalFlowCmc, warpBBox } from '../../src/plugins/cmc.js';

// End-to-end provider tests on analytically warped frames. The texture is
// evaluated at warped coordinates, so the second frame is an exact warp of
// the first (no resampling error) and the ground-truth matrix is known.

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

describe('SparseOpticalFlowCmc', () => {
  it('recovers a pure camera translation', async () => {
    // Scene content moves by t = (3.2, −2.4): curr(q) = prev(q − t),
    // so the prev → curr warp is [1, 0, 3.2, 0, 1, −2.4].
    const tx = 3.2;
    const ty = -2.4;
    const prev = render(192, 144);
    const curr = render(192, 144, (x, y) => [x - tx, y - ty]);

    const cmc = new SparseOpticalFlowCmc({ maxCorners: 300 });
    const m = await cmc.estimate(prev, curr);

    expect(m).not.toBeNull();
    if (m === null) return;
    expect(m[0]).toBeCloseTo(1, 2);
    expect(m[1]).toBeCloseTo(0, 2);
    expect(m[3]).toBeCloseTo(0, 2);
    expect(m[4]).toBeCloseTo(1, 2);
    expect(Math.abs((m[2] ?? 0) - tx)).toBeLessThan(0.3);
    expect(Math.abs((m[5] ?? 0) - ty)).toBeLessThan(0.3);
  });

  it('recovers a small rotation about the image center plus translation', async () => {
    // θ = 0.02 rad about c = (96, 72), then t = (1.5, −1):
    // q = R·(p − c) + c + t, i.e. warp [cosθ, −sinθ, tx', sinθ, cosθ, ty']
    // with [tx', ty'] = c + t − R·c. curr(q) = prev(R⁻¹·(q − c − t) + c).
    const theta = 0.02;
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    const cx = 96;
    const cy = 72;
    const tx = 1.5;
    const ty = -1;

    const prev = render(192, 144);
    const curr = render(192, 144, (x, y) => {
      const dx = x - cx - tx;
      const dy = y - cy - ty;
      return [cos * dx + sin * dy + cx, -sin * dx + cos * dy + cy];
    });

    const cmc = new SparseOpticalFlowCmc({ maxCorners: 300 });
    const m = await cmc.estimate(prev, curr);

    expect(m).not.toBeNull();
    if (m === null) return;
    expect(m[0]).toBeCloseTo(cos, 2);
    expect(m[1]).toBeCloseTo(-sin, 2);
    expect(m[3]).toBeCloseTo(sin, 2);
    expect(m[4]).toBeCloseTo(cos, 2);
    expect(Math.abs((m[2] ?? 0) - (cx + tx - cos * cx + sin * cy))).toBeLessThan(0.5);
    expect(Math.abs((m[5] ?? 0) - (cy + ty - sin * cx - cos * cy))).toBeLessThan(0.5);
  });

  it('returns null for textureless frames', async () => {
    const flat: GrayFrame = { data: new Float32Array(192 * 144).fill(80), width: 192, height: 144 };
    const cmc = new SparseOpticalFlowCmc();
    expect(await cmc.estimate(flat, flat)).toBeNull();
  });

  it('is deterministic: identical inputs give a bit-identical matrix', async () => {
    const prev = render(192, 144);
    const curr = render(192, 144, (x, y) => [x - 2, y + 1]);
    const cmc = new SparseOpticalFlowCmc({ maxCorners: 200 });

    const a = await cmc.estimate(prev, curr);
    const b = await cmc.estimate(prev, curr);
    expect(a).not.toBeNull();
    expect(a).toEqual(b);
  });

  it('rejects non-GrayFrame inputs with a specific error', async () => {
    const cmc = new SparseOpticalFlowCmc();
    const good = render(32, 32);
    await expect(cmc.estimate({}, good)).rejects.toThrow(/grayscale/i);
    await expect(cmc.estimate(good, 'frame')).rejects.toThrow(/grayscale/i);
  });

  it('rejects frames whose buffer length disagrees with the dimensions', async () => {
    const cmc = new SparseOpticalFlowCmc();
    const bad: GrayFrame = { data: new Float32Array(10), width: 32, height: 32 };
    await expect(cmc.estimate(bad, render(32, 32))).rejects.toThrow(/grayscale/i);
  });

  it('rejects frame pairs of different sizes', async () => {
    const cmc = new SparseOpticalFlowCmc();
    await expect(cmc.estimate(render(32, 32), render(64, 32))).rejects.toThrow(/size/i);
  });
});

describe('warpBBox', () => {
  it('translates a bbox', () => {
    const m = Float64Array.from([1, 0, 5, 0, 1, -3]);
    expect(warpBBox(m, [10, 20, 30, 40])).toEqual([15, 17, 35, 37]);
  });

  it('re-normalizes corner order under reflection-like warps', () => {
    // Pure 180° rotation about the origin: corners swap quadrants.
    const m = Float64Array.from([-1, 0, 0, 0, -1, 0]);
    expect(warpBBox(m, [10, 20, 30, 40])).toEqual([-30, -40, -10, -20]);
  });

  it('scales about the origin', () => {
    const m = Float64Array.from([2, 0, 0, 0, 2, 0]);
    expect(warpBBox(m, [1, 2, 3, 4])).toEqual([2, 4, 6, 8]);
  });
});
