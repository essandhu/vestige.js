import { describe, expect, it } from 'vitest';
import { CvXyahMotionModel } from '../../src/filters/motion-models/cv-xyah.js';

function expectCloseArray(actual: Float64Array, expected: ArrayLike<number>, eps = 1e-10) {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i] as number, -Math.log10(eps));
  }
}

const WP = 1 / 20;
const WV = 1 / 160;

describe('CvXyahMotionModel — dimensions and constant matrices', () => {
  it('exposes stateDim=8 and measDim=4', () => {
    const m = new CvXyahMotionModel();
    expect(m.stateDim).toBe(8);
    expect(m.measDim).toBe(4);
  });

  it('F is the 8x8 constant-velocity matrix [[I, I], [0, I]]', () => {
    const m = new CvXyahMotionModel();
    // biome-ignore format: matrix row layout
    const expected = [
      1, 0, 0, 0, 1, 0, 0, 0,
      0, 1, 0, 0, 0, 1, 0, 0,
      0, 0, 1, 0, 0, 0, 1, 0,
      0, 0, 0, 1, 0, 0, 0, 1,
      0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 1,
    ];
    expectCloseArray(m.F, expected);
  });

  it('H is the 4x8 [I_4 | 0] measurement matrix', () => {
    const m = new CvXyahMotionModel();
    // biome-ignore format: matrix row layout
    const expected = [
      1, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0,
    ];
    expectCloseArray(m.H, expected);
  });

  it('exposes the DeepSORT default weights', () => {
    const m = new CvXyahMotionModel();
    expect(m.stdWeightPosition).toBeCloseTo(WP, 12);
    expect(m.stdWeightVelocity).toBeCloseTo(WV, 12);
  });

  it('honors user-provided weights', () => {
    const m = new CvXyahMotionModel({ stdWeightPosition: 0.1, stdWeightVelocity: 0.001 });
    expect(m.stdWeightPosition).toBeCloseTo(0.1, 12);
    expect(m.stdWeightVelocity).toBeCloseTo(0.001, 12);
  });
});

describe('CvXyahMotionModel — state-dependent process noise', () => {
  it('Q scales with h: diag([(wp h)², (wp h)², 1e-4, (wp h)², (wv h)², (wv h)², 1e-10, (wv h)²])', () => {
    const m = new CvXyahMotionModel();
    const h = 100;
    const Q = m.processNoise(new Float64Array([10, 20, 0.5, h, 0, 0, 0, 0]));
    const expected = new Float64Array(64);
    const diag = [
      (WP * h) ** 2,
      (WP * h) ** 2,
      1e-4,
      (WP * h) ** 2,
      (WV * h) ** 2,
      (WV * h) ** 2,
      1e-10,
      (WV * h) ** 2,
    ];
    for (let i = 0; i < 8; i++) expected[i * 8 + i] = diag[i]!;
    expectCloseArray(Q, expected);
  });

  it('Q at h=200 has every position-block diagonal 4× the Q at h=100', () => {
    const m = new CvXyahMotionModel();
    const Q100 = m.processNoise(new Float64Array([0, 0, 0.5, 100, 0, 0, 0, 0]));
    const Q200 = m.processNoise(new Float64Array([0, 0, 0.5, 200, 0, 0, 0, 0]));
    // diagonals at 0, 1, 3 (position xy and h) scale with h²
    for (const i of [0, 1, 3]) {
      expect(Q200[i * 8 + i]!).toBeCloseTo(4 * Q100[i * 8 + i]!, 10);
    }
    // diagonal at 2 (aspect ratio) is constant 1e-4 regardless of h
    expect(Q100[2 * 8 + 2]!).toBeCloseTo(1e-4, 12);
    expect(Q200[2 * 8 + 2]!).toBeCloseTo(1e-4, 12);
  });
});

describe('CvXyahMotionModel — state-dependent measurement noise', () => {
  it('R scales with h: diag([(wp h)², (wp h)², 1e-2, (wp h)²])', () => {
    const m = new CvXyahMotionModel();
    const h = 100;
    const R = m.measurementNoise(new Float64Array([10, 20, 0.5, h, 0, 0, 0, 0]));
    const expected = new Float64Array(16);
    const diag = [(WP * h) ** 2, (WP * h) ** 2, 1e-2, (WP * h) ** 2];
    for (let i = 0; i < 4; i++) expected[i * 4 + i] = diag[i]!;
    expectCloseArray(R, expected);
  });
});

describe('CvXyahMotionModel.init', () => {
  it('mean has measurement in slots 0..3 and zero velocities', () => {
    const m = new CvXyahMotionModel();
    const s = m.init(new Float64Array([10, 20, 0.5, 100]));
    expectCloseArray(s.mean, [10, 20, 0.5, 100, 0, 0, 0, 0]);
  });

  it('initial covariance matches DeepSORT initiate() exactly for h=100', () => {
    const m = new CvXyahMotionModel();
    const h = 100;
    const s = m.init(new Float64Array([10, 20, 0.5, h]));
    const expected = new Float64Array(64);
    const std = [
      2 * WP * h,
      2 * WP * h,
      1e-2,
      2 * WP * h,
      10 * WV * h,
      10 * WV * h,
      1e-5,
      10 * WV * h,
    ];
    for (let i = 0; i < 8; i++) expected[i * 8 + i] = std[i]! * std[i]!;
    expectCloseArray(s.covariance, expected);
  });

  it('initial covariance at h=200 quadruples the position-block diagonal of h=100', () => {
    const m = new CvXyahMotionModel();
    const a = m.init(new Float64Array([0, 0, 0.5, 100]));
    const b = m.init(new Float64Array([0, 0, 0.5, 200]));
    for (const i of [0, 1, 3, 4, 5, 7]) {
      expect(b.covariance[i * 8 + i]!).toBeCloseTo(4 * a.covariance[i * 8 + i]!, 8);
    }
  });
});
