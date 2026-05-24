import { describe, expect, it } from 'vitest';
import {
  CvBBoxMotionModel,
  xysrToXyxy,
  xyxyToXysr,
} from '../../src/filters/motion-models/cv-bbox.js';

function expectCloseArray(actual: Float64Array, expected: ArrayLike<number>, eps = 1e-10) {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i] as number, -Math.log10(eps));
  }
}

describe('xyxyToXysr / xysrToXyxy', () => {
  it('converts a known box: [10, 20, 30, 60] → [20, 40, 800, 0.5]', () => {
    // w = 20, h = 40, area = 800, aspect = 0.5
    const got = xyxyToXysr([10, 20, 30, 60]);
    expectCloseArray(got, [20, 40, 800, 0.5]);
  });

  it('round-trips xyxy → xysr → xyxy within float tolerance', () => {
    const cases: Array<[number, number, number, number]> = [
      [0, 0, 10, 10],
      [10, 20, 30, 60],
      [-5, -5, 5, 5],
      [100, 200, 250, 500],
    ];
    for (const b of cases) {
      const xysr = xyxyToXysr(b);
      const back = xysrToXyxy(xysr);
      for (let i = 0; i < 4; i++) {
        expect(back[i]!).toBeCloseTo(b[i]!, 10);
      }
    }
  });

  it('xysrToXyxy with negative area returns a degenerate box at the center (per SORT)', () => {
    // SORT's convert_x_to_bbox clamps negative s by treating w = sqrt(s*r) and
    // h = s / w as NaN-free even when s < 0; abewley/sort returns the center
    // point in that case. We model it as a zero-area box at (cx, cy).
    const out = xysrToXyxy(new Float64Array([10, 20, -1, 1]));
    expect(out[0]).toBeCloseTo(10, 10);
    expect(out[1]).toBeCloseTo(20, 10);
    expect(out[2]).toBeCloseTo(10, 10);
    expect(out[3]).toBeCloseTo(20, 10);
  });
});

describe('CvBBoxMotionModel — dimensions and constant matrices', () => {
  it('exposes stateDim=7 and measDim=4', () => {
    const m = new CvBBoxMotionModel();
    expect(m.stateDim).toBe(7);
    expect(m.measDim).toBe(4);
  });

  it('F is the 7x7 constant-velocity matrix from abewley/sort', () => {
    const m = new CvBBoxMotionModel();
    // biome-ignore format: matrix row layout
    const expected = [
      1, 0, 0, 0, 1, 0, 0,
      0, 1, 0, 0, 0, 1, 0,
      0, 0, 1, 0, 0, 0, 1,
      0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 0,
      0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 1,
    ];
    expectCloseArray(m.F, expected);
  });

  it('H is the 4x7 [I_4 | 0] measurement matrix', () => {
    const m = new CvBBoxMotionModel();
    // biome-ignore format: matrix row layout
    const expected = [
      1, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0,
    ];
    expectCloseArray(m.H, expected);
  });
});

describe('CvBBoxMotionModel — noise covariances', () => {
  it('processNoise = diag(1, 1, 1, 1, 1e-2, 1e-2, 1e-4) regardless of state', () => {
    const m = new CvBBoxMotionModel();
    const Q = m.processNoise(new Float64Array([0, 0, 0, 0, 0, 0, 0]));
    const expected = new Float64Array(49);
    const diag = [1, 1, 1, 1, 1e-2, 1e-2, 1e-4];
    for (let i = 0; i < 7; i++) expected[i * 7 + i] = diag[i]!;
    expectCloseArray(Q, expected);
  });

  it('processNoise does not depend on the state mean', () => {
    const m = new CvBBoxMotionModel();
    const a = m.processNoise(new Float64Array([0, 0, 0, 0, 0, 0, 0]));
    const b = m.processNoise(new Float64Array([100, 200, 300, 1.5, 1, 1, 1]));
    expectCloseArray(a, b);
  });

  it('measurementNoise = diag(1, 1, 10, 10) regardless of state', () => {
    const m = new CvBBoxMotionModel();
    const R = m.measurementNoise(new Float64Array([0, 0, 0, 0, 0, 0, 0]));
    const expected = new Float64Array(16);
    const diag = [1, 1, 10, 10];
    for (let i = 0; i < 4; i++) expected[i * 4 + i] = diag[i]!;
    expectCloseArray(R, expected);
  });
});

describe('CvBBoxMotionModel.init', () => {
  it('mean has measurement in slots 0..3 and zero velocities', () => {
    const m = new CvBBoxMotionModel();
    const s = m.init(new Float64Array([20, 40, 800, 0.5]));
    expectCloseArray(s.mean, [20, 40, 800, 0.5, 0, 0, 0]);
  });

  it('covariance is diag(10, 10, 10, 10, 1e4, 1e4, 1e4) per abewley/sort init', () => {
    const m = new CvBBoxMotionModel();
    const s = m.init(new Float64Array([20, 40, 800, 0.5]));
    const expected = new Float64Array(49);
    const diag = [10, 10, 10, 10, 1e4, 1e4, 1e4];
    for (let i = 0; i < 7; i++) expected[i * 7 + i] = diag[i]!;
    expectCloseArray(s.covariance, expected);
  });
});
