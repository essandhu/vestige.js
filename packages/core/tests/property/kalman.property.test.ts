import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { KalmanFilter, type KalmanState, type MotionModel } from '../../src/filters/kalman.js';
import { cholesky } from '../../src/geometry/linalg.js';

function f(...xs: number[]): Float64Array {
  return new Float64Array(xs);
}

/**
 * Same minimal 2-d CV model used by the unit tests, duplicated here to keep
 * the property and unit suites independent.
 */
class TestCv1d implements MotionModel {
  readonly stateDim = 2;
  readonly measDim = 1;
  readonly F = f(1, 1, 0, 1);
  readonly H = f(1, 0);

  constructor(
    private readonly _Q: Float64Array = f(1e-4, 0, 0, 1e-4),
    private readonly _R: Float64Array = f(1),
  ) {}

  processNoise(): Float64Array {
    return this._Q;
  }

  measurementNoise(): Float64Array {
    return this._R;
  }

  init(measurement: Float64Array): KalmanState {
    return { mean: f(measurement[0]!, 0), covariance: f(1, 0, 0, 1) };
  }
}

const finiteFloat = fc.double({
  min: -1e3,
  max: 1e3,
  noNaN: true,
  noDefaultInfinity: true,
});

const measurement = finiteFloat.map((x) => f(x));

const initialState = fc
  .tuple(finiteFloat, finiteFloat, finiteFloat)
  .map(
    ([m0, m1, v]) =>
      ({ mean: f(m0, m1), covariance: f(v * v + 1, 0, 0, v * v + 1) }) as KalmanState,
  );

describe('KalmanFilter properties', () => {
  it('predict is deterministic: same input → bit-identical output', () => {
    const kf = new KalmanFilter(new TestCv1d());
    fc.assert(
      fc.property(initialState, (s) => {
        const a = kf.predict(s);
        const b = kf.predict(s);
        for (let i = 0; i < a.mean.length; i++) expect(a.mean[i]).toBe(b.mean[i]);
        for (let i = 0; i < a.covariance.length; i++) {
          expect(a.covariance[i]).toBe(b.covariance[i]);
        }
      }),
    );
  });

  it('update is deterministic: same input → bit-identical output', () => {
    const kf = new KalmanFilter(new TestCv1d());
    fc.assert(
      fc.property(initialState, measurement, (s, z) => {
        const a = kf.update(s, z);
        const b = kf.update(s, z);
        for (let i = 0; i < a.mean.length; i++) expect(a.mean[i]).toBe(b.mean[i]);
        for (let i = 0; i < a.covariance.length; i++) {
          expect(a.covariance[i]).toBe(b.covariance[i]);
        }
      }),
    );
  });

  it('predict does not mutate the input state', () => {
    const kf = new KalmanFilter(new TestCv1d());
    fc.assert(
      fc.property(initialState, (s) => {
        const meanCopy = new Float64Array(s.mean);
        const covCopy = new Float64Array(s.covariance);
        kf.predict(s);
        for (let i = 0; i < meanCopy.length; i++) expect(s.mean[i]).toBe(meanCopy[i]);
        for (let i = 0; i < covCopy.length; i++) expect(s.covariance[i]).toBe(covCopy[i]);
      }),
    );
  });

  it('update does not mutate the input state or measurement', () => {
    const kf = new KalmanFilter(new TestCv1d());
    fc.assert(
      fc.property(initialState, measurement, (s, z) => {
        const meanCopy = new Float64Array(s.mean);
        const covCopy = new Float64Array(s.covariance);
        const zCopy = new Float64Array(z);
        kf.update(s, z);
        for (let i = 0; i < meanCopy.length; i++) expect(s.mean[i]).toBe(meanCopy[i]);
        for (let i = 0; i < covCopy.length; i++) expect(s.covariance[i]).toBe(covCopy[i]);
        for (let i = 0; i < zCopy.length; i++) expect(z[i]).toBe(zCopy[i]);
      }),
    );
  });

  it('post-update covariance is symmetric to within 1e-9', () => {
    const kf = new KalmanFilter(new TestCv1d());
    fc.assert(
      fc.property(initialState, measurement, (s, z) => {
        const post = kf.update(s, z);
        const n = kf.model.stateDim;
        for (let i = 0; i < n; i++) {
          for (let j = i + 1; j < n; j++) {
            const a = post.covariance[i * n + j]!;
            const b = post.covariance[j * n + i]!;
            expect(Math.abs(a - b)).toBeLessThan(1e-9);
          }
        }
      }),
    );
  });

  it('post-update covariance is positive-definite (cholesky succeeds)', () => {
    const kf = new KalmanFilter(new TestCv1d());
    fc.assert(
      fc.property(initialState, measurement, (s, z) => {
        const post = kf.update(s, z);
        // Cholesky throws on non-PD matrices.
        expect(() => cholesky(post.covariance, kf.model.stateDim)).not.toThrow();
      }),
    );
  });

  it('post-update covariance is finite (no NaN/Infinity)', () => {
    const kf = new KalmanFilter(new TestCv1d());
    fc.assert(
      fc.property(initialState, measurement, (s, z) => {
        const post = kf.update(s, z);
        for (let i = 0; i < post.mean.length; i++) expect(Number.isFinite(post.mean[i])).toBe(true);
        for (let i = 0; i < post.covariance.length; i++) {
          expect(Number.isFinite(post.covariance[i])).toBe(true);
        }
      }),
    );
  });
});
