import { describe, expect, it } from 'vitest';
import { KalmanFilter, type KalmanState, type MotionModel } from '../../src/filters/kalman.js';

function f(...xs: number[]): Float64Array {
  return new Float64Array(xs);
}

function expectCloseArray(actual: Float64Array, expected: ArrayLike<number>, eps = 1e-10) {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i] as number, -Math.log10(eps));
  }
}

/**
 * Minimal 2-d constant-velocity test model: state = [x, ẋ], measurement = [x].
 * Lets us hand-verify predict / update without dragging in the SORT or DeepSORT
 * constants. Q and R are constructor-configurable so individual tests can pick
 * degenerate values (e.g. Q = 0, R = 0) to lock down the math.
 */
class TestCv1d implements MotionModel {
  readonly stateDim = 2;
  readonly measDim = 1;
  readonly F = f(1, 1, 0, 1);
  readonly H = f(1, 0);

  constructor(
    private readonly _Q: Float64Array = f(0, 0, 0, 0),
    private readonly _R: Float64Array = f(1),
    private readonly _initVar: number = 1,
  ) {}

  processNoise(): Float64Array {
    return this._Q;
  }

  measurementNoise(): Float64Array {
    return this._R;
  }

  init(measurement: Float64Array): KalmanState {
    return {
      mean: f(measurement[0]!, 0),
      covariance: f(this._initVar, 0, 0, this._initVar),
    };
  }
}

describe('KalmanFilter.predict', () => {
  it('on F=I, Q=0 the state is unchanged', () => {
    class Id implements MotionModel {
      readonly stateDim = 2;
      readonly measDim = 1;
      readonly F = f(1, 0, 0, 1);
      readonly H = f(1, 0);
      processNoise(): Float64Array {
        return f(0, 0, 0, 0);
      }
      measurementNoise(): Float64Array {
        return f(1);
      }
      init(): KalmanState {
        return { mean: f(0, 0), covariance: f(1, 0, 0, 1) };
      }
    }
    const kf = new KalmanFilter(new Id());
    const out = kf.predict({ mean: f(3, 4), covariance: f(2, 1, 1, 5) });
    expectCloseArray(out.mean, [3, 4]);
    expectCloseArray(out.covariance, [2, 1, 1, 5]);
  });

  it('propagates mean through F: x = [x, ẋ] = [0, 1] → [1, 1]', () => {
    const kf = new KalmanFilter(new TestCv1d());
    const out = kf.predict({ mean: f(0, 1), covariance: f(1, 0, 0, 1) });
    expectCloseArray(out.mean, [1, 1]);
  });

  it('propagates covariance: P → F P Fᵀ + Q', () => {
    // F = [[1,1],[0,1]], P = I, Q = diag(0.1, 0.2)
    // F P Fᵀ = [[2, 1], [1, 1]]; + Q = [[2.1, 1], [1, 1.2]]
    const kf = new KalmanFilter(new TestCv1d(f(0.1, 0, 0, 0.2)));
    const out = kf.predict({ mean: f(0, 0), covariance: f(1, 0, 0, 1) });
    expectCloseArray(out.covariance, [2.1, 1, 1, 1.2]);
  });

  it('does not mutate the input state', () => {
    const kf = new KalmanFilter(new TestCv1d(f(0.1, 0, 0, 0.2)));
    const mean = f(1, 2);
    const covariance = f(1, 0, 0, 1);
    kf.predict({ mean, covariance });
    expectCloseArray(mean, [1, 2]);
    expectCloseArray(covariance, [1, 0, 0, 1]);
  });

  it('returns a fresh allocation (not the same buffer as input)', () => {
    const kf = new KalmanFilter(new TestCv1d());
    const state = { mean: f(0, 1), covariance: f(1, 0, 0, 1) };
    const out = kf.predict(state);
    expect(out.mean).not.toBe(state.mean);
    expect(out.covariance).not.toBe(state.covariance);
  });
});

describe('KalmanFilter.project', () => {
  it('returns H x and H P Hᵀ + R', () => {
    // H = [1, 0], P = [[2, 1], [1, 3]], R = [4]
    // H x = [3]; H P Hᵀ = [[2]]; + R = [[6]]
    const kf = new KalmanFilter(new TestCv1d(f(0, 0, 0, 0), f(4)));
    const proj = kf.project({ mean: f(3, 7), covariance: f(2, 1, 1, 3) });
    expectCloseArray(proj.mean, [3]);
    expectCloseArray(proj.covariance, [6]);
  });
});

describe('KalmanFilter.update', () => {
  it('with R=0 the posterior mean exactly equals the measurement (in obs space)', () => {
    // 1-d KF: state = [x, ẋ], H = [1, 0], R = 0.
    // Update with z = 10 forces posterior mean[0] = 10 within float tolerance.
    const kf = new KalmanFilter(new TestCv1d(f(0, 0, 0, 0), f(0)));
    const out = kf.update({ mean: f(0, 0), covariance: f(1, 0, 0, 1) }, f(10));
    expect(out.mean[0]).toBeCloseTo(10, 12);
  });

  it('with R=0 the observed-dim posterior covariance is 0', () => {
    const kf = new KalmanFilter(new TestCv1d(f(0, 0, 0, 0), f(0)));
    const out = kf.update({ mean: f(0, 0), covariance: f(1, 0, 0, 1) }, f(10));
    // P[0,0] is the observed dimension; should collapse to 0.
    expect(out.covariance[0]).toBeCloseTo(0, 12);
  });

  it('with measurement equal to predicted obs (and R>0) shifts the mean toward z proportionally', () => {
    // x = [0, 0], P = I, R = [1]; predicted obs = 0; z = 4.
    // Kalman gain K = P Hᵀ / (H P Hᵀ + R) = [[1],[0]] / 2 = [[0.5],[0]].
    // posterior mean = x + K (z - H x) = [0, 0] + [0.5, 0] * 4 = [2, 0].
    const kf = new KalmanFilter(new TestCv1d(f(0, 0, 0, 0), f(1)));
    const out = kf.update({ mean: f(0, 0), covariance: f(1, 0, 0, 1) }, f(4));
    expectCloseArray(out.mean, [2, 0]);
  });

  it('shrinks observed-dim variance: P_post[0,0] < P_prior[0,0]', () => {
    const kf = new KalmanFilter(new TestCv1d(f(0, 0, 0, 0), f(1)));
    const prior = { mean: f(0, 0), covariance: f(1, 0, 0, 1) };
    const post = kf.update(prior, f(4));
    expect(post.covariance[0]!).toBeLessThan(prior.covariance[0]!);
  });

  it('does not mutate input state or measurement', () => {
    const kf = new KalmanFilter(new TestCv1d(f(0, 0, 0, 0), f(1)));
    const mean = f(0, 0);
    const covariance = f(1, 0, 0, 1);
    const measurement = f(4);
    kf.update({ mean, covariance }, measurement);
    expectCloseArray(mean, [0, 0]);
    expectCloseArray(covariance, [1, 0, 0, 1]);
    expectCloseArray(measurement, [4]);
  });

  it('returns a fresh allocation (not the same buffer as input)', () => {
    const kf = new KalmanFilter(new TestCv1d(f(0, 0, 0, 0), f(1)));
    const state = { mean: f(0, 0), covariance: f(1, 0, 0, 1) };
    const out = kf.update(state, f(4));
    expect(out.mean).not.toBe(state.mean);
    expect(out.covariance).not.toBe(state.covariance);
  });
});

describe('KalmanFilter.predict + update on a steady-state 1-d sequence', () => {
  it('converges to the true position under noiseless measurements', () => {
    // Constant-velocity ground truth: x(t) = 5 + 2t. Feed noiseless measurements
    // for 20 frames starting from a wildly-wrong init. With R small but nonzero,
    // the posterior mean should track the truth to within < 0.1 by frame 20.
    const kf = new KalmanFilter(new TestCv1d(f(1e-4, 0, 0, 1e-4), f(0.01), 1000));
    let state = kf.model.init(f(5));
    for (let t = 1; t <= 20; t++) {
      state = kf.predict(state);
      state = kf.update(state, f(5 + 2 * t));
    }
    expect(state.mean[0]).toBeCloseTo(5 + 2 * 20, 1);
    expect(state.mean[1]).toBeCloseTo(2, 1);
  });
});
