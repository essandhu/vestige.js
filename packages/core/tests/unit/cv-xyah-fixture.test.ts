import { describe, expect, it } from 'vitest';
// Fixture: scipy oracle for the DeepSORT-style cv-xyah KF.
// See packages/core/fixtures/kalman-update/README.md and ADR-0002.
import data from '../../fixtures/kalman-update/data.json' with { type: 'json' };
import { KalmanFilter, type KalmanState } from '../../src/filters/kalman.js';
import { CvXyahMotionModel } from '../../src/filters/motion-models/cv-xyah.js';

interface Op {
  kind: 'predict' | 'update';
  measurement?: number[];
}

interface Case {
  name: string;
  init_measurement: number[];
  ops: Op[];
  final_mean: number[];
  final_covariance: number[];
}

describe('CvXyahMotionModel + KalmanFilter — scipy oracle', () => {
  const kf = new KalmanFilter(new CvXyahMotionModel());

  for (const c of data.cases as Case[]) {
    it(c.name, () => {
      let state: KalmanState = kf.model.init(new Float64Array(c.init_measurement));
      for (const op of c.ops) {
        if (op.kind === 'predict') {
          state = kf.predict(state);
        } else {
          state = kf.update(state, new Float64Array(op.measurement!));
        }
      }

      expect(state.mean.length).toBe(c.final_mean.length);
      for (let i = 0; i < state.mean.length; i++) {
        expect(state.mean[i]).toBeCloseTo(c.final_mean[i]!, 9);
      }
      expect(state.covariance.length).toBe(c.final_covariance.length);
      for (let i = 0; i < state.covariance.length; i++) {
        expect(state.covariance[i]).toBeCloseTo(c.final_covariance[i]!, 9);
      }
    });
  }
});
