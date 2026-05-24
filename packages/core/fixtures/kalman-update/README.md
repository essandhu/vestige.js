# `kalman-update/` — cv-xyah Kalman filter oracles

Generates predict-and-update sequences for the DeepSORT-style 8-d motion model
(`packages/core/src/filters/motion-models/cv-xyah.ts`), using a direct numpy
reimplementation of the model. The TS test (`tests/unit/cv-xyah-fixture.test.ts`)
runs the same sequences through `KalmanFilter` + `CvXyahMotionModel` and asserts
bit-near-equality with the JSON.

## What this covers

The 8-d KF update collapses an `(8, 8)` covariance and an `(8,)` mean into a
shape no human can hand-verify. The state-dependent `Q(h)` and `R(h)` add
another layer. These cases pin down:

- **Predict propagates F and Q correctly** at non-trivial `h` (sequence A).
- **The Cholesky-based update path** produces the same posterior as
  `scipy.linalg.cho_solve` on the innovation covariance (sequence B).
- **Repeated predict+update** doesn't drift away from the reference over
  multiple frames (sequence C) — the property test asserts symmetry and PD,
  this asserts numerical equality.

## Cases

- **`init-predict5`** — init at `[100, 200, 0.5, 100]`, predict 5 times, no
  updates. Pins down the predict-only path.
- **`init-update`** — init at `[100, 200, 0.5, 100]`, single update with
  measurement `[105, 198, 0.52, 104]`. Pins down a single Cholesky-based update.
- **`init-predict-update-x10`** — init at `[100, 200, 0.5, 100]`, then ten
  frames of predict + update with a moving target. Pins down the steady-state
  filter loop.

## Regenerating

```powershell
py gen.py
```

The script writes `data.json` in place. Inspect the diff; if the only changes
are timestamps in the `generator` block, the math is unchanged.
