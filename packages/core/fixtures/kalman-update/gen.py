"""Generate cv-xyah Kalman filter oracle data for the TS test suite.

This is a direct numpy reimplementation of
`packages/core/src/filters/motion-models/cv-xyah.ts` plus
`packages/core/src/filters/kalman.ts`, used as a reference oracle. The TS
implementation runs the same sequences and asserts equality with the output of
this script (loaded from `data.json`).

Updates use scipy's `cho_factor` / `cho_solve` to match what DeepSORT does
(`nwojke/deep_sort/kalman_filter.py`).

Per `docs/decisions/0002-fixtures-layout.md`, this script and its output JSON
are committed together; never one without the other.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg

WP = 1.0 / 20.0
WV = 1.0 / 160.0


def _F() -> np.ndarray:
    F = np.eye(8)
    for i in range(4):
        F[i, i + 4] = 1.0
    return F


def _H() -> np.ndarray:
    return np.eye(4, 8)


def init(measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = float(measurement[3])
    mean = np.zeros(8)
    mean[:4] = measurement
    std = np.array(
        [
            2 * WP * h,
            2 * WP * h,
            1e-2,
            2 * WP * h,
            10 * WV * h,
            10 * WV * h,
            1e-5,
            10 * WV * h,
        ]
    )
    covariance = np.diag(std**2)
    return mean, covariance


def process_noise(mean: np.ndarray) -> np.ndarray:
    h = float(mean[3])
    sp = WP * h
    sv = WV * h
    diag = np.array([sp**2, sp**2, 1e-4, sp**2, sv**2, sv**2, 1e-10, sv**2])
    return np.diag(diag)


def measurement_noise(mean: np.ndarray) -> np.ndarray:
    h = float(mean[3])
    sp = WP * h
    return np.diag(np.array([sp**2, sp**2, 1e-2, sp**2]))


def predict(mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    F = _F()
    Q = process_noise(mean)
    new_mean = F @ mean
    new_cov = F @ covariance @ F.T + Q
    return new_mean, new_cov


def project(mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H = _H()
    R = measurement_noise(mean)
    proj_mean = H @ mean
    proj_cov = H @ covariance @ H.T + R
    return proj_mean, proj_cov


def update(
    mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    H = _H()
    proj_mean, proj_cov = project(mean, covariance)
    chol, lower = scipy.linalg.cho_factor(proj_cov, lower=True, check_finite=False)
    # K = P Hᵀ S⁻¹. Solve S Kᵀ = H P.
    HP = H @ covariance
    K_T = scipy.linalg.cho_solve((chol, lower), HP, check_finite=False)
    K = K_T.T
    innovation = measurement - proj_mean
    new_mean = mean + K @ innovation
    new_cov = covariance - K @ proj_cov @ K.T
    return new_mean, new_cov


def run_case(init_measurement: list[float], ops: list[dict]) -> dict:
    mean, cov = init(np.array(init_measurement))
    for op in ops:
        if op["kind"] == "predict":
            mean, cov = predict(mean, cov)
        elif op["kind"] == "update":
            mean, cov = update(mean, cov, np.array(op["measurement"]))
        else:
            raise ValueError(f"unknown op kind: {op['kind']}")
    return {
        "init_measurement": init_measurement,
        "ops": ops,
        "final_mean": mean.tolist(),
        "final_covariance": cov.flatten().tolist(),
    }


def main() -> None:
    cases = [
        {
            "name": "init-predict5",
            **run_case([100.0, 200.0, 0.5, 100.0], [{"kind": "predict"}] * 5),
        },
        {
            "name": "init-update",
            **run_case(
                [100.0, 200.0, 0.5, 100.0],
                [{"kind": "update", "measurement": [105.0, 198.0, 0.52, 104.0]}],
            ),
        },
        {
            "name": "init-predict-update-x10",
            **run_case(
                [100.0, 200.0, 0.5, 100.0],
                [
                    op
                    for t in range(10)
                    for op in [
                        {"kind": "predict"},
                        {
                            "kind": "update",
                            "measurement": [
                                100.0 + 3.0 * (t + 1),
                                200.0 - 1.5 * (t + 1),
                                0.5 + 0.005 * (t + 1),
                                100.0 + 0.5 * (t + 1),
                            ],
                        },
                    ]
                ],
            ),
        },
    ]

    envelope = {
        "$schema": "vestige.js fixture v1",
        "generator": {
            "script": "packages/core/fixtures/kalman-update/gen.py",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "cases": cases,
    }

    out_path = Path(__file__).with_name("data.json")
    # Force LF line endings so biome's formatter (lineEnding: "lf") doesn't trip
    # the file on Windows hosts where the default newline is CRLF.
    out_path.write_text(
        json.dumps(envelope, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {out_path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
