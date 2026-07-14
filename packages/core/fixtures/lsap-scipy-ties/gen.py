"""Generate scipy oracles for TIED linear-sum-assignment matrices.

Cross-implementation oracle for `packages/core/tests/validation/lsap-scipy-ties.test.ts`.

Why this fixture exists
-----------------------
`solveLsap` is a port of scipy's `_lsap.cpp` shortest-augmenting-path solver. On a
matrix with a unique optimum, any correct LSAP solver agrees. On a matrix with TIES,
the answer is a CHOICE, and the choice is decided by implementation detail -- in
scipy's case, by the order the `remaining` column array is initialized
(`remaining[it] = nc - it - 1`, i.e. REVERSE).

vestige originally filled that array forward, which inverted scipy's choice on every
tie. That is not cosmetic:

  * `solveLsap` backs the eval metrics' internal gt<->prediction matching, and
    TrackEval matches with `scipy.optimize.linear_sum_assignment`.
  * IDF1 in particular runs its assignment over INTEGER match counts, where exact
    ties are routine -- and the choice moves IDTP / IDFP / IDFN, hence the published
    number.

Ties are essentially absent from the tracker fixtures (their boxes are
well-separated, so every IoU is distinct), which is exactly why this went unnoticed.
A dedicated fixture is the only thing that pins it. See ADR-0006.

Note on the trackers
--------------------
The tracker references (sort.py, OC_SORT, ByteTrack) do NOT use scipy -- they call
`lap.lapjv`, which breaks ties DIFFERENTLY from scipy (verified: they disagree on
every tied matrix tried). So scipy parity is the correct target for the METRICS, and
lapjv parity is the correct target for the TRACKERS. That second gap is real but
currently unexercised; it is recorded in ADR-0006 rather than fixed here.

Per `docs/decisions/0002-fixtures-layout.md`, this script and its output JSON are
committed together; never one without the other.

Regenerate:

    python packages/core/fixtures/lsap-scipy-ties/gen.py
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

SEED = 20260713
N_RANDOM_CASES = 500

# Tie density is the whole point: drawing costs from a tiny value set guarantees that
# most matrices have many equally-optimal assignments, so the solver is forced to make
# the choice this fixture pins. Random floats would almost never tie.
TIE_VALUES = [0.0, 1.0, 2.0]

# Hand-picked cases where every solver must make a choice, kept first in the list so a
# failure report leads with something a human can read.
HANDPICKED: list[list[list[float]]] = [
    [[0.0, 0.0], [0.0, 0.0]],  # total tie, 2x2
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # total tie, 3x3
    [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [2.0, 2.0, 3.0]],  # tied 2x2 block
    [[0.0, 0.0, 5.0], [0.0, 0.0, 5.0], [5.0, 5.0, 0.0]],  # tied block + forced cell
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # rectangular, wide, total tie
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # rectangular, tall, total tie
    [[1.0]],  # 1x1
]


def solve(cost: np.ndarray) -> list[int]:
    """scipy's assignment as a rowToCol array (-1 = unmatched, for rectangular input)."""
    rows, cols = linear_sum_assignment(cost)
    row_to_col = [-1] * cost.shape[0]
    for i, j in zip(rows, cols):
        row_to_col[int(i)] = int(j)
    return row_to_col


def main() -> None:
    rng = np.random.default_rng(SEED)
    cases: list[dict] = []

    for rows in HANDPICKED:
        arr = np.array(rows, dtype=np.float64)
        cases.append(
            {
                "m": int(arr.shape[0]),
                "n": int(arr.shape[1]),
                "cost": [float(v) for v in arr.flatten()],
                "scipy_row_to_col": solve(arr),
            }
        )

    for _ in range(N_RANDOM_CASES):
        m = int(rng.integers(1, 6))
        n = int(rng.integers(1, 6))
        arr = rng.choice(TIE_VALUES, size=(m, n)).astype(np.float64)
        cases.append(
            {
                "m": m,
                "n": n,
                "cost": [float(v) for v in arr.flatten()],
                "scipy_row_to_col": solve(arr),
            }
        )

    envelope = {
        "$schema": "vestige.js fixture v1",
        "generator": {
            "script": "packages/core/fixtures/lsap-scipy-ties/gen.py",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "seed": SEED,
            "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "note": (
            "Oracles are scipy.optimize.linear_sum_assignment. Ties are resolved by "
            "scipy's reverse `remaining` fill; vestige's solveLsap must reproduce that "
            "choice exactly, because TrackEval matches with scipy. lap.lapjv resolves "
            "ties DIFFERENTLY and is the correct target for the trackers -- see ADR-0006."
        ),
        "cases": cases,
    }

    out_path = Path(__file__).with_name("data.json")
    out_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8", newline="\n")
    tied = sum(1 for c in cases if len(set(c["cost"])) < len(c["cost"]))
    print(f"wrote {out_path} ({len(cases)} cases, {tied} with repeated costs, scipy={scipy.__version__})")


if __name__ == "__main__":
    main()
