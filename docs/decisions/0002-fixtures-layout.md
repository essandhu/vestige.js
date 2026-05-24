# 0002 — Python fixtures layout (sketch)

- **Date:** 2026-05-24
- **Status:** Draft — to be finalized as the first fixture-needing function (`cholesky`) is implemented
- **Scope:** Conventions for per-function Python oracle fixtures (CONTRIBUTING.md §4 / ARCHITECTURE.md §13.1). Does **not** cover sequence-level tracker validation (ARCHITECTURE.md §13.3), which is deferred until v0.1 trackers exist.

## Context

ADR 0001 §5.1 anticipated this:

> When tests for the Kalman filter and Hungarian land in a future scaffold pass, scipy-derived fixtures will be needed because hand-verifying a Kalman update for an 8-dimensional state is not realistic. That layer of tests should commit the fixture *and* the scipy script that produced it.

The next scaffold layer (`linalg.ts` — `matMul`, `cholesky`, `choleskySolve`) is the inflection point where hand-computed oracles stop scaling. Settling the directory layout, naming, dependency pinning, and re-generation workflow **before** the first fixture lands means every subsequent function (Kalman, Hungarian) just follows the pattern instead of relitigating it.

Two things are explicitly **not** in scope:

1. **Sequence-level validation** (ARCHITECTURE §13.3) — `validation/` directory with pre-computed Python tracker outputs on fixed MOT sequences. Premature; no tracker exists yet.
2. **Python in CI** — the fixture JSON is committed; CI reads it. Python only runs locally when the maintainer regenerates.

---

## 1. Directory layout

```
packages/core/
├── src/
├── tests/
└── fixtures/
    ├── README.md                          # "how to regenerate" instructions
    ├── requirements.txt                   # pinned numpy/scipy versions
    ├── cholesky/
    │   ├── gen.py                         # the script
    │   ├── data.json                      # committed output
    │   └── README.md                      # what cases this covers + why
    ├── kalman-update/
    │   ├── gen.py
    │   ├── data.json
    │   └── README.md
    └── hungarian/
        ├── gen.py
        ├── data.json
        └── README.md
```

**Why per-function subdirectories rather than `fixtures/scripts/` + `fixtures/data/`:**
- Editing cholesky fixtures means touching one directory, not navigating between two.
- Each function gets its own README explaining *what cases the fixture covers and why those cases* — prevents the next person from re-deriving the test coverage from scratch.
- Naturally scoped commits: a Cholesky fixture update is one directory of changes, not a scattering.

**Why under `packages/core/fixtures/` rather than top-level `fixtures/`:**
- Fixtures are per-package consumers — `eval/` will eventually have its own (MOTChallenge sequence loaders test data).
- Vitest test discovery is scoped to `tests/`, so `fixtures/` doesn't accidentally get picked up as test files.

**Why not under `packages/core/tests/fixtures/`:**
- `tests/` is "test source code"; `fixtures/` is "test inputs + the scripts that produced them." Different review burden, different change cadence.
- The Python scripts are not tests; conflating them blurs the line.

---

## 2. Python policy

**Python is not a project dependency.** It's an offline tool the maintainer uses to regenerate fixtures. Specifically:

- **Not in `package.json`** (would imply runtime/test-time need).
- **Not in CI** (would require setup-python action and slow runs for no gained safety; CI reads committed JSON).
- **Required to *regenerate* fixtures**, not to run tests.

### 2.1 Version pinning

`packages/core/fixtures/requirements.txt`:
```
numpy==2.1.3
scipy==1.14.1
```

- Pin exact versions (`==`, not `>=`). Floating-point reproducibility across scipy minor versions is not guaranteed.
- When scipy releases a version with materially different numerics, this is a deliberate fixture re-baseline (commit message documents the change).
- No hash-pinning (this isn't a security-sensitive dependency surface).

**Python interpreter:** documented minimum is `python >= 3.11`. Not enforced by a script — the maintainer's `requirements.txt` install will fail loudly if the wheel set isn't compatible.

### 2.2 No `pnpm fixtures:regen` script yet

YAGNI. With 1–3 fixtures, `python packages/core/fixtures/cholesky/gen.py` documented in the README is enough. When fixtures hit ~5+ and there's real value in regenerating them all in one command, add the wrapper. Adding it speculatively now picks a Python-discovery strategy without seeing the actual ergonomics problem.

---

## 3. JSON fixture format

Standard envelope shared across all fixtures:

```json
{
  "$schema": "vestige.js fixture v1",
  "generator": {
    "script": "packages/core/fixtures/cholesky/gen.py",
    "python": "3.11.9",
    "numpy": "2.1.3",
    "scipy": "1.14.1",
    "generated": "2026-05-24T15:30:00Z"
  },
  "cases": [
    {
      "name": "3x3 well-conditioned PD",
      "n": 3,
      "A": [4, 12, -16, 12, 37, -43, -16, -43, 98],
      "L_expected": [2, 0, 0, 6, 1, 0, -8, 5, 3]
    }
  ]
}
```

**Why this envelope:**
- `generator` block gives provenance — when a TS test fails after a scipy upgrade, the diff is immediately diagnosable.
- `cases` is a flat array. Per-case `name` is the test-display string; per-case fields are function-specific.
- Row-major flat arrays for matrices, matching the Float64Array convention from ADR 0001 §4.2. The fixture format mirrors the runtime format — no reshape logic in either the script or the test.

**Loading convention in TS tests:**
```ts
import data from '../../fixtures/cholesky/data.json' with { type: 'json' };
for (const c of data.cases) {
  it(c.name, () => {
    const L = cholesky(new Float64Array(c.A), c.n);
    expect(Array.from(L)).toEqual(c.L_expected);  // or toBeCloseTo loop
  });
}
```

Single import, single iteration. Hand-written assertions only when the fixture format can't express what the test checks.

---

## 4. Re-generation workflow

**When a maintainer needs to add/change fixtures:**

1. `cd packages/core/fixtures && pip install -r requirements.txt` (one-time, ideally in a venv)
2. Edit the relevant `gen.py`
3. `python cholesky/gen.py > cholesky/data.json` (or the script writes the file directly)
4. Inspect the diff. JSON changes should be small and explicable; large noisy diffs mean the script changed in a way that shifted floating-point order-of-operations and needs a comment in the commit message.
5. Commit both `gen.py` and `data.json` in the same commit. **Never** commit one without the other.

**When CI fails on a fixture-backed test:** the maintainer re-runs `gen.py` locally. If the JSON changes, that's a real divergence (probably a TS bug, occasionally a scipy upgrade). If it doesn't, the TS code drifted.

**Stale-script detection** (optional, deferred):
- Add a `scriptHash` field to the JSON envelope (sha256 of `gen.py`).
- A `pnpm fixtures:check-hashes` script (pure-JS, no Python needed) verifies hashes match.
- Don't gate CI on this initially; revisit if it becomes a real problem.

---

## 5. What this convention costs and buys

| Buys | Costs |
|---|---|
| Functions that are genuinely impossible to hand-verify (Cholesky on 8×8, Kalman update on 8-d state, Hungarian on rectangular cost matrices) can have rigorous oracles. | Maintainer needs a working Python install with pinned scipy to regenerate. |
| Provenance: every fixture says which scipy/numpy version produced it. | JSON files are committed binary-ish blobs; reviewers can't usefully line-diff a big numeric change. Compensated by the per-function README explaining intent. |
| Scripts are committed alongside output, so any reviewer can verify "did this script produce this JSON" by running it. | Adds a `fixtures/` directory that contributors might assume is required setup — README must clearly say "only needed if you're touching numerics." |
| Pattern scales unchanged from one CIoU constant to a 50-case Hungarian fixture. | First setup is heavier than the immediate need (one Cholesky fixture). Amortized as more functions land. |

---

## 6. Things deliberately deferred

| Item | When to revisit |
|---|---|
| `pnpm fixtures:regen` wrapper script | When there are ≥5 fixture directories and regenerating one-by-one is real friction. |
| `pnpm fixtures:check-hashes` script | If a stale-script bug actually bites once. |
| Python in CI to verify scripts still produce committed JSON | Only if maintainer drift becomes a recurring problem; the cost (CI complexity, slower runs) probably exceeds the value. |
| Backfilling a CIoU fixture | Cheap, but not urgent — current property tests cover the bound/symmetry properties, and a single REPL-derived numeric constant inline in the test gives most of the value of a full fixture for far less ceremony. Bundle into the linalg PR or skip. |
| Sequence-level `validation/` directory (ARCHITECTURE §13.3) | When v0.1 trackers exist and there's something to validate against. |

---

## 7. Open questions

1. **JSON vs. NDJSON for large fixtures.** A 1000-case Hungarian fixture might be 5 MB. JSON parses entirely into memory; NDJSON streams. Unlikely to matter for v0.x sizes — revisit if a single fixture exceeds 1 MB.
2. **Where to put the `requirements.txt`** — currently proposed at `packages/core/fixtures/requirements.txt`. If `eval/` later grows its own Python tooling (sequence regenerators, metric cross-checks), we may want a top-level `fixtures-requirements.txt` shared across packages. Punt until eval needs one.
3. **Numeric tolerance per fixture.** Some functions are exact (integer-arithmetic Hungarian costs); some are float-only (Cholesky). The envelope should probably include a `tolerance: 1e-12` field per case (or per fixture) so the TS test doesn't have to hardcode it. Add when the first fixture needs a non-default tolerance.
4. **Multiple oracle sources per function.** For some functions both scipy and torchvision exist (e.g. `box_iou`). Should fixtures cross-check across both, or pick one? Default: pick the one ARCHITECTURE §4.2 / §10.2 names as canonical; only cross-check if the canonical reference has known quirks worth guarding against.
