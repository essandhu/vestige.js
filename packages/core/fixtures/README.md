# Test fixtures

This directory holds Python-generated oracle data for the TS test suite, plus the
scripts that produced it. Layout and conventions live in `docs/decisions/0002-fixtures-layout.md`.

You only need to read this directory if you're **touching numerics** (Kalman
filter, Hungarian, Cholesky, etc.) or **regenerating** an existing fixture.
Day-to-day work — adding trackers, plumbing types, optimising hot paths — never
touches `fixtures/`. The committed JSON is read by Vitest; no Python is required
to run `pnpm test`.

## When you'd regenerate

- You changed the math in `gen.py` (e.g. added a new test case).
- A scipy / numpy upgrade in `requirements.txt` would shift floating-point output,
  and you want to re-baseline deliberately.

## How

```powershell
# one-time, in a venv:
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# regenerate one fixture:
py kalman-update/gen.py
```

Commit `gen.py` and `data.json` together — never one without the other (ADR-0002
§4).

Fixtures with non-trivial setup beyond `requirements.txt` (e.g. cloning an
external reference repo) document their setup in their per-fixture README.

## What lives here today

| Directory | Used by | Extra setup |
|---|---|---|
| `kalman-update/` | `tests/unit/cv-xyah-fixture.test.ts` — cv-xyah KF predict+update oracles | none |
| `ocsort-noahcao/` | `tests/validation/ocsort-noahcao-fixture.test.ts` — cross-implementation faithfulness vs. `noahcao/OC_SORT` | clone `noahcao/OC_SORT` at the pinned commit (see fixture README) |
