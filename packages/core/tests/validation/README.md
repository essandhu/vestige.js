# tests/validation — end-to-end tracker behavior on engineered scenarios

This directory holds the **validation tier** of the test pyramid
(ARCHITECTURE.md §13.1, line 4 — "integration tests; full tracker runs on
synthetic sequences ... with hand-verified expected outputs"). Tests here
drive an entire `Tracker` implementation through a multi-frame scenario and
assert on observable invariants — track ID preservation, recovery from
occlusion, directional consistency — that none of the unit-tier or
property-tier tests can express.

This is **distinct from**:

- `tests/unit/` — pure-function or single-frame tracker behavior.
- `tests/property/` — `fast-check`-driven invariant probing.
- `packages/core/fixtures/` — Python-generated numerical oracles for the
  numerical core (linalg / Kalman). See ADR-0002.
- `packages/eval/` (future) — MOTChallenge-style HOTA/MOTA/IDF1 benchmarking
  against the published reference implementations (ARCHITECTURE.md §10).

## What lives here

| File | Tracker | Scenario class |
|---|---|---|
| `ocsort-synthetic.test.ts` | `OcSortTracker` | ORU (occlusion → re-association drift correction), OCM (directional cost), OCR (last-observation rescue) — see paper §3.2–3.4. |

## When to add a validation test

A new validation test is warranted when:

1. A tracker contribution (ORU, OCM, OCR, CMC, etc.) needs an observable
   end-to-end check that no unit test naturally provides.
2. The scenario can be hand-designed so the expected outcome is unambiguous
   without consulting a reference implementation. (Cross-implementation
   reference checks live in `packages/eval/`, not here.)
3. A regression in the tracker's pipeline would silently pass the unit
   suite but break the scenario.

If two of three apply, add the test.

## Conventions

- **Engineered scenarios, not random.** Each test constructs a specific
  motion + occlusion pattern and asserts on what the tracker *must* do.
  Property-based / randomized exploration belongs in `tests/property/`.
- **Hand-trace the expected output in a comment** above the scenario.
  Anyone reviewing or debugging a failure should not need to re-derive what
  the tracker is supposed to do.
- **Cite the paper section / reference-impl line numbers** that motivate
  the scenario. The whole point of the validation tier is that each test
  ties back to a published behavior we are claiming to implement.
- **One assertion per behavior.** If a single scenario tests ORU
  *and* OCR, split it into two tests so a failure tells you which mechanism
  broke.

## Why no Python fixture (yet)

ADR-0002 §6 defers sequence-level Python oracles to "when v0.1 trackers
exist and there's something to validate against." OC-SORT is the first
tracker where ORU's effect on the Kalman state is too subtle to
hand-verify in a single test — but a hand-designed scenario with a
specific ID-preservation outcome is sufficient to catch correctness
regressions at this stage. The full noahcao/OC_SORT cross-check fixture
(per the workflow in `packages/core/fixtures/README.md`) lands when the
maintainer next runs the reference locally; the harness here is shaped
to absorb that fixture as a parallel `data.json`-backed test without
restructuring.
