# Contributing to vestige.js

This file captures the project-specific conventions that don't live in code. ARCHITECTURE.md
is the design spec; this file is the day-to-day "how we work" reference.

---

## 1. Workflow

- **TDD scaffold-first.** Tests + stubs (`throw new Error('not implemented')`) land before
  the implementation in commit history. The implementation then turns red tests green, one
  function at a time, in foundational → derived order. See `docs/ARCHITECTURE.md` §13 for
  the test pyramid.
- **Scaffold and implementation are separate commits within the same feature branch.** The
  scaffold commit contains type contracts, signatures, JSDoc, and failing tests — never
  working implementations. Implementation lands in subsequent commits on the same branch so
  the PR's history shows the red → green arc.
- **Bundled PRs — `main` stays green.** Scaffold and implementation ship together in one
  PR per feature, never as separate PRs to `main`. The squash-merge keeps `main`'s history
  as one logical step per feature, and CI is meaningful because every commit on `main` is
  green. Concretely:
  1. `git checkout -b feature/<name>` from `main`.
  2. Scaffold tests + stubs — commit (CI red on the branch).
  3. Implement on the same branch — commit(s) (CI green on the branch tip).
  4. Open the PR when the branch is green; squash-merge after review.

  Granularity is per-feature: one PR for all of `bbox.ts`, one for all IoU variants, one
  for `cholesky` + `choleskySolve`, one tracker per PR, etc. — bundle related functions
  when implementing them in isolation wouldn't be coherent.

  **Do not push the scaffold directly to `main`.** That was the v0.0 mistake; it left
  `main` red until the impl caught up.
- **One package at a time.** `packages/core/` (zero runtime deps, publishable) is the
  primary target. `packages/eval/` (Node-only) only matters at benchmark time.

## 2. Code style — enforced by Biome

The full ruleset is in `biome.json`. The non-obvious pieces:

- **Single quotes**, **semicolons always**, **trailing commas everywhere**, **2-space indent**,
  **100-char line width**, **LF line endings**.
- **`import type` is required** for type-only imports — `verbatimModuleSyntax` is on in
  `tsconfig.base.json`, and Biome's `useImportType` enforces it. Same for `export type`.
- **Imports are auto-organized on `biome check --write`.** Don't hand-sort.

Run before committing:

```powershell
pnpm check       # lint + format check, read-only
pnpm check:fix   # lint + format + organize imports, write
```

## 3. Code style — conventions Biome can't catch

These are the patterns that make a vestige.js function look like it belongs. Internalize them.

### 3.1 Pure functions, no mutation of inputs

- Geometry, IoU, linalg, and cost-matrix functions are **pure**: same inputs → same outputs,
  no side effects, no `this`.
- **`BBox` is `readonly`.** Never mutate a caller's bbox. If you need a mutable buffer,
  allocate a new one.
- **`Float64Array` inputs are read-only by convention.** The only exception is `out`-style
  parameters explicitly documented to be written to (see `matMul`'s `out` arg).
- **`addInPlace` / `subInPlace` are the exception, not the rule.** They mutate the first
  argument by design and the name says so. Anywhere else, returning a fresh allocation is
  the default.

### 3.2 Return shapes

- **Return concrete tuple types** — `[number, number, number, number]` — not `number[]`.
  This is what makes `BBox`-typed return values flow through callers without casts.
- **One return shape per function.** No "returns `T | null` sometimes and throws other
  times." If a degenerate input has a defined return (e.g. `bboxArea` of a zero-area
  box returns 0), document it in JSDoc and honor it. If it doesn't, throw with a
  specific, matchable message.
- **Throw messages must be specific.** Tests use `.toThrow(/positive[\s-]?definite/i)`
  to distinguish a real failure from the `Error('not implemented')` stub. Generic
  `Error('bad input')` defeats this.

### 3.3 Degenerate inputs

Each function's JSDoc says how it handles degenerate inputs (zero area, `h == 0`, non-PD
matrices, etc.). The contract:

- **Geometry**: degenerate boxes return 0 area / 0 IoU. They do not throw. Callers may
  filter upstream.
- **Linalg**: non-positive-definite matrices throw from `cholesky`. The error message
  contains "positive definite" (case- and hyphen-insensitive) — tests match on this.
- **Out-of-range floats** (`NaN`, `Infinity` in bbox coordinates): undefined behavior;
  callers must filter. We do not pay the per-call cost of guarding.

### 3.4 Performance discipline (hot paths)

Architecture §9.3 codifies this; the day-to-day rules:

- **No closures inside per-frame loops.** If you need a helper, hoist it to module scope.
- **Pre-allocate scratch buffers** sized to the largest observed scene; grow lazily,
  never shrink.
- **Prefer `Float64Array` over `number[]`** when JSDoc specifies it. The cost-matrix
  and Kalman covariance paths assume flat row-major typed arrays.
- **Unroll fixed-size loops** in linalg primitives (8×8, 4×8, 4×4) once correctness is
  proven. Architecture target: <200 LOC total for linalg, fully unrolled.

### 3.5 Comments

- Default to **no inline comments**. Named identifiers and JSDoc on exported symbols
  carry the load.
- **Only add a comment when the why is non-obvious**: a numerical-stability concern, a
  paper-specific convention, a workaround for a subtle bug. "Computes the sum" on a
  function called `sum` is noise.
- **Don't reference tasks, PRs, or recent changes in comments.** That context belongs
  in the commit message and PR description; comments outlive the work that produced them.

---

## 4. Reference checking — how to know an implementation is correct

The geometry, linalg, and tracker code lives or dies by matching published math. Use this
checklist when implementing or reviewing any function in the numerical core.

### 4.1 The tests are the contract

If `pnpm test:watch` is green for a function, it satisfies:

- The unit-test cases (hand-built or scipy/numpy oracle values).
- The property-based invariants (e.g., `iou(a, a) === 1` for any positive-area box).
- The throw-message regex for documented error paths.

Tests passing is necessary but not always sufficient — see 4.4 below.

### 4.2 Cross-check against a Python reference for non-obvious math

For anything where a published paper or library defines the canonical formula:

| You're implementing… | Reference to cross-check against |
|---|---|
| `iou`, `iouMatrix` | `torchvision.ops.box_iou`, or scipy |
| `giou` | Rezatofighi et al., CVPR 2019, eq. 1–3 |
| `diou`, `ciou` | Zheng et al., AAAI 2020, eqs. 6–11 |
| `xyxyToXyah` / inverse | DeepSORT reference impl (`nwojke/deep_sort`) |
| `cholesky` / `choleskySolve` | scipy `cho_factor` / `cho_solve` |
| Kalman predict/update | DeepSORT or filterpy reference impl |
| Hungarian | scipy `linear_sum_assignment` |
| HOTA / MOTA / IDF1 | `JonathonLuiten/TrackEval` |

The architecture allows up to ~1e-12 numerical divergence from these references (§10.1).
Anything beyond that is a bug — usually order-of-operations or a missing/extra term.

### 4.3 Sanity checks that catch dumb mistakes

Before declaring a function done:

- **Identity inputs**: `iou(b, b) === 1` for any positive-area `b`. `xyahToXyxy(xyxyToXyah(b))`
  ≈ `b` within float tolerance. `cholesky` on the identity matrix returns the identity.
- **Symmetry**: `iou(a, b) === iou(b, a)`. `matMul(A, B)` differs from `matMul(B, A)`
  unless they commute — don't accidentally swap.
- **Boundary**: degenerate boxes (zero area, identical points) return the documented
  degenerate value, not `NaN`.
- **Determinism**: same inputs → exactly equal outputs (bit-for-bit on the same machine).
  No `Map`/`Set` iteration order in the output path.

### 4.4 When tests pass but you're still unsure

Tests can pass for the wrong reason. Smell-check:

- **Did the test actually exercise the function?** Search the test file for the function
  name. If it's only referenced via a higher-level function, you're testing the wrapper.
- **Is the oracle value hand-computed or fixture-loaded?** Hand-computed = high confidence.
  Fixture-loaded = check that the fixture-generation script is committed (per architecture
  §13.3) and that you understand what it did.
- **Run a known-bad input.** Flip a sign, swap two arguments, return a constant. If the
  test still passes, the test is weak — flag it, don't ship around it.

### 4.5 Reference impls to keep open while working

- abewley/sort — original SORT (Python). ~400 LOC, mandatory reading for any tracker work.
- nwojke/deep_sort — DeepSORT (Python). Reference for the cv-xyah motion model and the
  scipy-style Kalman update used by ByteTrack and OC-SORT.
- FoundationVision/ByteTrack — official ByteTrack.
- noahcao/OC_SORT — official OC-SORT. The ORU/OCM/OCR math is subtle; cross-read with
  the paper.
- NirAharon/BoT-SORT — official BoT-SORT.

The library does **not** vendor these. Pin commit hashes in benchmark reports only.

---

## 5. Tooling cheatsheet

| Task | Command |
|---|---|
| Watch tests on the core package | `pnpm test:watch` |
| Run all tests once | `pnpm test` |
| Typecheck (no emit) | `pnpm typecheck` |
| Lint + format check (read-only) | `pnpm check` |
| Lint + format + organize imports (write) | `pnpm check:fix` |
| Build dual ESM/CJS bundles | `pnpm build` |

`pnpm` is required (workspace setup uses `pnpm-workspace.yaml`). Node 18+.
