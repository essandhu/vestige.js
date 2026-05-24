# 0001 — Initial scaffold

- **Date:** 2026-05-23
- **Status:** Accepted
- **Scope:** Initial repository scaffolding only. All tracker / algorithm decisions deferred to future ADRs as they are made.

## Context

ARCHITECTURE.md describes a zero-dependency, TypeScript-first multi-object
tracking library to be built across roughly 12 months in four milestones
(v0.1 → v1.0). The repository starts effectively empty, with only
`README.md` and `ARCHITECTURE.md`.

The chosen foundation is **TDD-first**: scaffold + types + failing tests for
the first layer of primitives (bbox geometry, IoU, linalg). Implementations
flip the red tests green one function at a time on per-feature branches.
Implementing the algorithm code is intentionally not in scope for this
scaffold ADR; that work lands in per-feature PRs.

This document captures the decisions made during the scaffold so that future
contributors understand why the project is shaped the way it is, separate
from the *what* of the architecture.

---

## 1. Toolchain

### 1.1 pnpm workspaces (chosen)

**Decision:** Use pnpm workspaces as the monorepo tool.

**Alternatives considered:**

- **npm workspaces** — zero install friction, but slower and lacks pnpm's
  strict module resolution. Strict resolution matters here because the core
  package must ship with `dependencies: {}`; pnpm catches phantom dependency
  bugs at install time that npm hoists past.
- **yarn workspaces** — mature; requires an extra install step; offers no
  decisive advantage over pnpm for this project size.

**Tradeoff accepted:** Contributors who do not already have pnpm installed pay
a one-time install cost (`npm install -g pnpm`). The contributor docs (when
written) should call this out.

### 1.2 tsup for builds (chosen)

**Decision:** Use tsup for building `packages/core`.

**Alternatives considered:**

- **tshy** — zero-config dual ESM/CJS from package.json. Newer, smaller
  community, more opinionated about layout, slower because it uses `tsc`
  directly rather than esbuild.
- **Rolling our own rollup config** — maximum control, far more maintenance.
- **Just `tsc`** — no bundling, no dual-format support out of the box.

**Why tsup wins here:** widest community familiarity in the TS-library
ecosystem (matters for the architecture's contributor model in §14),
esbuild-backed so dual ESM/CJS + sourcemaps + types build is fast,
the architecture doc itself lists it first in §12.1.

**Tradeoff accepted:** tsup has its own small config DSL (tsup.config.ts).
That config has to be maintained when entry points change. The current
config exposes two entries (`index` and `geometry/index`) matching the
package's exports map; new subpaths require updates in both places.

### 1.3 Vitest for tests (chosen)

**Decision:** Use Vitest as the test runner.

**Why:** Native ESM, native TS, fast, mature, built-in property-test
compatible inline snapshots, watch mode that doesn't require a separate
build step. Architecture doc §13.2 specifies it.

**Tradeoff accepted:** Vitest 2.x requires Node 18+, so the *dev* engines
field on the workspace root is `>=18` even though the *published* core
package targets Node 14+. This is fine — engines is informational; the
emitted ES2020 artifact runs on Node 14+ regardless of the build host's
Node version.

### 1.4 fast-check for property tests (chosen)

**Decision:** Use fast-check for property-based tests.

**Why:** Architecture doc §13.2 specifies it. The de facto standard in the
TS world; well-maintained; tiny.

**Where applied initially:** roundtrip identity for bbox conversions; IoU
symmetry and bounds; IoU matrix cell-by-cell consistency with scalar IoU.

### 1.5 No CI, no size-limit, no LICENSE, no README expansion (deliberately deferred)

- **CI (GitHub Actions, etc.):** premature. Set up once the first real
  implementations land and the test loop is known to be green for someone
  else's environment.
- **size-limit:** architecture doc §12.3 names this for v1.0 bundle-size
  budgets. v0.0.0 has no bundle yet.
- **LICENSE:** the architecture doc references MIT for the reference Python
  implementations but does not commit the library's license. Deferred until
  before first publish.
- **README:** the existing one-line README was left untouched.

---

## 2. TypeScript configuration

The base config lives in `tsconfig.base.json` and is extended by both
packages.

### 2.1 Enabled — strictness flags

| Flag | Why |
|---|---|
| `strict: true` | Standard for any new TS project. |
| `noUncheckedIndexedAccess: true` | `arr[i]` is `T \| undefined`. Critical for code that touches typed-array buffers and cost matrices — silent out-of-bounds reads are exactly the bug class to prevent here. |
| `noImplicitReturns: true` | Avoids accidental `undefined` returns from branches in cost-matrix gating logic. |
| `noFallthroughCasesInSwitch: true` | Cheap insurance for lifecycle-state switches. |
| `isolatedModules: true` | Required for single-file transpilers like esbuild (which tsup uses). Surface concept: every file must be compilable independently. |
| `verbatimModuleSyntax: true` | Forces `import type` for type-only imports. Eliminates a class of dual-package hazards where types accidentally compile into runtime imports. |
| `forceConsistentCasingInFileNames: true` | Windows / macOS / Linux path-casing consistency. Cheap insurance for a cross-platform contributor base. |
| `declaration` + `declarationMap` + `sourceMap` | Published `.d.ts` files map back to TS source so consumers can navigate into the library in their editor. |

### 2.2 Deliberately *not* enabled

| Flag | Why not |
|---|---|
| `exactOptionalPropertyTypes` | The doc-prescribed `Detection.classId?: number` etc. would become `number` (key missing only) rather than `number \| undefined`. This forces unusual idioms on library consumers for diminishing real-world benefit. Revisit if it later becomes obvious it would catch a real bug. |
| `noUnusedLocals` / `noUnusedParameters` | Would fight the TDD stubs in `bbox.ts`, `iou.ts`, `linalg.ts` (every parameter is currently unused inside `throw new Error('not implemented')`). The underscore-prefix convention (`_b: BBox`) is used instead as a self-documenting signal; flip `noUnusedParameters` on later once stubs are filled in, and the convention will keep new code honest. |

### 2.3 Module system: ESNext + bundler resolution

**Decision:** `module: ESNext`, `moduleResolution: bundler`, source files use
`.js` extensions in imports (e.g. `import type { BBox } from '../types.js'`).

**Why this combination:**

- `moduleResolution: bundler` lets us write `.js` extensions in source while
  the type-checker resolves them to the corresponding `.ts` file. This is
  the modern recommended setting for libraries that will be consumed by
  bundlers *and* native Node ESM.
- Including the `.js` extension in source imports means the **emitted** ESM
  output works under native Node ESM without further rewriting (Node ESM
  requires fully-specified specifiers).

**Tradeoff accepted:** Contributors must remember to write `.js` even when
the source file is `.ts`. The first time a contributor forgets, the
type-checker will not catch it, but Node ESM will at runtime. Documenting
this in CONTRIBUTING.md is a follow-up.

---

## 3. Package layout

### 3.1 Two packages: `core` (published) and `eval` (private)

**Decision:** Split into `packages/core` and `packages/eval`.

**Why:**
- `core` ships to end users and **must** have `dependencies: {}`.
- `eval` runs benchmark + metric computation offline, may legitimately want
  Node-only dev dependencies (CSV parsing, file I/O, etc.).

Keeping them separate is the only structurally honest way to enforce the
zero-dependency constitutional rule on the published artifact. Architecture
doc §3 specifies exactly this split.

### 3.2 Package names

| Workspace folder | npm name | Published? |
|---|---|---|
| `packages/core` | `vestige.js` | Yes (eventually) |
| `packages/eval` | `@vestige.js/eval` | No (`"private": true`) |

`@vestige.js/eval` uses a scoped name so it cannot be accidentally
published as a public package (npm scopes require explicit `--access public`).

### 3.3 Exports map and sideEffects

```jsonc
"sideEffects": false,
"exports": {
  ".": { types, import, require },
  "./geometry": { types, import, require }
}
```

- `sideEffects: false` opts every consumer's bundler into aggressive
  tree-shaking. Importing only `SortTracker` from `vestige.js` should not
  pull `BotSortTracker` into the bundle (architecture §12.1).
- The `./geometry` subpath matches architecture §11.7's stated public API
  for re-using bbox/IoU primitives outside the tracker.

**Tradeoff accepted:** Adding more subpath exports later (e.g. `./async`
when the async tracker family lands in v0.3) requires updates in three
places: `tsup.config.ts` entries, `package.json` exports, and the
documentation. This is acceptable friction for the clarity it gives
consumers.

### 3.4 No top-level re-exports of geometry primitives

**Decision:** `vestige.js` (root) re-exports types only. To use `iou`,
`giou`, etc., users must import from `vestige.js/geometry`.

**Why:** Architecture §11.7 endorses the subpath approach; mixing in a
top-level re-export creates two ways to do the same thing, which always
drifts.

---

## 4. API surface in stubs

### 4.1 `BBox` as a `readonly` tuple

```ts
export type BBox = readonly [number, number, number, number];
```

The `readonly` modifier prevents `b[0] = 5` accidents inside tracker code
that operates on detector outputs. Architecture §4.1 specifies this.

### 4.2 Float64Array everywhere in numerics

Cost matrices, Cholesky factors, intermediate buffers — all `Float64Array`.

**Why double-precision and not Float32?**

- The 4×4 innovation covariance solves in the Kalman filter are
  numerically sensitive when detections are tightly clustered. Float32
  accumulates error to the point of identity instability over long
  sequences.
- Architecture §5.3 explicitly chooses Cholesky over naive inverse for the
  same reason; using Float32 would defeat that choice.
- JavaScript native `number` is already a double, so `Float64Array` is the
  "no conversion" path for arithmetic anyway.

### 4.3 Linalg signatures: generic, not fixed-size

Architecture §5.3 advocates fixed-size unrolled routines (8×8 matMul, 4×8
matMul). The stub exposes `matMul(a, b, m, k, n, out?)` instead.

**Reasoning:**
- A correct generic implementation is dramatically easier to test.
- The contract (matrix sizes, output layout) does not change between
  generic and specialized versions, so specialization is an optimization,
  not an API change.
- Specialize after profiling, not before. Premature unrolling commits the
  project to maintenance cost before there's any evidence it matters.

**Tradeoff accepted:** A pure-JS generic matMul will be slower per call
than an unrolled 8×8. Acceptable for v0.1; revisit in v1.0 hot-path
profiling.

### 4.4 Optional `out` parameters on linalg functions

Every linalg function accepts an optional `out?: Float64Array`. Architecture
§9.3 calls out "pre-allocated scratch buffers" as a per-frame allocation
strategy. The `out` pattern is how the tracker passes those scratch
buffers in without forcing every caller to allocate.

**Tradeoff accepted:** Two code paths to test per function (with and
without `out`). The test suite currently covers the `out=undefined` case
plus a single explicit-`out` smoke test on `matMul`. Coverage will need to
expand once the trackers actually use the scratch-buffer path.

### 4.5 Cholesky throws on non-PD; `choleskySolve` takes the factor

```ts
cholesky(A, n)              // -> L, or throws
choleskySolve(L, b, n)      // -> x such that A x = b
```

**Why this shape:** matches the LAPACK / scipy convention. The factor `L`
can be reused across multiple right-hand sides (multiple track measurement
updates per frame for the same innovation covariance), which the
single-call API `choleskySolveDirect(A, b)` would not allow. Throwing on
non-PD is the assertive-fail-fast choice — the alternative (returning
NaN-filled output) is the bug class that ruins tracker tests silently.

### 4.6 Underscore-prefix params in stubs

Every stub function uses `_b: BBox` rather than `b: BBox`. With
`noUnusedParameters` *off* this has no compiler effect. With it eventually
*on*, the underscore prefix is the conventional TS escape hatch for
"intentionally unused." Today the convention is purely documentary; it
flips on automatically when the lint setting flips on, with zero churn.

---

## 5. Tests

### 5.1 Hand-verified oracle values, not externally-computed fixtures

Unit tests use oracle values worked out by hand or by inspection
(e.g. `iou([0,0,10,10], [0,5,10,15]) = 50/150`). They do not pre-compute
values via scipy/numpy and commit them as fixtures.

**Why:**
- Keeps the test suite zero-dependency (no Python in the loop).
- Forces reviewer-level understanding of what is being tested.
- Small enough corner cases that hand-verification is reliable.

**Tradeoff accepted:** When tests for the Kalman filter and Hungarian
land in a future scaffold pass, scipy-derived fixtures will be needed
because hand-verifying a Kalman update for an 8-dimensional state is
not realistic. That layer of tests should commit the fixture *and* the
scipy script that produced it (per architecture §13.3).

### 5.2 Property generators bound away from degeneracy

The `positiveBBox` generator in property tests requires width and height
≥ 0.5. This avoids tripping the xyah conversion math (division by `h`) and
zero-area edge cases that are tested separately in the unit suite.

**Tradeoff accepted:** Property tests do not currently exercise the
degenerate-bbox branches. Those are tested by explicit unit cases in
`bbox.test.ts` instead. Good enough — properties cover the dense interior
of the input space; unit tests pin the boundary.

### 5.3 Throw-asserting tests must match the error message

The cholesky non-PD test asserts `.toThrow(/positive[\s-]?definite/i)`, not a
bare `.toThrow()`. Reason: every stub in this scaffold throws
`Error('not implemented')`, so a bare `.toThrow()` would be green against
an unimplemented function — a false positive that defeats the TDD signal.
Tests that assert *that* a function throws must also constrain *what* it
throws, so they stay red until real behavior lands.

### 5.4 Vitest config: globals off

`globals: false` in `vitest.config.ts`. Every test file imports
`describe, it, expect` explicitly.

**Why:** Explicit imports survive refactors better, are friendlier to
the TS server, and remove the need to inject `vitest/globals` into the
`types` array of every tsconfig.

---

## 6. Things deliberately not in the v0.0 scaffold

| Item | Reason |
|---|---|
| Trackers (SortTracker, ByteTracker, …) | Out of scope for foundation TDD. Layer 2. |
| Hungarian solver | Layer 2. |
| Kalman filter | Layer 2 — depends on linalg being green. |
| Plugin interfaces (MotionPredictor, etc.) | v0.3 per architecture §15.3. |
| Async tracker variants | v0.3 per architecture §8.3. |
| Decision log entries for the tracker algorithms | Will be added when each tracker lands. |
| `validation/` cross-implementation directory | Added when the first tracker reaches snapshot-test stability. |
| `examples/` content | Empty placeholder. Browser-YOLOv8 example is v0.1 deliverable per §15.1. |
| `benchmarks/` content | Empty placeholder. Filled out with `mitata`/`tinybench` once there's anything to benchmark. |
| LICENSE | Not chosen yet. |
| CI config | Premature — no implementations to validate. |
| size-limit | Premature — no bundle to size. |

---

## 7. Open questions / things to revisit

1. **`exactOptionalPropertyTypes`** — re-evaluate when the first tracker is
   landing. If `Track`'s optional fields cause shape mismatches during
   internal updates, this flag might prevent a real class of bug.
2. **Generic vs. specialized linalg** — profile after Kalman lands. If the
   inner loop is dominated by `matMul(8,8,8)`, hand-unroll that one call
   site and benchmark.
3. **Single-package vs. multi-package within `core`** — currently
   `packages/core` is one package with multiple entry points. If `geometry`
   primitives start being depended on by more than one consumer (e.g. eval
   reuses them), splitting into `@vestige.js/geometry` + `vestige.js` could
   be considered. Not for v0.x.

---

## 8. Summary table of tradeoffs

| Decision | What we gain | What we give up |
|---|---|---|
| pnpm | Strict resolution, fast installs | Contributors must install pnpm |
| tsup | Fast dual-format builds, conventional | A small config DSL to maintain |
| Vitest | Native ESM/TS, watch mode | Node 18+ for dev |
| Strict TS minus exactOptional + minus noUnused* | Catches real bugs; doesn't fight stubs | Slightly less strict than maximum |
| Generic linalg | Easy to test, easy to read | Slower per call than unrolled |
| Float64 everywhere | Numerical stability | 2× memory vs Float32 |
| `out?` buffer params | Zero-alloc hot path possible | Two code paths per function |
| Cholesky throws on non-PD | Fast-fail | Caller must handle the throw |
| Hand-verified test oracles | Zero deps, reviewer comprehension | Doesn't scale to Kalman/Hungarian |
| Property generators avoid degeneracy | Stable runs | Edge cases need separate unit tests |
| Subpath-only `./geometry` export | One way to do it | Slightly more verbose imports |
| `verbatimModuleSyntax` | Catches dual-package hazards | Contributors must write `import type` |
| Two packages (core / eval) | Constitutional zero-deps on published artifact | Two `package.json`s to keep aligned |
