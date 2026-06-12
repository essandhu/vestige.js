# 0004 — Eval package: metric oracles, layering, and core source-bridge

- **Date:** 2026-06-11
- **Status:** Accepted
- **Scope:** `packages/eval/` — the MOTChallenge evaluation harness
  (ARCHITECTURE.md §10): format parsing, CLEAR-MOT / ID / HOTA metrics, and the
  sequence runner.

## Context

ARCHITECTURE.md §10.4 makes benchmark numbers the acceptance criterion for
every tracker ("if a tracker fails the window, it's not released"), but
through v0.2 the eval package was an empty stub — none of the three shipped
trackers (SORT, ByteTrack, OC-SORT) could actually be validated against the
published numbers. This ADR records the design choices made when the harness
was implemented.

## 1. TrackEval is the canonical oracle; tests are hand-traced miniatures

`JonathonLuiten/TrackEval` is the reference implementation for **all three**
metric families (CONTRIBUTING.md §4.2 already names it for HOTA / MOTA / IDF1).
Where the literature and TrackEval disagree in convention, we follow TrackEval,
because published MOT17/MOT20 leaderboard numbers are produced by it.
Concretely:

- **MOTP is mean IoU over matches** (similarity form), not the legacy
  mean-distance form of the 2008 CLEAR paper.
- **CLEAR matching uses the `1000 ·` previous-match continuity bonus** before
  Hungarian assignment, so IDSW counts match the devkit's.
- **IDSW compares against the gt id's last-ever matched tracker id** (gaps do
  not reset it).
- **MT is `ratio > 0.8` strictly; ML is `ratio < 0.2` strictly** (TrackEval's
  `np.greater` / `np.less`).
- **HOTA's per-alpha LocA with zero TPs is 1** (TrackEval's
  `np.maximum(1e-10, …)` guard).
- **Threshold comparisons use `sim ≥ thr − ε`** (`ε = Number.EPSILON`),
  mirroring TrackEval's `np.finfo('float').eps` slack, so boxes engineered to
  sit exactly on a threshold land on the matched side deterministically.

Unlike the Kalman/Cholesky fixtures (ADR-0002), the metric tests use
**hand-traced miniature sequences, not Python-generated fixtures**: every
oracle value (e.g. "split track ⇒ AssA = 0.5, HOTA = √0.5") is derivable by
hand from the papers' formulas, which CONTRIBUTING.md §4.4 ranks above
fixture-loaded values. A committed TrackEval-output fixture becomes worthwhile
only when we run a real MOT17 sequence — deferred until the detection-file
download story (§16.2) lands.

## 2. Aggregation scope: single sequence only

All metrics take one sequence (`ReadonlyArray<EvalFrame>`). Multi-sequence /
benchmark-level aggregation (TrackEval's `COMBINED_SEQ`: pooling counts, not
averaging ratios) is the runner's future concern, not the metric modules'.
Keeping the metric signature sequence-shaped means there is exactly one place
where pooling semantics can go wrong, and it isn't written yet.

## 3. `EvalFrame` decouples metrics from the MOTChallenge format

The metrics consume a minimal `EvalFrame { gtIds, gtBoxes, trackIds,
trackBoxes }` shape rather than parsed `MotEntry` rows. Synthetic-scenario
tests (and future DanceTrack-style loaders) construct `EvalFrame`s directly;
only `runner.evalFramesFromEntries` knows how MOT files map onto it. The
dense-id/similarity preprocessing shared by the three metrics lives in
`metrics/frames.ts` (`indexSequence`), mirroring TrackEval's
`get_preprocessed_seq_data` split.

## 4. Degenerate contract: empty ground truth throws

A sequence with zero gt detections has no defined MOTA/IDF1/HOTA (TrackEval
emits NaNs from `0/0`). Per CONTRIBUTING.md §3.2, instead of propagating NaN
we throw with a matchable message (`/ground[\s-]?truth/i`). An empty *tracker*
output is well-defined (everything is a miss) and returns scores normally.

## 5. Eval reaches core through a source bridge, not the built dist

`packages/eval/src/core.ts` re-exports what eval needs from
`../../core/src/...` directly. Alternatives considered:

- **Import the `vestige.js` workspace dep** (dist): makes `pnpm test` /
  `pnpm typecheck` depend on a prior `pnpm build`, and CI runs tests before
  build. Also `solveLsap` is deliberately not in the public API yet
  (ARCHITECTURE.md §11 exposes geometry only; the associator surface arrives
  with the v0.3 plugin work), and eval should not force that decision early.
- **tsconfig `paths` + vitest aliases**: same effect, two extra config
  surfaces to keep in sync, and the dependency is less greppable than an
  explicit re-export module.

The bridge module is the single allowed crossing point so the dependency
stays auditable; eval code importing `../../core/...` anywhere else is a
review flag. This is an intra-monorepo convenience of a private package — it
adds nothing to the published API.

## 6. Preprocessing scope deltas vs. the MOTChallenge devkit

`filterGt` covers the consider-flag, class, and visibility filters. TrackEval's
full MOT17 preprocessing additionally **removes tracker detections that match
distractor-class gt** before scoring. That step was not implemented when this
ADR was written; until it was, numbers computed on real MOT17 gt files would
read slightly lower than the leaderboard's. It belongs in the loader layer,
not the metrics.

**Status update:** closed by `motchallenge/preprocess.ts`
(`preprocessMotSequence`), which ports TrackEval's
`get_preprocessed_seq_data` (distractor matching + zero-mark/class gt
filtering; MOT Challenge has no crowd-ignore regions per TrackEval's own
docstring). The whole parse → preprocess → metrics pipeline is pinned against
a TrackEval-generated fixture (`packages/eval/fixtures/trackeval-preproc/`),
which also serves as the §1 cross-implementation check the metrics previously
lacked. Remaining before §10.4 numbers: real MOT17 sequence data + the
detector-file download story (ARCHITECTURE.md §16.2) and multi-sequence
pooling (§2 above).
