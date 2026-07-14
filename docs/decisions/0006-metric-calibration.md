# 0006 — The metrics are an instrument, and instruments get calibrated

- **Date:** 2026-07-13
- **Status:** Accepted
- **Scope:** `packages/eval/src/metrics/` (HOTA, CLEAR-MOT, Identity) and `solvers/hungarian.ts`'s tie-breaking. Introduces `packages/eval/fixtures/trackeval-metrics/` and `packages/core/fixtures/lsap-scipy-ties/`.

## Context

Every tracker in this repo is validated against its Python reference by a
cross-implementation fixture. The metrics that will *measure* those trackers had no
such fixture. Their unit tests were honest about it:

> *"Oracle values hand-traced from the HOTA definition (Luiten et al., IJCV 2020, eqs. 5–13) as implemented by TrackEval `metrics/hota.py`."*

Hand-traced from the definition — not generated from the reference. That is the wrong
shape of assurance for this particular code, for a reason that is easy to miss:

**The metrics are the instrument we measure the trackers with.** A tracker bug shows up
as a bad number. A *metric* bug shows up as a bad number too — and it looks exactly the
same. If HOTA's AssA denominators are subtly wrong, every published figure is wrong,
the delta against the paper looks like a tracker faithfulness gap, and we would go
hunting in `ocsort.ts` for a bug that lives in `hota.ts`. An uncalibrated instrument
doesn't just produce wrong readings; it produces *misleading diagnoses*.

This ADR was written before the first MOT17 run, which is the only time it is cheap.

## What calibration found

Building `trackeval-metrics/` — nine synthetic scenarios driven through TrackEval's own
`CLEAR`, `Identity` and `HOTA` classes, then through ours — turned up two live bugs and
exonerated the rest.

### 1. CLEAR-MOT wiped its continuity memory on empty frames

`clearmot.ts` committed `prevFrameTrack.set(currFrameTrack)` **outside** the
`if (m > 0 && n > 0)` guard. TrackEval `clear.py:70-76` hits `continue` on a timestep
with zero gt or zero tracker detections — *before* the `prev_timestep_tracker_id[:] = np.nan`
reset further down. Its continuity memory survives the hole; ours was blanked by it.

Two consequences, both on data that is entirely routine (a detector drops out for a
frame; preprocessing filters every gt in a frame to a distractor):

- the `+1000` continuity bonus stops protecting the existing gt↔tracker pairing on the
  frame after the hole, so the matcher is free to pick a different tracker id — a
  **spurious ID switch**; and
- `prevFrameTrack[gtId] === -1` reads as "was not tracked", so the resumed track counts
  as a new segment — a **spurious Frag**.

The subtlety worth internalizing is the *asymmetry* in TrackEval's bookkeeping, because
the naive mental model gets it backwards. IDSW is scored off `prev_tracker_id` — the
**last-ever** matched id, which survives any gap. Frag and the matching bonus are driven
by `prev_timestep_tracker_id` — the **immediately previous frame**. An empty timestep is
skipped without resetting *either*. So across a dropout: an id change still counts as a
switch, but the resumption does **not** count as a fragment.

Two committed unit tests asserted `Frag = 1` across exactly such a gap. They were
hand-traced, they were wrong, and they had been passing green against the bug the whole
time. Both were re-verified against TrackEval on their exact sequences and corrected to
`Frag = 0`. **A hand-written oracle is a hypothesis, not a measurement.**

### 2. `solveLsap` broke assignment ties backwards

`solveLsap` is a port of scipy's `_lsap.cpp`. scipy initializes its `remaining` column
array in **reverse** (`remaining[it] = nc - it - 1`); vestige filled it forward. The
Dijkstra scan keeps the first strict minimum, so the fill order decides **every tie** —
and forward fill inverts scipy's choice on all of them. On an all-zero 3×3, scipy
returns `[0,1,2]` and vestige returned `[2,1,0]`. Across 507 tie-dense matrices, **276
disagreed**.

This is not cosmetic, because `solveLsap` backs the metrics' internal gt↔prediction
matching and **TrackEval matches with `scipy.optimize.linear_sum_assignment`**. IDF1 in
particular assigns over *integer* match counts, where exact ties are routine.

It was listed as a deferred item in ADR-0005 ("if a crowded fixture ever fails on an id
permutation with matching bboxes, this is the first suspect"). Deferring it was a
mistake: the tracker fixtures use well-separated boxes, so no two IoUs are ever exactly
equal and no tie ever arises. **The place it actually bites is the one place no fixture
was looking.**

### 3. HOTA was already correct

All 19 alphas, DetA, AssA and LocA matched TrackEval exactly across all nine scenarios,
with no changes. The hand-traced oracles were right. This is worth stating plainly: the
audit's value was not that everything was broken.

### 4. `identity.ts` carried an epsilon it should not have

`identity.py:55` is a bare `np.greater_equal(sim, threshold)`, while `clear.py:82` is
`sim < threshold - np.finfo('float').eps`. TrackEval is internally inconsistent here;
we now match it metric-for-metric. The window is ~1 ULP and cannot move a real number —
fixed because it is free, not because it mattered.

## Decision

**Any code whose output is a published number must be validated against the
implementation that produced the published numbers it will be compared to.** For the
metrics that is TrackEval, pinned at `12c8791b`.

Concretely:

| Fixture | Pins |
|---|---|
| `packages/eval/fixtures/trackeval-metrics/` | HOTA / CLEAR / Identity against TrackEval's own metric classes, over nine scenarios |
| `packages/core/fixtures/lsap-scipy-ties/` | `solveLsap`'s tie-breaking against scipy, over 507 tie-dense matrices |

The scenarios are chosen adversarially, not representatively — `empty-tracker-frame`,
`empty-gt-frame`, `tied-boxes` and `truncated-tracker` exist because those are the
regimes where a metric goes quietly wrong, and every one of them is *ordinary* on real
MOT data.

Note `tied-boxes` took two attempts. The obvious construction — duplicate gt boxes and
duplicate tracker boxes — proves nothing: totals like IDTP and CLR_TP are invariant
under a tie *by definition*, so MOTA and IDF1 come out identical either way. The
scenario only discriminates once the two tied candidates have **different futures** (one
of them disappears later), which turns the tie-break into an ID switch. A tie-breaking
test that cannot fail is worse than none, because it looks like coverage.

## Consequences

- MOTA, IDSW and Frag change for any sequence containing an empty frame — i.e. most
  real ones. Any number computed before this ADR is void. This lands before the first
  benchmark run, deliberately.
- The trackers' tie-breaking is a **separate, still-open** question, recorded below.
- Two committed unit-test expectations were corrected. They were not "adjusted to make
  the build pass"; they were re-derived from the reference, and the reference disagreed
  with them.

## Things deliberately deferred

| Item | When to revisit |
|---|---|
| **Tracker-side tie-breaking.** `solveLsap` now matches scipy. But the tracker references all call `lap.lapjv`, which breaks ties differently from scipy — verified: they disagree on *every* tied matrix tried, including the trivial all-zero 3×3. So scipy parity is right for the metrics and *wrong* for the trackers. It is currently unexercised (no fixture produces an exact IoU tie), but real MOT data contains duplicate boxes. Fixing it means giving `solveLsap` a tie-break policy rather than one global rule. | Before the MOT17 run, or the first time a tracker fixture fails on an id permutation with matching bboxes. |
| Multi-sequence aggregation (MOT17 is 7 train / 14 test sequences). TrackEval accumulates **counts** across sequences and computes the metric once; averaging per-sequence metrics gives a different, wrong number. Not yet exercised — every fixture here is single-sequence. | When the benchmark runner grows multi-sequence support. This is the next thing most likely to silently corrupt a headline figure. |
