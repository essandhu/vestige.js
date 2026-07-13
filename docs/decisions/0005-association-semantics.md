# 0005 — Association semantics: each tracker uses its own reference's convention

- **Date:** 2026-07-13
- **Status:** Accepted
- **Scope:** How a solved cost matrix becomes a set of matches. Replaces the "gate to `+Infinity`, then solve" rule that ARCHITECTURE.md §5.6 mandated and all three trackers implemented. Introduces `solvers/association.ts`.

## Context

Every tracker in this family builds an `M × N` cost matrix and asks a solver for matches. Until this ADR, all three did it the same way:

```ts
cost[i][j] = score[i][j] < threshold ? Number.POSITIVE_INFINITY : 1 - score[i][j];
const { rowToCol } = solveLsap(cost, M, N);
```

That came straight from ARCHITECTURE.md §5.6, which said gating before the solve was "faster than post-filtering" — asserting an equivalence that does not hold. `solveLsap` returns the minimum-cost **maximum-cardinality** matching. Gating changes *which* cells are legal; it does not stop the solver from **maximizing how many rows get matched**. So a track whose only surviving cells are mediocre is still forced to take one, even when that means stealing the detection another track owns outright.

None of the three references does this. Each has a way for a track to simply decline, and they are not the same way:

| Reference | Call site | Convention |
|---|---|---|
| `abewley/sort` | `sort.py:170,186` | Solve the **full, ungated** IoU matrix (`linear_assignment(-iou_matrix)`), **then** drop matched pairs whose IoU `< iou_threshold`. |
| `noahcao/OC_SORT` | `association.py:277-281`, `ocsort.py:270,291` | Same solve-then-filter. The primary stage solves on `iou + angle_diff_cost` but filters on **raw** `iou`. |
| `FoundationVision/ByteTrack` | `matching.py:43` | `lap.lapjv(cost, extend_cost=True, cost_limit=thresh)` — a **rejection-cost** model. |

### The counterexample

`cost = [[0.1, 0.79], [0.79, 1.0]]`, cutoff `0.8`. Run against the pinned reference and against vestige:

| | assignment | total cost |
|---|---|---|
| `lap.lapjv(cost_limit=0.8)` | row 0 → col 0 only; row 1 **declines** | 0.10 |
| gate-then-`solveLsap` | row 0 → col **1**, row 1 → col 0 | 1.58 |

The reference declines row 1 because matching it costs more than walking away. The max-cardinality solver cannot walk away while an admissible cell exists, so it pairs both rows — and in doing so drags row 0 off the cell it clearly owned.

### Why three green fixtures did not catch it

`sort-abewley/`, `ocsort-noahcao/`, and `bytetrack-foundationvision/` all use well-separated boxes. Every cost matrix they produce has an unambiguous optimum — each track's own detection is the only admissible cell in its row. Under those conditions all four conventions (the three references' and vestige's) agree exactly. The bug is invisible until detections **compete**, which is what crowded frames are, which is what MOT17/MOT20 are. Three green cross-implementation fixtures coexisted with a bug that corrupts ids on any crowded frame.

That is the real lesson, and it generalizes past association: **a fixture only tests the regime its sequence puts the code in.** `association-crowded/` exists to cover the regime the other three structurally cannot.

## Decision

**Association is not a shared rule. Each tracker uses its own reference's convention.**

This follows ARCHITECTURE.md §2.2 ("algorithms own their options") — the same reasoning that rejected a shared `TrackerOptions` union rejects a shared association rule. The two conventions live in `solvers/association.ts`:

### `solveWithRejectionCost(cost, M, N, rejectCost)` — ByteTrack

Minimizes

```text
Σ_{matched (i,j)} cost[i][j]  +  rejectCost × (number of unmatched ROWS)
```

with cells where `cost > rejectCost` forbidden. Columns carry **no** penalty for going unmatched — only rows do. A row takes a detection only when that is cheaper than declining, and declining is always available at a flat price of `rejectCost`.

Those semantics were established empirically against `lap.lapjv`, not read off its docs. The discriminating probe is the second unit test in `tests/unit/association.test.ts`: on `[[0.1, 0.75], [0.75, 0.9]]` with limit `0.8`, matching both rows costs 1.50 while matching row 0 and declining row 1 costs 0.90 — and `lapjv` picks the latter. Had unmatched *columns* also been penalized, the totals would be 1.50 vs 1.70 and it would have picked the former. It does not, so there is no column term.

Implementation is a reduction, not a new solver: extend the matrix to `M × (N + M)` and give row `i` a private "decline" column at index `N + i` priced at `rejectCost`. Every row then has a finite option, so max-cardinality matches all `M` rows and its cardinality forcing becomes a no-op. Rows landing on their own dummy column are reported unmatched. A brute-force test over 300 random matrices confirms the reduction attains the true minimum of the objective above.

One wrinkle: a real cell costing **exactly** `rejectCost` ties with declining. The reference breaks that tie toward the real match; `solveLsap` may break it either way. `solveWithRejectionCost` repairs it afterwards, which is provably cost-neutral — at an optimum no *free* column can cost less than `rejectCost` (the solver would already have taken it), so any free column found in the repair pass costs exactly `rejectCost` and swapping onto it leaves the objective unchanged.

### `solveThenFilter(solveScore, filterScore, M, N, threshold, useShortcut, requireAnyAboveThreshold)` — SORT and OC-SORT

1. Solve the **full, ungated** matrix, maximizing `solveScore`.
2. **Then** drop matched pairs whose `filterScore < threshold`.

The order is the entire point. Solving first lets a strong pair win the column it deserves; filtering afterwards discards the junk pairs the solver was forced into to fill out its cardinality. Gate first and those junk pairs become the *only* pairs, and they displace the strong ones.

`solveScore` and `filterScore` are separate parameters because OC-SORT's primary stage solves on `iou + angle_diff_cost` but thresholds on raw `iou` — the OCM angle bonus can win a pair the solve and that pair still gets dropped if its raw IoU is too low. SORT and OC-SORT's secondary stages pass the same array twice.

Two reference quirks are replicated rather than tidied away:

- **The unambiguous-matching shortcut.** Both references skip the solver when the above-threshold cells already form a partial matching (`a.sum(1).max() == 1 and a.sum(0).max() == 1`). Note the mask is `score > threshold` (strict) while the post-filter is `score < threshold → drop` — so a cell sitting exactly on the threshold survives the filter but does not count toward the shortcut. Faithfully reproduced.
- **`requireAnyAboveThreshold`.** OC-SORT guards its BYTE and OCR stages with `if iou_left.max() > self.iou_threshold:` — strictly above. A matrix whose maximum sits *exactly* on the threshold yields no matches at all, even though the post-filter alone would have kept it.

## Consequences

- **Benchmark numbers move.** This changes association on every crowded frame, so any MOT17/MOT20 result produced before this ADR is void. Landing it *before* the first benchmark run is deliberate — publishing numbers from the old association and then silently revising them would be worse than the delay.
- **`solveLsap` is unchanged** and remains the min-cost max-cardinality primitive. It was never wrong; it was being asked the wrong question.
- **No public API change.** `iouThreshold` / `matchThresh` keep their meanings; what changed is when they are applied.
- **Cost:** `solveWithRejectionCost` solves an `M × (N + M)` matrix instead of `M × N`. Assignment is not the per-frame hot path (cost-matrix construction dominates at realistic track counts), and correctness is not negotiable for a faithfulness-first library. Revisit only if profiling says so.

## Things deliberately deferred

| Item | When to revisit |
|---|---|
| Tie-breaking parity with `lap.lapjv` on equal-cost assignments | Ties on *distinct* total costs are impossible; ties on equal totals are resolved arbitrarily but deterministically by `solveLsap`. If a crowded fixture ever fails on an id permutation with matching bboxes, this is the first suspect. |
| `strackPool` row order in ByteTracker (reference: all confirmed, then all lost; vestige: `Map` insertion order) | Permutes the cost-matrix rows, so it can only surface through solver tie-breaking. Recorded in ADR-0003 §7; cheap to fix, no known failing case. |
| Exposing the two conventions as a public `AssociationStrategy` plugin | ARCHITECTURE.md §8 anticipates this. They are internal until BoT-SORT lands and there is a third consumer to design against. |
