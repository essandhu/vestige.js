/**
 * The two association conventions used by the tracking-by-detection references.
 *
 * {@link solveLsap} answers "what is the minimum-cost MAXIMUM-CARDINALITY matching?"
 * That is the right question for a bare assignment problem, and it is the wrong
 * question for every tracker in this family. A tracker must be free to leave a track
 * unmatched when its only remaining options are bad — otherwise a track whose real
 * detection is occluded will be shoved onto a neighbour's box, stealing it from the
 * track that actually owns it. Each reference has its own way of expressing that,
 * and this module implements both of them exactly. See ADR-0005.
 *
 * Neither function gates cells to `+Infinity` and then solves. That combination
 * (which is what vestige originally did) is equivalent to neither convention: it
 * hard-forbids the bad cells but then *maximizes cardinality* over whatever is
 * left, which is precisely the behaviour both references go out of their way to
 * avoid.
 */

// biome-ignore-all lint/style/noNonNullAssertion: indices into row-major cost /
// score buffers are bounded by the M*N rectangular contract, as in hungarian.ts.

import { solveLsap } from './hungarian.js';

/**
 * ByteTrack's convention — a **rejection cost**.
 *
 * Reference: `yolox/tracker/matching.py:linear_assignment`, which calls
 * `lap.lapjv(cost, extend_cost=True, cost_limit=thresh)`. Empirically (and this is
 * the whole ballgame) that minimizes
 *
 * ```text
 *   Σ_{matched (i,j)} cost[i][j]  +  rejectCost × (number of unmatched ROWS)
 * ```
 *
 * with cells where `cost > rejectCost` forbidden outright. Columns carry no
 * penalty for going unmatched — only rows do. So a row takes a detection only when
 * doing so is cheaper than *declining*, and declining is always available at a flat
 * price of `rejectCost`.
 *
 * That is exactly a standard assignment problem once every row is handed its own
 * private "decline" column priced at `rejectCost`, which is how this is
 * implemented: extend the M × N matrix to M × (N + M), where column `N + i` is
 * reachable only from row `i`. Every row then has a finite option, so the solver
 * matches all M rows, and a row that lands on its own dummy column is reported as
 * unmatched.
 *
 * Note `cost === rejectCost` is admissible (the reference's `cost_limit` is an
 * inclusive upper bound), matching the `cost > cutoff` gate direction vestige
 * already used.
 *
 * @param cost row-major M × N cost matrix (lower is better)
 * @param rejectCost the reference's `cost_limit` — ByteTrack's `match_thresh`
 * @returns `rowToCol[i]` = matched column, or -1 if row i declined
 */
export function solveWithRejectionCost(
  cost: Float64Array,
  m: number,
  n: number,
  rejectCost: number,
): Int32Array {
  const rowToCol = new Int32Array(m).fill(-1);
  if (m === 0 || n === 0) return rowToCol;

  const width = n + m;
  const extended = new Float64Array(m * width).fill(Number.POSITIVE_INFINITY);
  for (let i = 0; i < m; i++) {
    const src = i * n;
    const dst = i * width;
    for (let j = 0; j < n; j++) {
      const c = cost[src + j]!;
      extended[dst + j] = c > rejectCost ? Number.POSITIVE_INFINITY : c;
    }
    // Row i's private decline column. Always finite, so every row is matchable and
    // the max-cardinality solver's cardinality forcing becomes a no-op.
    extended[dst + n + i] = rejectCost;
  }

  const solved = solveLsap(extended, m, width);
  for (let i = 0; i < m; i++) {
    const j = solved.rowToCol[i]!;
    rowToCol[i] = j >= 0 && j < n ? j : -1;
  }

  // Tie repair. A real cell costing EXACTLY `rejectCost` is worth precisely as much
  // as declining, so both choices are optimal and the solver may land on either.
  // The reference breaks that tie toward the real match, so we must too.
  //
  // Moving a declined row onto a FREE real column is cost-neutral by construction:
  // at an optimum no free column can cost LESS than `rejectCost` (the solver would
  // already have taken it), so any free column we find here costs exactly
  // `rejectCost`. Swapping to it leaves the objective unchanged and only changes
  // which of two equally-optimal assignments we report.
  const columnTaken = new Uint8Array(n);
  for (let i = 0; i < m; i++) {
    const j = rowToCol[i]!;
    if (j >= 0) columnTaken[j] = 1;
  }
  for (let i = 0; i < m; i++) {
    if (rowToCol[i] !== -1) continue;
    for (let j = 0; j < n; j++) {
      if (columnTaken[j] === 1) continue;
      if (cost[i * n + j]! <= rejectCost) {
        rowToCol[i] = j;
        columnTaken[j] = 1;
        break;
      }
    }
  }
  return rowToCol;
}

/**
 * SORT's and OC-SORT's convention — **solve, then filter**.
 *
 * Reference: `sort.py:associate_detections_to_trackers` and
 * `OC_SORT/trackers/ocsort_tracker/association.py:associate`. Both:
 *
 * 1. Solve the **full, ungated** matrix, maximizing score
 *    (`linear_assignment(-score_matrix)`).
 * 2. **Then** discard any matched pair whose score is below `threshold`.
 *
 * The order is the point. Solving first lets a strong pair win the column it
 * deserves; filtering afterwards drops the junk pairs the solver was forced to make
 * to fill out its cardinality. Gating first and solving second inverts that — the
 * junk pairs become the *only* pairs, and they displace the strong ones.
 *
 * Both references also short-circuit: if the above-threshold cells already form a
 * partial matching (at most one per row and per column), they are used directly and
 * the solver is skipped. Replicated here, including the reference's strictness —
 * the shortcut's mask is `score > threshold` (strict) while the post-filter is
 * `score < threshold → drop` (so `score === threshold` survives the filter but does
 * not count toward the shortcut).
 *
 * `solveScore` and `filterScore` are separate because OC-SORT's primary stage
 * solves on `iou + angle_diff_cost` but filters on raw `iou`. For SORT, and for
 * OC-SORT's secondary stages, callers pass the same array for both.
 *
 * @param solveScore row-major M × N score matrix to MAXIMIZE
 * @param filterScore row-major M × N score matrix the threshold applies to
 * @param threshold matches scoring below this are dropped after the solve
 * @param useShortcut replicate the reference's unambiguous-matching short-circuit
 * @param requireAnyAboveThreshold when set, produce no matches at all unless some
 *   `filterScore` cell is strictly above `threshold`. This is OC-SORT's
 *   `if iou_left.max() > self.iou_threshold:` guard on its BYTE and OCR stages; it
 *   differs observably from plain solve-then-filter only when the maximum cell sits
 *   *exactly* on the threshold (the guard rejects it, the filter would keep it).
 * @returns `rowToCol[i]` = matched column, or -1 if row i is unmatched
 */
export function solveThenFilter(
  solveScore: Float64Array,
  filterScore: Float64Array,
  m: number,
  n: number,
  threshold: number,
  useShortcut: boolean,
  requireAnyAboveThreshold = false,
): Int32Array {
  const rowToCol = new Int32Array(m).fill(-1);
  if (m === 0 || n === 0) return rowToCol;

  if (requireAnyAboveThreshold) {
    let anyAbove = false;
    for (let k = 0; k < m * n; k++) {
      if (filterScore[k]! > threshold) {
        anyAbove = true;
        break;
      }
    }
    if (!anyAbove) return rowToCol;
  }

  if (useShortcut) {
    // The reference's `a = (score > threshold)` mask, and its
    // `a.sum(1).max() == 1 and a.sum(0).max() == 1` test: the above-threshold cells
    // already form a partial matching, so no solver is needed. Note `== 1`, not
    // `<= 1` — an all-zero mask falls through to the solve, which then filters
    // everything out anyway, so the two paths agree.
    const rowCounts = new Int32Array(m);
    const colCounts = new Int32Array(n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        if (filterScore[i * n + j]! > threshold) {
          rowCounts[i]!++;
          colCounts[j]!++;
        }
      }
    }
    let maxRow = 0;
    for (let i = 0; i < m; i++) if (rowCounts[i]! > maxRow) maxRow = rowCounts[i]!;
    let maxCol = 0;
    for (let j = 0; j < n; j++) if (colCounts[j]! > maxCol) maxCol = colCounts[j]!;
    if (maxRow === 1 && maxCol === 1) {
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          if (filterScore[i * n + j]! > threshold) rowToCol[i] = j;
        }
      }
      return rowToCol;
    }
  }

  // Solve the FULL, UNGATED matrix. `linear_assignment(-score)` in both references.
  const cost = new Float64Array(m * n);
  for (let k = 0; k < m * n; k++) cost[k] = -solveScore[k]!;
  const solved = solveLsap(cost, m, n);

  // ...and only NOW drop the sub-threshold pairs.
  for (let i = 0; i < m; i++) {
    const j = solved.rowToCol[i]!;
    if (j !== -1 && filterScore[i * n + j]! >= threshold) rowToCol[i] = j;
  }
  return rowToCol;
}
