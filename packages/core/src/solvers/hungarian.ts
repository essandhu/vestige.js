/**
 * Linear sum assignment via the Jonker-Volgenant shortest-augmenting-path
 * algorithm (Jonker & Volgenant, *Computing* 1987). See ARCHITECTURE.md §5.4.
 *
 * Inputs are row-major Float64Array cost matrices. Forbidden cells are
 * encoded as `Number.POSITIVE_INFINITY` and are never selected; if every
 * augmenting path from some row goes through a forbidden cell, that row
 * remains unmatched and the solver proceeds with the remaining rows.
 */

// biome-ignore-all lint/style/noNonNullAssertion: indices into the internal
// scratch buffers are bounded by the loop counters and the rectangular m/n
// contracts. Asserting at each read avoids `number | undefined` from
// `noUncheckedIndexedAccess` in the inner Dijkstra loops without runtime cost.

/**
 * Result of a linear sum assignment problem solve.
 *
 * `rowToCol` and `colToRow` are parallel inverse mappings:
 *
 * - `rowToCol[i]` is the column matched to row i, or -1 if row i is unmatched.
 * - `colToRow[j]` is the row matched to column j, or -1 if column j is unmatched.
 *
 * Unmatched entries appear when the cost matrix is rectangular (|M − N|
 * rows or columns have no partner) or when forbidden cells block every
 * augmenting path for some row.
 *
 * `totalCost` is the sum of `cost[i, rowToCol[i]]` over all matched rows.
 * Always finite (forbidden cells are never selected).
 */
export interface LsapResult {
  readonly rowToCol: Int32Array;
  readonly colToRow: Int32Array;
  readonly totalCost: number;
}

/**
 * Solve the linear sum assignment problem on a row-major M × N cost matrix
 * using the shortest-augmenting-path Jonker-Volgenant algorithm. See
 * ARCHITECTURE.md §5.4 for the choice of JV over Munkres.
 *
 * The minimum-cost matching has size `min(M, N)` when the cost matrix has
 * no forbidden cells. Cells set to `Number.POSITIVE_INFINITY` are never
 * selected — if every augmenting path from some row goes through a forbidden
 * cell, that row is left unmatched and the solver proceeds with the rest.
 *
 * Tie-breaking is deterministic: same input → same output, bit-for-bit on
 * the same machine. This is the property that makes snapshot regression
 * tests viable (ARCHITECTURE.md §2.5, §10.5).
 *
 * The input matrix is not mutated.
 *
 * @param cost row-major cost matrix of length `m * n`
 * @param m number of rows (≥ 0)
 * @param n number of columns (≥ 0)
 */
export function solveLsap(cost: Float64Array, m: number, n: number): LsapResult {
  const rowToCol = new Int32Array(m).fill(-1);
  const colToRow = new Int32Array(n).fill(-1);
  if (m === 0 || n === 0) {
    return { rowToCol, colToRow, totalCost: 0 };
  }

  // Always augment from the smaller dimension; transpose internally if m > n.
  const transposed = m > n;
  const mi = transposed ? n : m;
  const ni = transposed ? m : n;
  const cellCost = transposed
    ? (i: number, j: number): number => cost[j * n + i]!
    : (i: number, j: number): number => cost[i * n + j]!;

  // col4row[i] = column matched to internal row i; row4col[j] = row matched to internal col j.
  const col4row = new Int32Array(mi).fill(-1);
  const row4col = new Int32Array(ni).fill(-1);

  // Dual variables (row and column potentials) and Dijkstra scratch buffers.
  const u = new Float64Array(mi);
  const v = new Float64Array(ni);
  const shortestPathCosts = new Float64Array(ni);
  const path = new Int32Array(ni);
  const SR = new Uint8Array(mi);
  const SC = new Uint8Array(ni);
  const remaining = new Int32Array(ni);

  for (let curRow = 0; curRow < mi; curRow++) {
    for (let j = 0; j < ni; j++) {
      shortestPathCosts[j] = Number.POSITIVE_INFINITY;
      path[j] = -1;
      remaining[j] = j;
      SC[j] = 0;
    }
    for (let i = 0; i < mi; i++) SR[i] = 0;

    let numRemaining = ni;
    let sink = -1;
    let minVal = 0;
    let i = curRow;
    let failed = false;

    while (sink === -1) {
      SR[i] = 1;
      let lowest = Number.POSITIVE_INFINITY;
      let indexLowest = -1;
      const ui = u[i]!;

      for (let it = 0; it < numRemaining; it++) {
        const j = remaining[it]!;
        const r = minVal + cellCost(i, j) - ui - v[j]!;
        if (r < shortestPathCosts[j]!) {
          path[j] = i;
          shortestPathCosts[j] = r;
        }
        const spcj = shortestPathCosts[j]!;
        // Tie-break: lower remaining-index wins on a strict-less check, and a
        // free column displaces an already-matched column at the same cost.
        // The latter clause is what scipy's `_lsap.cpp` does — it shortens
        // augmenting paths on tie cases by preferring an immediate sink.
        if (spcj < lowest || (spcj === lowest && row4col[j]! === -1)) {
          lowest = spcj;
          indexLowest = it;
        }
      }

      if (lowest === Number.POSITIVE_INFINITY) {
        // No augmenting path from curRow — this row stays unmatched. The
        // dual variables and matching arrays are unchanged from before
        // this row's Dijkstra began, so subsequent rows are unaffected.
        failed = true;
        break;
      }

      const jSink = remaining[indexLowest]!;
      if (row4col[jSink]! === -1) {
        sink = jSink;
      } else {
        i = row4col[jSink]!;
      }
      SC[jSink] = 1;
      remaining[indexLowest] = remaining[--numRemaining]!;
      minVal = lowest;
    }

    if (failed) continue;

    // Update duals along the alternating tree built by Dijkstra.
    u[curRow] = u[curRow]! + minVal;
    for (let i2 = 0; i2 < mi; i2++) {
      if (SR[i2]! === 1 && i2 !== curRow) {
        u[i2] = u[i2]! + minVal - shortestPathCosts[col4row[i2]!]!;
      }
    }
    for (let j = 0; j < ni; j++) {
      if (SC[j]! === 1) {
        v[j] = v[j]! - minVal + shortestPathCosts[j]!;
      }
    }

    // Augment along the discovered path.
    let s = sink;
    while (true) {
      const i2 = path[s]!;
      row4col[s] = i2;
      const swap = col4row[i2]!;
      col4row[i2] = s;
      s = swap;
      if (i2 === curRow) break;
    }
  }

  // Materialize public mappings.
  let totalCost = 0;
  if (transposed) {
    for (let i = 0; i < mi; i++) {
      const j = col4row[i]!;
      if (j !== -1) {
        // Internal (i, j) ↔ external (j, i).
        rowToCol[j] = i;
        colToRow[i] = j;
        totalCost += cost[j * n + i]!;
      }
    }
  } else {
    for (let i = 0; i < mi; i++) {
      const j = col4row[i]!;
      if (j !== -1) {
        rowToCol[i] = j;
        colToRow[j] = i;
        totalCost += cost[i * n + j]!;
      }
    }
  }

  return { rowToCol, colToRow, totalCost };
}
