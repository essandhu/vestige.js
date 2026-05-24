/**
 * Linear sum assignment via the Jonker-Volgenant shortest-augmenting-path
 * algorithm (Jonker & Volgenant, *Computing* 1987). See ARCHITECTURE.md §5.4.
 *
 * Inputs are row-major Float64Array cost matrices. Forbidden cells are
 * encoded as `Number.POSITIVE_INFINITY` and are never selected; if every
 * augmenting path from some row goes through a forbidden cell, that row
 * remains unmatched and the solver proceeds with the remaining rows.
 */

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
export function solveLsap(_cost: Float64Array, _m: number, _n: number): LsapResult {
  throw new Error('solveLsap: not implemented');
}
