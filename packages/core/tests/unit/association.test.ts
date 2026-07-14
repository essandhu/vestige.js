/**
 * Unit tests for the two reference association conventions (ADR-0005).
 *
 * The `solveWithRejectionCost` oracles are not hand-invented: they are the actual
 * return values of `lap.lapjv(cost, extend_cost=True, cost_limit=limit)` — the call
 * `yolox/tracker/matching.py:linear_assignment` makes — recorded by running it
 * against the pinned reference. Each case is annotated with what it pins down.
 *
 * The `solveThenFilter` oracles are hand-verifiable: the matrices are 2×2, and the
 * two candidate assignments' totals are written out in the comments.
 */
import { describe, expect, it } from 'vitest';
import { solveThenFilter, solveWithRejectionCost } from '../../src/solvers/association.js';

const f64 = (rows: number[][]): Float64Array => Float64Array.from(rows.flat());
const arr = (a: Int32Array): number[] => Array.from(a);

describe('solveWithRejectionCost — ByteTrack (lap.lapjv cost_limit)', () => {
  it('admits a cell whose cost is EXACTLY the reject cost (cost_limit is inclusive)', () => {
    // lapjv: matches [(0,0)] at cost 0.8. The 2.0 cells are forbidden.
    expect(
      arr(
        solveWithRejectionCost(
          f64([
            [0.8, 2.0],
            [2.0, 2.0],
          ]),
          2,
          2,
          0.8,
        ),
      ),
    ).toEqual([0, -1]);
  });

  it('leaves a row unmatched rather than displace a cheap match — no column penalty', () => {
    // THE case that proves the objective has no penalty for unmatched COLUMNS.
    //   match both:            (0→1)=0.75 + (1→0)=0.75            = 1.50
    //   match row 0, reject 1: (0→0)=0.10 + reject(0.80)          = 0.90  <- lapjv picks this
    // A column penalty would have made it 0.10+0.80+0.80 = 1.70 and flipped the answer.
    expect(
      arr(
        solveWithRejectionCost(
          f64([
            [0.1, 0.75],
            [0.75, 0.9],
          ]),
          2,
          2,
          0.8,
        ),
      ),
    ).toEqual([0, -1]);
  });

  it('rejects every row when all cells exceed the reject cost', () => {
    expect(arr(solveWithRejectionCost(f64([[0.9, 0.85]]), 1, 2, 0.8))).toEqual([-1]);
  });

  it('gives a contested column to the cheaper row and rejects the other', () => {
    expect(arr(solveWithRejectionCost(f64([[0.1], [0.2]]), 2, 1, 0.8))).toEqual([0, -1]);
  });

  it('does NOT maximize cardinality: it declines rather than take two mediocre cells', () => {
    // The counterexample from ADR-0005.
    //   match both:            (0→1)=0.79 + (1→0)=0.79   = 1.58   <- what a max-cardinality solver is forced into
    //   match row 0, reject 1: (0→0)=0.10 + reject(0.80) = 0.90   <- what lapjv actually does
    expect(
      arr(
        solveWithRejectionCost(
          f64([
            [0.1, 0.79],
            [0.79, 1.0],
          ]),
          2,
          2,
          0.8,
        ),
      ),
    ).toEqual([0, -1]);
  });

  it('still matches both rows when matching both is genuinely optimal', () => {
    //   match both:            0.10 + 0.50              = 0.60  <- optimal
    //   match row 0, reject 1: 0.10 + reject(0.80)      = 0.90
    // Guards against "fixing" the bug by simply never matching aggressively.
    expect(
      arr(
        solveWithRejectionCost(
          f64([
            [0.1, 0.79],
            [0.79, 0.5],
          ]),
          2,
          2,
          0.8,
        ),
      ),
    ).toEqual([0, 1]);
  });

  it('minimizes the reference objective exactly (brute-forced over all matchings)', () => {
    // Independent check of the dummy-column reduction: for every small matrix, the
    // returned assignment must attain the true minimum of
    //     Σ matched cost  +  rejectCost × (# unmatched rows),
    // with cells above rejectCost forbidden. Brute-force every partial matching.
    const REJECT = 0.6;
    let seed = 12345;
    const rand = (): number => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };

    for (let trial = 0; trial < 300; trial++) {
      const m = 1 + Math.floor(rand() * 3);
      const n = 1 + Math.floor(rand() * 3);
      const cost = new Float64Array(m * n);
      for (let k = 0; k < m * n; k++) cost[k] = Math.round(rand() * 100) / 100;

      const objective = (rowToCol: number[]): number => {
        let total = 0;
        for (let i = 0; i < m; i++) {
          const j = rowToCol[i]!;
          if (j === -1) total += REJECT;
          else total += cost[i * n + j]!;
        }
        return total;
      };

      // Brute force: every assignment of rows to distinct admissible columns or -1.
      let best = Number.POSITIVE_INFINITY;
      const current: number[] = new Array(m).fill(-1);
      const used = new Set<number>();
      const recurse = (i: number): void => {
        if (i === m) {
          best = Math.min(best, objective(current));
          return;
        }
        current[i] = -1;
        recurse(i + 1);
        for (let j = 0; j < n; j++) {
          if (used.has(j) || cost[i * n + j]! > REJECT) continue;
          used.add(j);
          current[i] = j;
          recurse(i + 1);
          used.delete(j);
          current[i] = -1;
        }
      };
      recurse(0);

      const got = arr(solveWithRejectionCost(cost, m, n, REJECT));
      // Sanity: no column reused, no forbidden cell selected.
      const seen = new Set<number>();
      for (let i = 0; i < m; i++) {
        const j = got[i]!;
        if (j === -1) continue;
        expect(seen.has(j)).toBe(false);
        seen.add(j);
        expect(cost[i * n + j]!).toBeLessThanOrEqual(REJECT);
      }
      expect(objective(got)).toBeCloseTo(best, 9);
    }
  });
});

describe('solveThenFilter — SORT / OC-SORT (solve the full matrix, then drop)', () => {
  it('drops the junk pair instead of letting it displace the strong one', () => {
    // The discriminating structure from the crowded fixture, as a bare matrix.
    // IoU, threshold 0.3:
    //   solve (maximize, FULL matrix): (0→0)+(1→1) = 1.0000 + 0.0909 = 1.0909  <- wins
    //                                  (0→1)+(1→0) = 0.4118 + 0.4118 = 0.8236
    //   then filter: (1,1) scores 0.0909 < 0.3 -> dropped.
    // A gate-then-solve implementation instead forbids (1,1), which leaves the
    // anti-diagonal as the only 2-match and hands row 0 the WRONG column.
    const iou = f64([
      [1.0, 0.4118],
      [0.4118, 0.0909],
    ]);
    expect(arr(solveThenFilter(iou, iou, 2, 2, 0.3, true))).toEqual([0, -1]);
  });

  it('takes the unambiguous-matching shortcut when the mask is already a matching', () => {
    // mask (score > 0.3) = [[1,0],[0,1]] -> one per row, one per column -> used directly.
    const iou = f64([
      [0.5, 0.1],
      [0.1, 0.4],
    ]);
    expect(arr(solveThenFilter(iou, iou, 2, 2, 0.3, true))).toEqual([0, 1]);
  });

  it('produces no matches when every cell is below threshold', () => {
    const iou = f64([
      [0.1, 0.2],
      [0.2, 0.1],
    ]);
    expect(arr(solveThenFilter(iou, iou, 2, 2, 0.3, true))).toEqual([-1, -1]);
  });

  it('solves on solveScore but filters on filterScore (OC-SORT primary stage)', () => {
    // OC-SORT solves on `iou + angle_diff_cost` but thresholds on RAW iou.
    //   solveScore: (0→0)+(1→1) = 0.5 + 0.9 = 1.4  <- the angle bonus wins (1,1)
    //               (0→1)+(1→0) = 0.35 + 0.35 = 0.7
    //   filter on raw iou: iou[1][1] = 0.0 < 0.3 -> dropped despite winning the solve.
    // If the filter mistakenly used solveScore, (1,1) would survive at 0.9.
    const solve = f64([
      [0.5, 0.35],
      [0.35, 0.9],
    ]);
    const iou = f64([
      [0.5, 0.35],
      [0.35, 0.0],
    ]);
    expect(arr(solveThenFilter(solve, iou, 2, 2, 0.3, true))).toEqual([0, -1]);
  });

  it('keeps a pair scoring EXACTLY the threshold (filter is `< threshold` -> drop)', () => {
    const iou = f64([[0.3, 0.1]]);
    expect(arr(solveThenFilter(iou, iou, 1, 2, 0.3, true))).toEqual([0]);
  });

  it('requireAnyAboveThreshold rejects an exactly-on-threshold maximum (OC-SORT BYTE/OCR guard)', () => {
    // The reference guards those stages with `if iou_left.max() > iou_threshold:`
    // — STRICTLY above. A maximum sitting exactly on the threshold produces no
    // matches at all, even though the post-filter alone would have kept it (above).
    const iou = f64([[0.3, 0.1]]);
    expect(arr(solveThenFilter(iou, iou, 1, 2, 0.3, false, true))).toEqual([-1]);
  });
});
