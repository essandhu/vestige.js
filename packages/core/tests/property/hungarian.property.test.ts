import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { solveLsap } from '../../src/solvers/hungarian.js';

/**
 * Brute-force optimum of an m × n linear sum assignment problem for the
 * property tests. Enumerates all (max m, n)! / (max − min)! ordered selections
 * of size min(m, n). At our property-test bound of m, n ≤ 4, that's at most
 * 4! = 24 enumerations per case.
 *
 * Assumes the cost matrix has no forbidden cells (no `+Infinity`).
 */
function bruteForceOptimum(cost: readonly number[], m: number, n: number): number {
  if (m === 0 || n === 0) return 0;
  const rows = Math.min(m, n);
  const cols = Math.max(m, n);
  const transposed = m > n;
  const cell = transposed
    ? (i: number, j: number) => cost[j * n + i]!
    : (i: number, j: number) => cost[i * n + j]!;

  let best = Number.POSITIVE_INFINITY;
  const used = new Array<boolean>(cols).fill(false);
  function rec(row: number, partial: number): void {
    if (row === rows) {
      if (partial < best) best = partial;
      return;
    }
    for (let j = 0; j < cols; j++) {
      if (used[j]) continue;
      used[j] = true;
      rec(row + 1, partial + cell(row, j));
      used[j] = false;
    }
  }
  rec(0, 0);
  return best;
}

const finiteMatrix = fc
  .tuple(fc.integer({ min: 1, max: 4 }), fc.integer({ min: 1, max: 4 }))
  .chain(([m, n]) =>
    fc
      .array(fc.float({ min: -10, max: 10, noNaN: true, noDefaultInfinity: true }), {
        minLength: m * n,
        maxLength: m * n,
      })
      .map((entries) => ({ cost: new Float64Array(entries), m, n })),
  );

describe('solveLsap invariants (finite cost matrices)', () => {
  it('rowToCol and colToRow are mutual inverses', () => {
    fc.assert(
      fc.property(finiteMatrix, ({ cost, m, n }) => {
        const r = solveLsap(cost, m, n);
        for (let i = 0; i < m; i++) {
          const j = r.rowToCol[i]!;
          if (j !== -1) expect(r.colToRow[j]).toBe(i);
        }
        for (let j = 0; j < n; j++) {
          const i = r.colToRow[j]!;
          if (i !== -1) expect(r.rowToCol[i]).toBe(j);
        }
      }),
    );
  });

  it('returns exactly min(m, n) matches when no cells are forbidden', () => {
    fc.assert(
      fc.property(finiteMatrix, ({ cost, m, n }) => {
        const r = solveLsap(cost, m, n);
        let matched = 0;
        for (let i = 0; i < m; i++) if (r.rowToCol[i] !== -1) matched++;
        expect(matched).toBe(Math.min(m, n));
      }),
    );
  });

  it('totalCost equals the sum of matched cell costs', () => {
    fc.assert(
      fc.property(finiteMatrix, ({ cost, m, n }) => {
        const r = solveLsap(cost, m, n);
        let s = 0;
        for (let i = 0; i < m; i++) {
          const j = r.rowToCol[i]!;
          if (j !== -1) s += cost[i * n + j]!;
        }
        expect(r.totalCost).toBeCloseTo(s, 9);
      }),
    );
  });

  it('matches the brute-force optimum within float tolerance', () => {
    fc.assert(
      fc.property(finiteMatrix, ({ cost, m, n }) => {
        const r = solveLsap(cost, m, n);
        const opt = bruteForceOptimum(Array.from(cost), m, n);
        expect(r.totalCost).toBeCloseTo(opt, 9);
      }),
    );
  });

  it('is deterministic — same input twice yields the same matching', () => {
    fc.assert(
      fc.property(finiteMatrix, ({ cost, m, n }) => {
        const r1 = solveLsap(new Float64Array(cost), m, n);
        const r2 = solveLsap(new Float64Array(cost), m, n);
        expect(Array.from(r1.rowToCol)).toEqual(Array.from(r2.rowToCol));
        expect(Array.from(r1.colToRow)).toEqual(Array.from(r2.colToRow));
        expect(r1.totalCost).toBe(r2.totalCost);
      }),
    );
  });
});
