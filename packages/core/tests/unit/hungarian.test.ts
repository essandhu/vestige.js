import { describe, expect, it } from 'vitest';
import { solveLsap } from '../../src/solvers/hungarian.js';

const INF = Number.POSITIVE_INFINITY;

function f(...xs: number[]): Float64Array {
  return new Float64Array(xs);
}

function arr(x: Int32Array): number[] {
  return Array.from(x);
}

describe('solveLsap — trivial cases', () => {
  it('empty matrix (m=0, n=0)', () => {
    const r = solveLsap(f(), 0, 0);
    expect(arr(r.rowToCol)).toEqual([]);
    expect(arr(r.colToRow)).toEqual([]);
    expect(r.totalCost).toBe(0);
  });

  it('no rows (m=0, n>0): all columns unmatched, zero cost', () => {
    const r = solveLsap(f(), 0, 3);
    expect(arr(r.rowToCol)).toEqual([]);
    expect(arr(r.colToRow)).toEqual([-1, -1, -1]);
    expect(r.totalCost).toBe(0);
  });

  it('no columns (m>0, n=0): all rows unmatched, zero cost', () => {
    const r = solveLsap(f(), 3, 0);
    expect(arr(r.rowToCol)).toEqual([-1, -1, -1]);
    expect(arr(r.colToRow)).toEqual([]);
    expect(r.totalCost).toBe(0);
  });

  it('1×1 matches the only pair', () => {
    const r = solveLsap(f(5), 1, 1);
    expect(arr(r.rowToCol)).toEqual([0]);
    expect(arr(r.colToRow)).toEqual([0]);
    expect(r.totalCost).toBe(5);
  });
});

describe('solveLsap — square cases (unique optimum)', () => {
  it('2×2 identity-pick is optimum', () => {
    // cost = [[1, 10], [10, 1]] — diagonal wins, total = 2
    const r = solveLsap(f(1, 10, 10, 1), 2, 2);
    expect(arr(r.rowToCol)).toEqual([0, 1]);
    expect(arr(r.colToRow)).toEqual([0, 1]);
    expect(r.totalCost).toBeCloseTo(2, 12);
  });

  it('2×2 swap-pick is optimum', () => {
    // cost = [[10, 1], [1, 10]] — anti-diagonal wins, total = 2
    const r = solveLsap(f(10, 1, 1, 10), 2, 2);
    expect(arr(r.rowToCol)).toEqual([1, 0]);
    expect(arr(r.colToRow)).toEqual([1, 0]);
    expect(r.totalCost).toBeCloseTo(2, 12);
  });

  it('3×3 classical example', () => {
    // cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    // Six assignments evaluated by hand; unique optimum is
    // (0→1) + (1→0) + (2→2) = 1 + 2 + 2 = 5.
    const r = solveLsap(f(4, 1, 3, 2, 0, 5, 3, 2, 2), 3, 3);
    expect(arr(r.rowToCol)).toEqual([1, 0, 2]);
    expect(arr(r.colToRow)).toEqual([1, 0, 2]);
    expect(r.totalCost).toBeCloseTo(5, 12);
  });

  it('handles negative costs (rewards) symmetrically', () => {
    // cost = [[-3, -1], [-2, -4]] — optimum is the diagonal at -7.
    const r = solveLsap(f(-3, -1, -2, -4), 2, 2);
    expect(arr(r.rowToCol)).toEqual([0, 1]);
    expect(r.totalCost).toBeCloseTo(-7, 12);
  });
});

describe('solveLsap — rectangular cases', () => {
  it('m<n (2×3) leaves exactly one column unmatched', () => {
    // cost = [[8, 1, 3], [1, 3, 7]] — optimum (0→1)+(1→0) = 2, col 2 unused
    const r = solveLsap(f(8, 1, 3, 1, 3, 7), 2, 3);
    expect(arr(r.rowToCol)).toEqual([1, 0]);
    expect(arr(r.colToRow)).toEqual([1, 0, -1]);
    expect(r.totalCost).toBeCloseTo(2, 12);
  });

  it('m>n (3×2) leaves exactly one row unmatched', () => {
    // cost = [[8, 1], [1, 3], [3, 7]] — optimum (0→1)+(1→0) = 2, row 2 unused
    const r = solveLsap(f(8, 1, 1, 3, 3, 7), 3, 2);
    expect(arr(r.rowToCol)).toEqual([1, 0, -1]);
    expect(arr(r.colToRow)).toEqual([1, 0]);
    expect(r.totalCost).toBeCloseTo(2, 12);
  });

  it('single row (1×3) picks the cheapest column', () => {
    const r = solveLsap(f(5, 1, 3), 1, 3);
    expect(arr(r.rowToCol)).toEqual([1]);
    expect(arr(r.colToRow)).toEqual([-1, 0, -1]);
    expect(r.totalCost).toBeCloseTo(1, 12);
  });

  it('single column (3×1) picks the cheapest row', () => {
    const r = solveLsap(f(5, 1, 3), 3, 1);
    expect(arr(r.rowToCol)).toEqual([-1, 0, -1]);
    expect(arr(r.colToRow)).toEqual([1]);
    expect(r.totalCost).toBeCloseTo(1, 12);
  });
});

describe('solveLsap — forbidden cells (+Infinity)', () => {
  it('respects a single forbidden cell, forcing the alternative', () => {
    // cost = [[1, INF], [1, 1]] — row 0 must take col 0; row 1 takes col 1
    const r = solveLsap(f(1, INF, 1, 1), 2, 2);
    expect(arr(r.rowToCol)).toEqual([0, 1]);
    expect(arr(r.colToRow)).toEqual([0, 1]);
    expect(r.totalCost).toBeCloseTo(2, 12);
  });

  it('forbidden cells force an anti-diagonal matching', () => {
    // cost = [[INF, 1], [1, INF]] — swap is the only feasible matching
    const r = solveLsap(f(INF, 1, 1, INF), 2, 2);
    expect(arr(r.rowToCol)).toEqual([1, 0]);
    expect(arr(r.colToRow)).toEqual([1, 0]);
    expect(r.totalCost).toBeCloseTo(2, 12);
  });

  it('leaves a row unmatched when every column is forbidden for it', () => {
    // cost = [[INF, INF], [1, 2]] — row 0 has no feasible column; row 1 takes col 0
    const r = solveLsap(f(INF, INF, 1, 2), 2, 2);
    expect(arr(r.rowToCol)).toEqual([-1, 0]);
    expect(arr(r.colToRow)).toEqual([1, -1]);
    expect(r.totalCost).toBeCloseTo(1, 12);
  });

  it('totalCost never includes a forbidden cell', () => {
    // Diagonal-forced 3×3 — only feasible matching is the identity.
    const r = solveLsap(f(1, INF, INF, INF, 1, INF, INF, INF, 1), 3, 3);
    expect(arr(r.rowToCol)).toEqual([0, 1, 2]);
    expect(r.totalCost).toBeCloseTo(3, 12);
    expect(Number.isFinite(r.totalCost)).toBe(true);
  });
});

describe('solveLsap — determinism', () => {
  it('is bit-for-bit reproducible on the same input', () => {
    // All-ones tie-case: any matching is optimum; pin the exact answer
    // across repeated calls (snapshot-test viability — ARCHITECTURE.md §2.5).
    const cost = f(1, 1, 1, 1);
    const r1 = solveLsap(cost.slice(), 2, 2);
    const r2 = solveLsap(cost.slice(), 2, 2);
    expect(arr(r1.rowToCol)).toEqual(arr(r2.rowToCol));
    expect(arr(r1.colToRow)).toEqual(arr(r2.colToRow));
    expect(r1.totalCost).toBe(r2.totalCost);
  });
});

describe('solveLsap — does not mutate input', () => {
  it('leaves the cost matrix unchanged after solving', () => {
    const cost = f(4, 1, 3, 2, 0, 5, 3, 2, 2);
    const snapshot = Array.from(cost);
    solveLsap(cost, 3, 3);
    expect(Array.from(cost)).toEqual(snapshot);
  });
});
