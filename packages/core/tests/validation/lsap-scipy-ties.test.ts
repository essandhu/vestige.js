/**
 * Cross-implementation validation: {@link solveLsap} must reproduce scipy's
 * `linear_sum_assignment` EXACTLY, including its choice among equally-optimal
 * assignments on tied matrices.
 *
 * On a matrix with a unique optimum every correct LSAP solver agrees, so a test
 * built from distinct costs proves nothing about tie-breaking. This fixture is
 * therefore built entirely from tie-dense matrices (costs drawn from {0, 1, 2}),
 * where the answer is a CHOICE.
 *
 * That choice is load-bearing: `solveLsap` backs the eval metrics' internal
 * gt↔prediction matching, TrackEval matches with scipy, and IDF1 assigns over
 * INTEGER match counts where exact ties are routine — so the tie rule moves
 * IDTP/IDFP/IDFN and hence the published number. vestige originally filled the
 * solver's `remaining` array forward where scipy fills it in reverse, inverting
 * scipy's choice on every tie. See ADR-0006.
 *
 * The tracker fixtures cannot catch this: their boxes are well-separated, so every
 * IoU is distinct and no tie ever arises. That is exactly why it went unnoticed.
 */
import { describe, expect, it } from 'vitest';
import data from '../../fixtures/lsap-scipy-ties/data.json' with { type: 'json' };
import { solveLsap } from '../../src/solvers/hungarian.js';

interface Case {
  m: number;
  n: number;
  cost: number[];
  scipy_row_to_col: number[];
}

interface Envelope {
  generator: { scipy: string; seed: number };
  cases: Case[];
}

const envelope = data as unknown as Envelope;

describe('solveLsap — scipy tie-breaking parity', () => {
  it(`reproduces scipy ${envelope.generator.scipy} on all ${envelope.cases.length} tie-dense cases`, () => {
    const mismatches: string[] = [];

    for (let k = 0; k < envelope.cases.length; k++) {
      const c = envelope.cases[k]!;
      const got = Array.from(solveLsap(Float64Array.from(c.cost), c.m, c.n).rowToCol);
      if (JSON.stringify(got) !== JSON.stringify(c.scipy_row_to_col)) {
        mismatches.push(
          `case ${k} (${c.m}×${c.n}) cost=[${c.cost.join(',')}]: vestige=[${got.join(',')}] scipy=[${c.scipy_row_to_col.join(',')}]`,
        );
      }
    }

    expect(
      mismatches.length,
      `${mismatches.length}/${envelope.cases.length} cases disagree with scipy:\n  ${mismatches.slice(0, 5).join('\n  ')}`,
    ).toBe(0);
  });

  it('still finds a minimum-cost assignment (ties are a choice, not a licence to be wrong)', () => {
    // Tie parity is worthless if we matched scipy by being wrong in the same way.
    // Independently assert optimality: vestige's total must equal scipy's total.
    for (const c of envelope.cases) {
      const { rowToCol } = solveLsap(Float64Array.from(c.cost), c.m, c.n);
      let ours = 0;
      let theirs = 0;
      for (let i = 0; i < c.m; i++) {
        const a = rowToCol[i]!;
        if (a !== -1) ours += c.cost[i * c.n + a]!;
        const b = c.scipy_row_to_col[i]!;
        if (b !== -1) theirs += c.cost[i * c.n + b]!;
      }
      expect(ours).toBeCloseTo(theirs, 12);
    }
  });
});
