/**
 * Cross-implementation validation of the METRICS themselves against
 * `JonathonLuiten/TrackEval` — the canonical implementation of HOTA, CLEAR-MOT and
 * Identity, and the one every published MOTChallenge number is computed with.
 *
 * Until this landed, `packages/eval/src/metrics/` had no oracle at all: the unit
 * tests' expected values were hand-traced from the papers. That is the wrong shape of
 * assurance for this code. These metrics are the instrument we measure the trackers
 * WITH — a subtle error in HOTA's AssA denominators or CLEAR's ID-switch bookkeeping
 * doesn't announce itself, it just quietly makes every published number wrong, and it
 * looks like a *tracker* bug. The instrument has to be calibrated against the
 * reference, not against our own reading of the definition.
 *
 * It caught two live bugs immediately (ADR-0006):
 *
 *  - `clearmot.ts` committed its continuity memory outside the "did this frame match
 *    anything" guard, so any empty frame wiped it — inventing an ID switch and a Frag
 *    after every detector dropout. Pinned by `empty-tracker-frame` / `empty-gt-frame`.
 *  - `solveLsap` broke assignment ties the opposite way from scipy, which is what
 *    TrackEval matches with. Pinned by `tied-boxes`.
 *
 * See `packages/eval/fixtures/trackeval-metrics/gen.py` for how each scenario is
 * built and what it targets.
 */
import { describe, expect, it } from 'vitest';
import data from '../../fixtures/trackeval-metrics/data.json' with { type: 'json' };
import { combineClearMot, combineHota, combineIdentity } from '../../src/aggregate.js';
import type { BBox } from '../../src/core.js';
import type { EvalFrame } from '../../src/metrics/frames.js';
import { clearMot, hota, identity } from '../../src/metrics/index.js';

interface FixtureFrame {
  gt: Array<{ id: number; bbox: number[] }>;
  track: Array<{ id: number; bbox: number[] }>;
}

interface Scenario {
  name: string;
  frames: FixtureFrame[];
  expected: {
    clear: {
      tp: number;
      fp: number;
      fn: number;
      idsw: number;
      mota: number;
      motp: number;
      mt: number;
      pt: number;
      ml: number;
      frag: number;
    };
    identity: { idtp: number; idfp: number; idfn: number; idf1: number; idp: number; idr: number };
    hota: {
      hota: number;
      deta: number;
      assa: number;
      loca: number;
      hota_per_alpha: number[];
      deta_per_alpha: number[];
      assa_per_alpha: number[];
    };
  };
}

interface Combined {
  clear: Scenario['expected']['clear'];
  identity: Scenario['expected']['identity'];
  hota: Scenario['expected']['hota'];
  naive_mean_mota: number;
  naive_mean_idf1: number;
  naive_mean_hota: number;
}

interface Envelope {
  generator: { trackeval_sha: string };
  threshold: number;
  scenarios: Scenario[];
  combined: Combined;
}

const envelope = data as unknown as Envelope;

// Counts are compared exactly; rates are compared to 9 decimal places. TrackEval
// computes in float64 and we compare against values round-tripped through JSON, so
// exact equality is not the right bar — but anything looser would start hiding real
// divergences. HOTA's alpha integral accumulates 19 terms and still lands well inside
// this.

function toEvalFrames(frames: FixtureFrame[]): EvalFrame[] {
  return frames.map((f) => ({
    gtIds: f.gt.map((g) => g.id),
    gtBoxes: f.gt.map((g) => g.bbox as unknown as BBox),
    trackIds: f.track.map((t) => t.id),
    trackBoxes: f.track.map((t) => t.bbox as unknown as BBox),
  }));
}

describe(`metrics — TrackEval cross-implementation (sha ${envelope.generator.trackeval_sha.slice(0, 8)})`, () => {
  for (const scenario of envelope.scenarios) {
    describe(scenario.name, () => {
      const frames = toEvalFrames(scenario.frames);
      const opts = { simThreshold: envelope.threshold };

      it('CLEAR-MOT matches TrackEval', () => {
        const got = clearMot(frames, opts);
        const want = scenario.expected.clear;
        // Counts first: they are integers, so a mismatch is unambiguous and reads
        // better in the failure output than a MOTA delta four decimals down.
        expect({ tp: got.tp, fp: got.fp, fn: got.fn, idsw: got.idsw, frag: got.frag }).toEqual({
          tp: want.tp,
          fp: want.fp,
          fn: want.fn,
          idsw: want.idsw,
          frag: want.frag,
        });
        expect({ mt: got.mt, pt: got.pt, ml: got.ml }).toEqual({
          mt: want.mt,
          pt: want.pt,
          ml: want.ml,
        });
        expect(got.mota).toBeCloseTo(want.mota, 9);
        expect(got.motp).toBeCloseTo(want.motp, 9);
      });

      it('Identity (IDF1) matches TrackEval', () => {
        const got = identity(frames, opts);
        const want = scenario.expected.identity;
        expect({ idtp: got.idtp, idfp: got.idfp, idfn: got.idfn }).toEqual({
          idtp: want.idtp,
          idfp: want.idfp,
          idfn: want.idfn,
        });
        expect(got.idf1).toBeCloseTo(want.idf1, 9);
        expect(got.idp).toBeCloseTo(want.idp, 9);
        expect(got.idr).toBeCloseTo(want.idr, 9);
      });

      it('HOTA matches TrackEval at every alpha', () => {
        const got = hota(frames);
        const want = scenario.expected.hota;

        // Per-alpha first — the aggregate is a mean over these, so it can mask a
        // compensating error in the AssA/DetA split at individual alphas.
        for (let k = 0; k < want.hota_per_alpha.length; k++) {
          expect(
            got.hotaPerAlpha[k],
            `HOTA at alpha ${(k + 1) / 20} (scenario ${scenario.name})`,
          ).toBeCloseTo(want.hota_per_alpha[k]!, 9);
          expect(
            got.detaPerAlpha[k],
            `DetA at alpha ${(k + 1) / 20} (scenario ${scenario.name})`,
          ).toBeCloseTo(want.deta_per_alpha[k]!, 9);
          expect(
            got.assaPerAlpha[k],
            `AssA at alpha ${(k + 1) / 20} (scenario ${scenario.name})`,
          ).toBeCloseTo(want.assa_per_alpha[k]!, 9);
        }

        expect(got.hota).toBeCloseTo(want.hota, 9);
        expect(got.deta).toBeCloseTo(want.deta, 9);
        expect(got.assa).toBeCloseTo(want.assa, 9);
        expect(got.locA).toBeCloseTo(want.loca, 9);
      });
    });
  }
});

/**
 * Cross-sequence AGGREGATION — the step that turns per-sequence results into the single
 * number a benchmark actually publishes (MOT17 is seven sequences; MOT20 is four).
 *
 * Before this, the eval package had no aggregation at all, so there was no way to produce
 * an MOT17 headline figure — and the obvious thing anyone would reach for, averaging the
 * per-sequence values, is WRONG. TrackEval sums the raw counts and recomputes the metric
 * once from the totals; `_compute_final_fields` is literally the same code for a single
 * sequence and for the combination. See ADR-0006 and `src/aggregate.ts`.
 *
 * The oracles here are TrackEval's own `combine_sequences` over the nine scenarios above,
 * treated as nine sequences of one benchmark.
 */
describe('cross-sequence aggregation — TrackEval combine_sequences', () => {
  const allFrames = envelope.scenarios.map((s) => toEvalFrames(s.frames));
  const opts = { simThreshold: envelope.threshold };
  const want = envelope.combined;

  it('CLEAR-MOT combines by summing counts, not averaging rates', () => {
    const got = combineClearMot(allFrames.map((f) => clearMot(f, opts)));
    expect({ tp: got.tp, fp: got.fp, fn: got.fn, idsw: got.idsw, frag: got.frag }).toEqual({
      tp: want.clear.tp,
      fp: want.clear.fp,
      fn: want.clear.fn,
      idsw: want.clear.idsw,
      frag: want.clear.frag,
    });
    expect({ mt: got.mt, pt: got.pt, ml: got.ml }).toEqual({
      mt: want.clear.mt,
      pt: want.clear.pt,
      ml: want.clear.ml,
    });
    expect(got.mota).toBeCloseTo(want.clear.mota, 9);
    // MOTP is Σmotp_sum / Σtp — NOT a mean of the per-sequence MOTPs.
    expect(got.motp).toBeCloseTo(want.clear.motp, 9);
  });

  it('Identity combines by summing IDTP/IDFP/IDFN', () => {
    const got = combineIdentity(allFrames.map((f) => identity(f, opts)));
    expect({ idtp: got.idtp, idfp: got.idfp, idfn: got.idfn }).toEqual({
      idtp: want.identity.idtp,
      idfp: want.identity.idfp,
      idfn: want.identity.idfn,
    });
    expect(got.idf1).toBeCloseTo(want.identity.idf1, 9);
    expect(got.idp).toBeCloseTo(want.identity.idp, 9);
    expect(got.idr).toBeCloseTo(want.identity.idr, 9);
  });

  it('HOTA combines by summing counts per alpha and TP-WEIGHTING AssA / LocA', () => {
    const got = combineHota(allFrames.map((f) => hota(f)));

    // Per-alpha first: the aggregate is a mean over these, so a compensating error in the
    // DetA/AssA split at individual alphas can cancel out in the scalar.
    for (let k = 0; k < want.hota.hota_per_alpha.length; k++) {
      expect(got.hotaPerAlpha[k], `combined HOTA at alpha ${(k + 1) / 20}`).toBeCloseTo(
        want.hota.hota_per_alpha[k]!,
        9,
      );
      expect(got.detaPerAlpha[k], `combined DetA at alpha ${(k + 1) / 20}`).toBeCloseTo(
        want.hota.deta_per_alpha[k]!,
        9,
      );
      expect(got.assaPerAlpha[k], `combined AssA at alpha ${(k + 1) / 20}`).toBeCloseTo(
        want.hota.assa_per_alpha[k]!,
        9,
      );
    }

    expect(got.hota).toBeCloseTo(want.hota.hota, 9);
    expect(got.deta).toBeCloseTo(want.hota.deta, 9);
    expect(got.assa).toBeCloseTo(want.hota.assa, 9);
    expect(got.locA).toBeCloseTo(want.hota.loca, 9);
  });

  it('is NOT the mean of the per-sequence values (the trap this exists to avoid)', () => {
    // Guard against a "passing" aggregation that is secretly a mean. If the scenarios ever
    // drift into a shape where the two coincide, this fixture silently stops discriminating
    // and would go green for an implementation that just averages. Assert they differ by a
    // margin far larger than any floating-point slack — and by a margin that MATTERS: in
    // MOT terms these gaps (MOTA ~5 points, HOTA ~2 points) are the difference between
    // papers.
    const gotMota = combineClearMot(allFrames.map((f) => clearMot(f, opts))).mota;
    const gotHota = combineHota(allFrames.map((f) => hota(f))).hota;
    const gotIdf1 = combineIdentity(allFrames.map((f) => identity(f, opts))).idf1;

    expect(Math.abs(gotMota - want.naive_mean_mota)).toBeGreaterThan(0.01);
    expect(Math.abs(gotHota - want.naive_mean_hota)).toBeGreaterThan(0.01);
    expect(Math.abs(gotIdf1 - want.naive_mean_idf1)).toBeGreaterThan(0.005);
  });

  it('refuses to combine an empty list rather than silently reporting zero', () => {
    // A zeroed aggregate reads as a catastrophically bad tracker, not as a bug.
    expect(() => combineClearMot([])).toThrow(/empty/i);
    expect(() => combineIdentity([])).toThrow(/empty/i);
    expect(() => combineHota([])).toThrow(/empty/i);
  });
});
