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

interface Envelope {
  generator: { trackeval_sha: string };
  threshold: number;
  scenarios: Scenario[];
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
