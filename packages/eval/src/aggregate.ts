/**
 * Combine per-sequence metric results into the single number a benchmark publishes.
 *
 * MOT17 is seven training sequences (fourteen for test); MOT20 is four. The headline
 * HOTA / MOTA / IDF1 for a tracker is **one** number covering all of them, and getting it
 * from the per-sequence results is not a mean.
 *
 * The obvious thing — average the per-sequence HOTAs — is **wrong**, and wrong in a
 * direction that flatters short sequences: MOT17-05 has 837 frames and MOT17-04 has 1050
 * with far more crowding, and a plain mean gives them equal say. TrackEval instead sums
 * the raw COUNTS across sequences and recomputes the metric once from the totals
 * (`_base_metric.py:_combine_sum`, `_combine_weighted_av`). Its `_compute_final_fields`
 * carries the comment
 *
 *   > "This function is used for both per-sequence calculation, and in combining values
 *   >  across sequences."
 *
 * — i.e. the recompute is literally the same code in both cases. That invariant is what
 * these functions mirror: **sum the extensive quantities, then re-derive the rates.**
 *
 * Which quantities are extensive differs per metric, and the details are exactly where a
 * plausible-looking implementation goes wrong:
 *
 *   * CLEAR sums TP/FN/FP/IDSW/MT/PT/ML/Frag **and `MOTP_sum`** — MOTP is
 *     `Σ motp_sum / Σ tp`, never a mean of per-sequence MOTPs.
 *   * Identity sums IDTP/IDFN/IDFP and re-derives IDF1.
 *   * HOTA sums TP/FN/FP **per alpha**, but AssA / AssRe / AssPr / LocA are
 *     **TP-weighted averages**, not sums and not plain means. DetA then comes from the
 *     summed counts and `HOTA = √(DetA · AssA)` is recomputed — so the combined HOTA is
 *     NOT any average of the per-sequence HOTAs.
 *
 * Validated against TrackEval's own `combine_sequences` in
 * `tests/validation/trackeval-metrics-fixture.test.ts`. See ADR-0006.
 */

import type { ClearMotResult } from './metrics/clearmot.js';
import type { HotaResult } from './metrics/hota.js';
import type { IdentityResult } from './metrics/identity.js';

/** TrackEval guards every rate denominator with `np.maximum(1.0, …)`; mirror it exactly. */
function safeDiv(numerator: number, denominator: number): number {
  return numerator / Math.max(1, denominator);
}

/**
 * Combine per-sequence CLEAR-MOT results (`clear.py:combine_sequences`).
 *
 * Sums the count fields — including `motpSum`, which is why {@link ClearMotResult}
 * exposes it — then re-derives MOTA and MOTP from the totals.
 *
 * Note MT/PT/ML/Frag are **summed**, not recomputed: they are per-gt-identity tallies, and
 * an identity belongs to exactly one sequence, so the totals are simply the counts across
 * the benchmark.
 *
 * @throws if `results` is empty — an aggregate over nothing has no meaningful MOTA, and
 *   silently returning 0 would look like a catastrophically bad tracker rather than a bug.
 */
export function combineClearMot(results: ReadonlyArray<ClearMotResult>): ClearMotResult {
  if (results.length === 0) {
    throw new Error('combineClearMot: cannot combine an empty list of sequence results');
  }

  let tp = 0;
  let fn = 0;
  let fp = 0;
  let idsw = 0;
  let mt = 0;
  let pt = 0;
  let ml = 0;
  let frag = 0;
  let motpSum = 0;
  let numGtDets = 0;
  let numGtIds = 0;

  for (const r of results) {
    tp += r.tp;
    fn += r.fn;
    fp += r.fp;
    idsw += r.idsw;
    mt += r.mt;
    pt += r.pt;
    ml += r.ml;
    frag += r.frag;
    motpSum += r.motpSum;
    numGtDets += r.numGtDets;
    numGtIds += r.numGtIds;
  }

  return {
    // TrackEval: (TP − FP − IDSW) / max(1, TP + FN). `TP + FN` is the gt detection total.
    mota: safeDiv(tp - fp - idsw, tp + fn),
    motp: safeDiv(motpSum, tp),
    tp,
    fn,
    fp,
    idsw,
    frag,
    mt,
    pt,
    ml,
    numGtDets,
    numGtIds,
    motpSum,
  };
}

/**
 * Combine per-sequence Identity results (`identity.py:combine_sequences`).
 *
 * Sums IDTP / IDFN / IDFP and re-derives IDF1 / IDP / IDR. The per-sequence trajectory
 * assignment is NOT redone globally: an identity lives in one sequence, so the optimal
 * assignment is per-sequence and only the resulting counts combine.
 */
export function combineIdentity(results: ReadonlyArray<IdentityResult>): IdentityResult {
  if (results.length === 0) {
    throw new Error('combineIdentity: cannot combine an empty list of sequence results');
  }

  let idtp = 0;
  let idfn = 0;
  let idfp = 0;
  for (const r of results) {
    idtp += r.idtp;
    idfn += r.idfn;
    idfp += r.idfp;
  }

  return {
    idf1: safeDiv(idtp, idtp + 0.5 * idfp + 0.5 * idfn),
    idr: safeDiv(idtp, idtp + idfn),
    idp: safeDiv(idtp, idtp + idfp),
    idtp,
    idfn,
    idfp,
  };
}

/** LocA's denominator guard in TrackEval is 1e-10, not 1.0 — it is a similarity, not a count. */
const LOCA_EPS = 1e-10;

/**
 * Combine per-sequence HOTA results (`hota.py:combine_sequences`).
 *
 * The subtle one, and the one most likely to be quietly wrong. Per alpha:
 *
 * 1. **Sum** TP, FN, FP across sequences.
 * 2. **TP-weight** AssA, AssRe, AssPr — `Σ(value · tp) / max(1, Σtp)`. These are averages
 *    over true positives, so combining them means re-weighting by how many TPs each
 *    sequence contributed, not averaging the sequence-level rates.
 * 3. **TP-weight** LocA the same way (with a 1e-10 guard rather than 1.0, since it is a
 *    similarity rather than a count).
 * 4. Re-derive DetA / DetRe / DetPr from the summed counts, and `HOTA = √(DetA · AssA)`.
 *
 * The combined HOTA is therefore **not any average of the per-sequence HOTAs** — it is
 * recomputed from combined components. Averaging them gives a different number, and it is
 * not the one MOTChallenge publishes.
 */
export function combineHota(results: ReadonlyArray<HotaResult>): HotaResult {
  const [first] = results;
  if (first === undefined) {
    throw new Error('combineHota: cannot combine an empty list of sequence results');
  }
  const numAlphas = first.alphas.length;

  const tpPerAlpha = new Float64Array(numAlphas);
  const fnPerAlpha = new Float64Array(numAlphas);
  const fpPerAlpha = new Float64Array(numAlphas);
  const assaWeighted = new Float64Array(numAlphas);
  const assReWeighted = new Float64Array(numAlphas);
  const assPrWeighted = new Float64Array(numAlphas);
  const locAWeighted = new Float64Array(numAlphas);

  for (const r of results) {
    if (r.alphas.length !== numAlphas) {
      throw new Error(
        `combineHota: sequence results disagree on the alpha sweep (${r.alphas.length} vs ${numAlphas})`,
      );
    }
    for (let a = 0; a < numAlphas; a++) {
      const tp = r.tpPerAlpha[a] ?? 0;
      tpPerAlpha[a] = (tpPerAlpha[a] ?? 0) + tp;
      fnPerAlpha[a] = (fnPerAlpha[a] ?? 0) + (r.fnPerAlpha[a] ?? 0);
      fpPerAlpha[a] = (fpPerAlpha[a] ?? 0) + (r.fpPerAlpha[a] ?? 0);
      // Weight by THIS sequence's TP count at THIS alpha.
      assaWeighted[a] = (assaWeighted[a] ?? 0) + (r.assaPerAlpha[a] ?? 0) * tp;
      assReWeighted[a] = (assReWeighted[a] ?? 0) + (r.assRePerAlpha[a] ?? 0) * tp;
      assPrWeighted[a] = (assPrWeighted[a] ?? 0) + (r.assPrPerAlpha[a] ?? 0) * tp;
      locAWeighted[a] = (locAWeighted[a] ?? 0) + (r.locAPerAlpha[a] ?? 0) * tp;
    }
  }

  const hotaPerAlpha = new Float64Array(numAlphas);
  const detaPerAlpha = new Float64Array(numAlphas);
  const assaPerAlpha = new Float64Array(numAlphas);
  const locAPerAlpha = new Float64Array(numAlphas);
  const assRePerAlpha = new Float64Array(numAlphas);
  const assPrPerAlpha = new Float64Array(numAlphas);

  let detReSum = 0;
  let detPrSum = 0;
  let assReSum = 0;
  let assPrSum = 0;

  for (let a = 0; a < numAlphas; a++) {
    const tp = tpPerAlpha[a] ?? 0;
    const fn = fnPerAlpha[a] ?? 0;
    const fp = fpPerAlpha[a] ?? 0;

    const assa = safeDiv(assaWeighted[a] ?? 0, tp);
    const assRe = safeDiv(assReWeighted[a] ?? 0, tp);
    const assPr = safeDiv(assPrWeighted[a] ?? 0, tp);
    // hota.py: np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, HOTA_TP)
    const locA = Math.max(LOCA_EPS, locAWeighted[a] ?? 0) / Math.max(LOCA_EPS, tp);
    const deta = safeDiv(tp, tp + fn + fp);

    assaPerAlpha[a] = assa;
    assRePerAlpha[a] = assRe;
    assPrPerAlpha[a] = assPr;
    locAPerAlpha[a] = locA;
    detaPerAlpha[a] = deta;
    hotaPerAlpha[a] = Math.sqrt(deta * assa);

    detReSum += safeDiv(tp, tp + fn);
    detPrSum += safeDiv(tp, tp + fp);
    assReSum += assRe;
    assPrSum += assPr;
  }

  const mean = (arr: Float64Array): number => {
    let s = 0;
    for (const v of arr) s += v;
    return s / arr.length;
  };

  return {
    hota: mean(hotaPerAlpha),
    deta: mean(detaPerAlpha),
    assa: mean(assaPerAlpha),
    detRe: detReSum / numAlphas,
    detPr: detPrSum / numAlphas,
    assRe: assReSum / numAlphas,
    assPr: assPrSum / numAlphas,
    locA: mean(locAPerAlpha),
    alphas: first.alphas,
    hotaPerAlpha,
    detaPerAlpha,
    assaPerAlpha,
    locAPerAlpha,
    tpPerAlpha,
    fnPerAlpha,
    fpPerAlpha,
    assRePerAlpha,
    assPrPerAlpha,
  };
}
