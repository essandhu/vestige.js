import { solveLsap } from '../core.js';
import type { EvalFrame } from './frames.js';
import { indexSequence } from './frames.js';

/**
 * HOTA results (Luiten et al., IJCV 2020), computed the way
 * `JonathonLuiten/TrackEval` (`metrics/hota.py`) computes them — that is the
 * §4.2 reference implementation for this module.
 *
 * Every scalar is the mean of the corresponding per-alpha value over the 19
 * localization thresholds `alphas = 0.05, 0.10, …, 0.95`. The per-alpha
 * arrays are exposed so benchmark reports can show the full curve.
 */
export interface HotaResult {
  /** Higher Order Tracking Accuracy: mean over alphas of `√(DetA·AssA)`. */
  readonly hota: number;
  /** Detection accuracy: mean over alphas of `tp / (tp + fn + fp)`. */
  readonly deta: number;
  /** Association accuracy: mean over alphas of the TP-averaged `A(c)` score. */
  readonly assa: number;
  /** Detection recall: mean over alphas of `tp / (tp + fn)`. */
  readonly detRe: number;
  /** Detection precision: mean over alphas of `tp / (tp + fp)`. */
  readonly detPr: number;
  /** Association recall: mean over alphas of the TP-averaged gt-side coverage. */
  readonly assRe: number;
  /** Association precision: mean over alphas of the TP-averaged tracker-side coverage. */
  readonly assPr: number;
  /**
   * Localization accuracy: mean over alphas of mean-IoU-over-TPs. Follows
   * TrackEval's convention that an alpha with zero TPs contributes 1 (the
   * `np.maximum(1e-10, …)` guard in `hota.py`), so LocA stays a pure
   * localization signal and never double-counts detection failures.
   */
  readonly locA: number;
  /** The 19 localization thresholds, `(k+1)/20` for `k = 0…18`. */
  readonly alphas: ReadonlyArray<number>;
  /** Per-alpha HOTA, parallel to `alphas`. */
  readonly hotaPerAlpha: Float64Array;
  /** Per-alpha DetA, parallel to `alphas`. */
  readonly detaPerAlpha: Float64Array;
  /** Per-alpha AssA, parallel to `alphas`. */
  readonly assaPerAlpha: Float64Array;
  /** Per-alpha LocA, parallel to `alphas`. */
  readonly locAPerAlpha: Float64Array;
}

/**
 * Compute HOTA and its decomposition over a sequence.
 *
 * Matching follows TrackEval `hota.py`: a first pass accumulates a global
 * alignment score per (gt id, tracker id) pair; a second pass solves one
 * per-frame assignment maximizing `globalAlignment · IoU` (so association
 * context breaks per-frame ties), then each alpha counts the matched pairs
 * with `IoU ≥ alpha` as TPs. Association scores `A(c)` are computed per TP
 * from the per-pair match counts.
 *
 * Throws on a sequence with zero ground-truth detections
 * (`/ground[\s-]?truth/i`); see {@link indexSequence} for the shared
 * degenerate-input contract.
 */
export function hota(frames: ReadonlyArray<EvalFrame>): HotaResult {
  const seq = indexSequence(frames);
  const numGt = seq.numGtIds;
  const numTr = seq.numTrackIds;
  const numAlphas = ALPHAS.length;

  // Pass 1 (hota.py "First loop"): accumulate the global alignment evidence.
  // Each frame contributes a normalized similarity sim/(rowSum + colSum − sim)
  // per (gt, tracker) pair, so pairs that consistently co-occur dominate.
  const potential = new Float64Array(numGt * numTr);
  for (const frame of seq.frames) {
    const m = frame.gt.length;
    const n = frame.track.length;
    if (m === 0 || n === 0) continue;

    const rowSum = new Float64Array(m);
    const colSum = new Float64Array(n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        const sim = frame.sim[i * n + j] ?? 0;
        rowSum[i] = (rowSum[i] ?? 0) + sim;
        colSum[j] = (colSum[j] ?? 0) + sim;
      }
    }
    for (let i = 0; i < m; i++) {
      const gtId = frame.gt[i] ?? 0;
      for (let j = 0; j < n; j++) {
        const sim = frame.sim[i * n + j] ?? 0;
        const denom = (rowSum[i] ?? 0) + (colSum[j] ?? 0) - sim;
        if (denom > Number.EPSILON) {
          const pair = gtId * numTr + (frame.track[j] ?? 0);
          potential[pair] = (potential[pair] ?? 0) + sim / denom;
        }
      }
    }
  }

  const globalAlignment = new Float64Array(numGt * numTr);
  for (let g = 0; g < numGt; g++) {
    for (let t = 0; t < numTr; t++) {
      const p = potential[g * numTr + t] ?? 0;
      globalAlignment[g * numTr + t] =
        p / ((seq.gtIdCounts[g] ?? 0) + (seq.trackIdCounts[t] ?? 0) - p);
    }
  }

  // Pass 2 (hota.py "second loop"): one assignment per frame maximizing
  // globalAlignment·IoU, then per-alpha TP counting of the selected pairs.
  const tpPerAlpha = new Float64Array(numAlphas);
  const locSumPerAlpha = new Float64Array(numAlphas);
  const matchCounts: Float64Array[] = [];
  for (let a = 0; a < numAlphas; a++) matchCounts.push(new Float64Array(numGt * numTr));

  for (const frame of seq.frames) {
    const m = frame.gt.length;
    const n = frame.track.length;
    if (m === 0 || n === 0) continue;

    // score ≤ 1, so cost = 2 − score stays positive for the minimizing solver.
    const cost = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      const gtId = frame.gt[i] ?? 0;
      for (let j = 0; j < n; j++) {
        const sim = frame.sim[i * n + j] ?? 0;
        cost[i * n + j] = 2 - (globalAlignment[gtId * numTr + (frame.track[j] ?? 0)] ?? 0) * sim;
      }
    }

    const { rowToCol } = solveLsap(cost, m, n);
    for (let i = 0; i < m; i++) {
      const j = rowToCol[i] ?? -1;
      if (j === -1) continue;
      const sim = frame.sim[i * n + j] ?? 0;
      const pair = (frame.gt[i] ?? 0) * numTr + (frame.track[j] ?? 0);
      for (let a = 0; a < numAlphas; a++) {
        const alpha = ALPHAS[a] ?? 0;
        if (sim >= alpha - Number.EPSILON) {
          tpPerAlpha[a] = (tpPerAlpha[a] ?? 0) + 1;
          locSumPerAlpha[a] = (locSumPerAlpha[a] ?? 0) + sim;
          const counts = matchCounts[a];
          if (counts !== undefined) counts[pair] = (counts[pair] ?? 0) + 1;
        }
      }
    }
  }

  // Per-alpha scores, then average over alphas.
  const hotaPerAlpha = new Float64Array(numAlphas);
  const detaPerAlpha = new Float64Array(numAlphas);
  const assaPerAlpha = new Float64Array(numAlphas);
  const locAPerAlpha = new Float64Array(numAlphas);
  let detReSum = 0;
  let detPrSum = 0;
  let assReSum = 0;
  let assPrSum = 0;

  for (let a = 0; a < numAlphas; a++) {
    const tp = tpPerAlpha[a] ?? 0;
    const fn = seq.numGtDets - tp;
    const fp = seq.numTrackDets - tp;
    const counts = matchCounts[a] ?? new Float64Array(0);

    let assaSumTp = 0;
    let assReSumTp = 0;
    let assPrSumTp = 0;
    for (let g = 0; g < numGt; g++) {
      const gtCount = seq.gtIdCounts[g] ?? 0;
      for (let t = 0; t < numTr; t++) {
        const c = counts[g * numTr + t] ?? 0;
        if (c === 0) continue;
        const trCount = seq.trackIdCounts[t] ?? 0;
        assaSumTp += c * (c / (gtCount + trCount - c));
        assReSumTp += c * (c / gtCount);
        assPrSumTp += c * (c / trCount);
      }
    }

    const deta = tp / (tp + fn + fp);
    const assa = tp > 0 ? assaSumTp / tp : 0;
    detaPerAlpha[a] = deta;
    assaPerAlpha[a] = assa;
    hotaPerAlpha[a] = Math.sqrt(deta * assa);
    locAPerAlpha[a] = tp > 0 ? (locSumPerAlpha[a] ?? 0) / tp : 1;

    detReSum += tp / (tp + fn);
    detPrSum += tp + fp > 0 ? tp / (tp + fp) : 0;
    assReSum += tp > 0 ? assReSumTp / tp : 0;
    assPrSum += tp > 0 ? assPrSumTp / tp : 0;
  }

  return {
    hota: mean(hotaPerAlpha),
    deta: mean(detaPerAlpha),
    assa: mean(assaPerAlpha),
    detRe: detReSum / numAlphas,
    detPr: detPrSum / numAlphas,
    assRe: assReSum / numAlphas,
    assPr: assPrSum / numAlphas,
    locA: mean(locAPerAlpha),
    alphas: ALPHAS,
    hotaPerAlpha,
    detaPerAlpha,
    assaPerAlpha,
    locAPerAlpha,
  };
}

/** The 19 standard HOTA localization thresholds, 0.05 … 0.95. */
const ALPHAS: ReadonlyArray<number> = Array.from({ length: 19 }, (_, k) => (k + 1) / 20);

function mean(values: Float64Array): number {
  let sum = 0;
  for (const v of values) sum += v;
  return sum / values.length;
}
