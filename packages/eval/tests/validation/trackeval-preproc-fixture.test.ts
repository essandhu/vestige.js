import { describe, expect, it } from 'vitest';
import fixture from '../../fixtures/trackeval-preproc/data.json' with { type: 'json' };
import { clearMot } from '../../src/metrics/clearmot.js';
import type { EvalFrame } from '../../src/metrics/frames.js';
import { hota } from '../../src/metrics/hota.js';
import { identity } from '../../src/metrics/identity.js';
import { parseMotChallenge } from '../../src/motchallenge/parse.js';
import { preprocessMotSequence } from '../../src/motchallenge/preprocess.js';

// Cross-implementation parity with JonathonLuiten/TrackEval over the whole
// pipeline: parse → preprocess (distractor removal, zero-mark/class drop) →
// CLEAR / Identity / HOTA. The fixture (see its README for the sequence
// design and regeneration workflow) was produced by running TrackEval at the
// commit pinned in the JSON envelope on the exact same input lines.

const preprocessed = (): EvalFrame[] => {
  const gt = parseMotChallenge(fixture.sequence.gtLines.join('\n'));
  const tracker = parseMotChallenge(fixture.sequence.trackerLines.join('\n'));
  return preprocessMotSequence(gt, tracker, { numFrames: fixture.sequence.numFrames });
};

/**
 * TrackEval relabels kept ids to `0…n-1` in ascending original-id order
 * before scoring; apply the same relabeling to our (original-id) output so
 * the per-frame arrays are directly comparable.
 */
const relabel = (frames: ReadonlyArray<EvalFrame>) => {
  const toMap = (ids: number[]) =>
    new Map([...new Set(ids)].sort((a, b) => a - b).map((id, i) => [id, i]));
  const gtMap = toMap(frames.flatMap((f) => [...f.gtIds]));
  const trackMap = toMap(frames.flatMap((f) => [...f.trackIds]));
  return frames.map((f) => ({
    gtIds: f.gtIds.map((id) => gtMap.get(id) ?? -1),
    trackerIds: f.trackIds.map((id) => trackMap.get(id) ?? -1),
  }));
};

describe('TrackEval cross-implementation parity (trackeval-preproc fixture)', () => {
  it('preprocesses to the same per-frame gt and tracker detections', () => {
    const frames = preprocessed();

    let gtDets = 0;
    let trackDets = 0;
    for (const f of frames) {
      gtDets += f.gtIds.length;
      trackDets += f.trackIds.length;
    }
    expect(gtDets).toBe(fixture.preprocessed.numGtDets);
    expect(trackDets).toBe(fixture.preprocessed.numTrackerDets);
    expect(new Set(frames.flatMap((f) => [...f.gtIds])).size).toBe(fixture.preprocessed.numGtIds);
    expect(new Set(frames.flatMap((f) => [...f.trackIds])).size).toBe(
      fixture.preprocessed.numTrackerIds,
    );

    expect(relabel(frames)).toEqual(fixture.preprocessed.frames);
  });

  it('reproduces TrackEval CLEAR results', () => {
    const r = clearMot(preprocessed());
    const e = fixture.metrics.clear;
    expect(r.tp).toBe(e.CLR_TP);
    expect(r.fn).toBe(e.CLR_FN);
    expect(r.fp).toBe(e.CLR_FP);
    expect(r.idsw).toBe(e.IDSW);
    expect(r.frag).toBe(e.Frag);
    expect(r.mt).toBe(e.MT);
    expect(r.pt).toBe(e.PT);
    expect(r.ml).toBe(e.ML);
    expect(r.mota).toBeCloseTo(e.MOTA, 9);
    expect(r.motp).toBeCloseTo(e.MOTP, 9);
  });

  it('reproduces TrackEval Identity results', () => {
    const r = identity(preprocessed());
    const e = fixture.metrics.identity;
    expect(r.idtp).toBe(e.IDTP);
    expect(r.idfn).toBe(e.IDFN);
    expect(r.idfp).toBe(e.IDFP);
    expect(r.idf1).toBeCloseTo(e.IDF1, 9);
    expect(r.idr).toBeCloseTo(e.IDR, 9);
    expect(r.idp).toBeCloseTo(e.IDP, 9);
  });

  it('reproduces TrackEval HOTA results per alpha and in the mean', () => {
    const r = hota(preprocessed());
    const e = fixture.metrics.hota;

    expect(r.hota).toBeCloseTo(e.HOTA.mean, 9);
    expect(r.deta).toBeCloseTo(e.DetA.mean, 9);
    expect(r.assa).toBeCloseTo(e.AssA.mean, 9);
    expect(r.detRe).toBeCloseTo(e.DetRe.mean, 9);
    expect(r.detPr).toBeCloseTo(e.DetPr.mean, 9);
    expect(r.assRe).toBeCloseTo(e.AssRe.mean, 9);
    expect(r.assPr).toBeCloseTo(e.AssPr.mean, 9);
    expect(r.locA).toBeCloseTo(e.LocA.mean, 9);

    for (let a = 0; a < 19; a++) {
      expect(r.hotaPerAlpha[a], `HOTA alpha ${a}`).toBeCloseTo(e.HOTA.perAlpha[a] ?? -1, 9);
      expect(r.detaPerAlpha[a], `DetA alpha ${a}`).toBeCloseTo(e.DetA.perAlpha[a] ?? -1, 9);
      expect(r.assaPerAlpha[a], `AssA alpha ${a}`).toBeCloseTo(e.AssA.perAlpha[a] ?? -1, 9);
      expect(r.locAPerAlpha[a], `LocA alpha ${a}`).toBeCloseTo(e.LocA.perAlpha[a] ?? -1, 9);
    }
  });
});
