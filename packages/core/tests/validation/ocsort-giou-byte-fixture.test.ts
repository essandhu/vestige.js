/**
 * Cross-implementation validation of OC-SORT's two previously-uncovered option paths:
 * `asoFunc: 'giou'` and `useByte: true`.
 *
 * Both are documented, exported options on {@link OcSortTrackerOptions}, and until this
 * landed neither had a single line of reference-backed coverage — every other
 * cross-implementation fixture in the repo runs `asso_func='iou'`, `use_byte=False`.
 * Two whole branches of shipped code, compared against nothing.
 *
 * `giou` is not a cosmetic swap: it changes the BYTE stage's cost matrix
 * (`ocsort.py:262`), the OCR stage's cost matrix (`ocsort.py:283`), and therefore which
 * tracks get recovered at all. It does NOT change the primary stage —
 * `association.py:associate` hard-codes `iou_batch` even under `asso_func='giou'`, and
 * the port preserves that asymmetry.
 *
 * Detections carry the same σ=4px Gaussian noise as `detection-noise/`: the clean regime
 * is what hid the export-convention bug, and there is no reason to extend it credit here.
 *
 * NOT covered here, deliberately: the never-observed-track OCR sentinel. Under `giou`
 * the reference will OCR-match a track that has never been observed against its
 * `[-1,-1,-1,-1]` PLACEHOLDER box, which scores 0.46 near the image origin. vestige
 * declines to recover a track from a box it never saw. That divergence is pinned by
 * `tests/unit/ocsort-ocr-sentinel.test.ts` and recorded in ADR-0007; this fixture keeps
 * its detections in an origin keep-out zone so it tests the paths where the two
 * implementations should AGREE. The carve-out is stated rather than left implicit —
 * a fixture that quietly steers around a divergence is how the previous green-but-wrong
 * fixtures happened.
 */
import { describe, expect, it } from 'vitest';
import data from '../../fixtures/ocsort-giou-byte/data.json' with { type: 'json' };
import { OcSortTracker } from '../../src/trackers/ocsort.js';
import type { BBox, Detection, Track } from '../../src/types.js';

interface FixtureFrame {
  frame_index: number;
  detections: number[][];
  tracks_out: number[][];
}

interface FixtureEnvelope {
  generator: { noahcao_sha: string };
  ocsort_config: {
    det_thresh: number;
    max_age: number;
    min_hits: number;
    iou_threshold: number;
    delta_t: number;
    asso_func: string;
    inertia: number;
    use_byte: boolean;
  };
  sequence: { noise_sigma: number; origin_keepout: number };
  frames: FixtureFrame[];
}

const envelope = data as unknown as FixtureEnvelope;
const BBOX_TOLERANCE = 1e-3;

function detFromRow(row: number[]): Detection {
  return { bbox: [row[0]!, row[1]!, row[2]!, row[3]!] as BBox, score: row[4]! };
}

function maxBboxDelta(a: BBox, b: BBox): number {
  let m = 0;
  for (let i = 0; i < 4; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
}

describe(`OcSortTracker — giou + useByte cross-implementation (sha ${envelope.generator.noahcao_sha.slice(0, 8)})`, () => {
  it(`reproduces noahcao across all ${envelope.frames.length} frames with asso_func=giou, use_byte=true`, () => {
    const c = envelope.ocsort_config;
    const tracker = new OcSortTracker({
      detThresh: c.det_thresh,
      maxAge: c.max_age,
      minHits: c.min_hits,
      iouThreshold: c.iou_threshold,
      deltaT: c.delta_t,
      asoFunc: c.asso_func as 'giou',
      inertia: c.inertia,
      useByte: c.use_byte,
    });

    const idMap = new Map<number, number>();
    const reverseIdMap = new Map<number, number>();

    for (const fixture of envelope.frames) {
      const out: Track[] = tracker.update(fixture.detections.map(detFromRow));
      const refRows = fixture.tracks_out;

      expect(
        out.length,
        `frame ${fixture.frame_index}: output count mismatch (reference=${refRows.length}, vestige=${out.length})`,
      ).toBe(refRows.length);

      if (out.length === 0) continue;

      const usedRefIds = new Set<number>();
      for (const track of out) {
        let best: number[] | null = null;
        let bestDelta = Number.POSITIVE_INFINITY;
        for (const row of refRows) {
          const d = maxBboxDelta(track.bbox, [row[0]!, row[1]!, row[2]!, row[3]!]);
          if (d < bestDelta) {
            bestDelta = d;
            best = row;
          }
        }
        expect(
          bestDelta,
          `frame ${fixture.frame_index}: vestige id=${track.id} bbox=[${track.bbox.map((v) => v.toFixed(2)).join(',')}] is ${bestDelta.toFixed(4)}px from the closest reference bbox`,
        ).toBeLessThanOrEqual(BBOX_TOLERANCE);

        const refId = best![4]!;
        expect(
          usedRefIds.has(refId),
          `frame ${fixture.frame_index}: reference id=${refId} matched to two vestige tracks`,
        ).toBe(false);
        usedRefIds.add(refId);

        const mapped = idMap.get(track.id);
        if (mapped === undefined) {
          expect(
            reverseIdMap.get(refId),
            `frame ${fixture.frame_index}: reference id=${refId} already maps from a different vestige id`,
          ).toBeUndefined();
          idMap.set(track.id, refId);
          reverseIdMap.set(refId, track.id);
        } else {
          expect(
            mapped,
            `frame ${fixture.frame_index}: vestige id=${track.id} was mapped to reference id=${mapped}, now matches reference id=${refId} (ID SWITCH)`,
          ).toBe(refId);
        }
      }
    }
  });
});
