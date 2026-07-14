/**
 * Cross-implementation validation under REALISTIC DETECTION NOISE.
 *
 * Every other synthetic fixture in this repo feeds the trackers noise-free detections
 * on perfectly linear trajectories. That is not a neutral simplification — it is a
 * regime, and it hides a whole class of bug.
 *
 * Under noise-free linear motion a Kalman filter's posterior converges exactly onto
 * the measurement: the two become numerically identical (measured delta on the
 * `ocsort-noahcao` sequence: 0.000000 px). So a tracker that exports the KALMAN STATE
 * and one that exports the RAW DETECTION emit byte-identical output, and no fixture
 * built on clean detections can tell them apart. `noahcao/OC_SORT` exports the raw last
 * observation (`ocsort.py:313-320`); vestige exported the Kalman posterior, under a
 * code comment asserting the two were "equal-by-design … within Kalman-filter noise".
 * They are equal only in the regime the fixtures happened to use.
 *
 * `sort.py` exports `get_state()` and ByteTrack exports `STrack.tlbr`, both
 * Kalman-derived, so the port is correct for those two. They are driven here anyway:
 * showing that the noise regime does NOT move them is what isolates the OC-SORT bug. A
 * fixture that only drives the tracker you already suspect isn't evidence.
 *
 * Noise is Gaussian, σ = 4 px per box coordinate, applied to detections only —
 * ground-truth motion stays linear and the lanes stay well-separated, so nothing here
 * is about crowding (that's `association-crowded/`) or about exotic trajectories. Any
 * divergence is about measurement noise and nothing else.
 *
 * See ADR-0007.
 */
import { describe, expect, it } from 'vitest';
import data from '../../fixtures/detection-noise/data.json' with { type: 'json' };
import { ByteTracker } from '../../src/trackers/bytetrack.js';
import { OcSortTracker } from '../../src/trackers/ocsort.js';
import { SortTracker } from '../../src/trackers/sort.js';
import type { BBox, Detection, Track, Tracker } from '../../src/types.js';

interface FixtureFrame {
  frame_index: number;
  detections: number[][];
  sort_out: number[][];
  ocsort_out: number[][];
  bytetrack_out: number[][];
}

interface FixtureEnvelope {
  generator: { abewley_sha: string; noahcao_sha: string; foundationvision_sha: string };
  sort_config: { max_age: number; min_hits: number; iou_threshold: number };
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
  bytetrack_config: {
    track_thresh: number;
    track_buffer: number;
    match_thresh: number;
    mot20: boolean;
    frame_rate: number;
  };
  sequence: { noise_sigma: number; seed: number };
  frames: FixtureFrame[];
}

const envelope = data as unknown as FixtureEnvelope;

/**
 * Same 1e-3 as every sister fixture. Do NOT loosen this to accommodate noise — the
 * noise is in the *inputs*, which both implementations see identically (the noisy boxes
 * are committed in `data.json`, not re-drawn). Both then run the same Kalman filter on
 * the same measurements, so agreement should still be near machine precision. A delta
 * of a few px here means the two are exporting different QUANTITIES, which is exactly
 * the bug this fixture exists to catch.
 */
const BBOX_TOLERANCE = 1e-3;

function detFromRow(row: number[]): Detection {
  return { bbox: [row[0]!, row[1]!, row[2]!, row[3]!] as BBox, score: row[4]! };
}

function maxBboxDelta(a: BBox, b: BBox): number {
  let m = 0;
  for (let i = 0; i < 4; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
}

function assertParity(tracker: Tracker, refKey: 'sort_out' | 'ocsort_out' | 'bytetrack_out'): void {
  const idMap = new Map<number, number>();
  const reverseIdMap = new Map<number, number>();

  for (const fixture of envelope.frames) {
    const refRows = fixture[refKey];
    const out: Track[] = tracker.update(fixture.detections.map(detFromRow));

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
        `frame ${fixture.frame_index}: vestige id=${track.id} bbox=[${track.bbox.map((v) => v.toFixed(2)).join(',')}] is ${bestDelta.toFixed(3)}px from the closest reference bbox. A delta of this size means the two are exporting DIFFERENT QUANTITIES (Kalman posterior vs raw observation), not drifting.`,
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
}

describe(`detection noise (σ=${envelope.sequence.noise_sigma}px) — cross-implementation`, () => {
  it(`SortTracker matches abewley/sort (sha ${envelope.generator.abewley_sha.slice(0, 8)})`, () => {
    const c = envelope.sort_config;
    assertParity(
      new SortTracker({ maxAge: c.max_age, minHits: c.min_hits, iouThreshold: c.iou_threshold }),
      'sort_out',
    );
  });

  it(`ByteTracker matches FoundationVision/ByteTrack (sha ${envelope.generator.foundationvision_sha.slice(0, 8)})`, () => {
    const c = envelope.bytetrack_config;
    assertParity(
      new ByteTracker({
        trackThresh: c.track_thresh,
        trackBuffer: c.track_buffer,
        matchThresh: c.match_thresh,
        frameRate: c.frame_rate,
        mot20: c.mot20,
      }),
      'bytetrack_out',
    );
  });

  it(`OcSortTracker matches noahcao/OC_SORT (sha ${envelope.generator.noahcao_sha.slice(0, 8)})`, () => {
    const c = envelope.ocsort_config;
    assertParity(
      new OcSortTracker({
        detThresh: c.det_thresh,
        maxAge: c.max_age,
        minHits: c.min_hits,
        iouThreshold: c.iou_threshold,
        deltaT: c.delta_t,
        asoFunc: c.asso_func as 'iou',
        inertia: c.inertia,
        useByte: c.use_byte,
      }),
      'ocsort_out',
    );
  });
});
