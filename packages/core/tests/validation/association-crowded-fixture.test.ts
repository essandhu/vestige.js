/**
 * Cross-implementation validation of ASSOCIATION SEMANTICS, under competition.
 *
 * The three sister fixtures (`sort-abewley`, `ocsort-noahcao`,
 * `bytetrack-foundationvision`) all use well-separated boxes, so every cost matrix
 * they build has an unambiguous optimum — and under those conditions all three
 * reference conventions and vestige's agree. This fixture is the one that puts the
 * conventions under competition, where they do NOT agree. See ADR-0005.
 *
 * One crowded sequence, driven through all three references
 * (`packages/core/fixtures/association-crowded/gen.py`) and all three vestige
 * trackers here.
 *
 * The discriminating structure: a pair of overlapping objects (LEAD, TRAIL) one
 * gap apart, where TRAIL is occluded for a single frame at the same moment a NEW
 * object appears one gap on LEAD's far side. On that frame the tracks are
 * {LEAD, TRAIL} and the detections are {lead's box, new box}:
 *
 *                  det_lead   det_new
 *     track LEAD      1.00      0.41
 *     track TRAIL     0.41      0.09     <- below every threshold in play
 *
 * Every reference matches LEAD→det_lead, leaves TRAIL unmatched, and spawns a
 * fresh track for det_new. A MAX-CARDINALITY solver cannot leave TRAIL unmatched
 * while an admissible cell remains, so it is forced into LEAD→det_new +
 * TRAIL→det_lead — both tracks jump onto the wrong boxes and det_new never spawns.
 */
import { describe, expect, it } from 'vitest';
import data from '../../fixtures/association-crowded/data.json' with { type: 'json' };
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
  frames: FixtureFrame[];
}

const envelope = data as unknown as FixtureEnvelope;

/** Same tolerance as the three sister fixtures. If it bites, the divergence is real. */
const BBOX_TOLERANCE = 1e-3;

function detFromRow(row: number[]): Detection {
  return { bbox: [row[0]!, row[1]!, row[2]!, row[3]!] as BBox, score: row[4]! };
}

function maxBboxDelta(a: BBox, b: BBox): number {
  let m = 0;
  for (let i = 0; i < 4; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
  return m;
}

/**
 * Drive `tracker` through the whole sequence, asserting per-frame parity against
 * `refKey`'s reference output: identical output cardinality, every vestige bbox
 * within tolerance of a distinct reference bbox, and a stable id bijection.
 */
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
        `frame ${fixture.frame_index}: vestige id=${track.id} bbox=[${track.bbox.map((v) => v.toFixed(1)).join(',')}] has no reference bbox within tolerance (closest delta=${bestDelta.toFixed(4)})`,
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

describe('association semantics under competition — cross-implementation', () => {
  it(`SortTracker matches abewley/sort (sha ${envelope.generator.abewley_sha.slice(0, 8)})`, () => {
    const c = envelope.sort_config;
    assertParity(
      new SortTracker({ maxAge: c.max_age, minHits: c.min_hits, iouThreshold: c.iou_threshold }),
      'sort_out',
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
});
