/**
 * Cross-implementation validation: drive {@link ByteTracker} through the same
 * synthetic sequence used to produce
 * `packages/core/fixtures/bytetrack-foundationvision/data.json` (generated against
 * `FoundationVision/ByteTrack` at a pinned commit) and assert per-frame output parity.
 *
 * Sister fixture to `sort-abewley-fixture.test.ts` and
 * `ocsort-noahcao-fixture.test.ts` — same harness, third reference. Retires the
 * ByteTracker row in `docs/decisions/0003-tracker-lifecycle-bookkeeping.md` §6.
 *
 * ByteTracker is the structurally riskiest of the three (three-stage association,
 * asymmetric tentative/lost retention, frame-1 instant activation), so the sequence
 * is built to walk every lifecycle edge rather than just the happy path. See
 * `packages/core/fixtures/bytetrack-foundationvision/README.md` for the per-track
 * design and for the two documented divergences from the reference (the cython_bbox
 * "+1" IoU convention, and the reference's one-frame track-resurrection bug) that
 * the sequence deliberately steers clear of.
 */
import { describe, expect, it } from 'vitest';
import data from '../../fixtures/bytetrack-foundationvision/data.json' with { type: 'json' };
import { ByteTracker } from '../../src/trackers/bytetrack.js';
import type { BBox, Detection, Track } from '../../src/types.js';

interface FixtureFrame {
  frame_index: number;
  /** Reference input format: rows of [x1, y1, x2, y2, score]. */
  detections: number[][];
  /** Reference output format: rows of [x1, y1, x2, y2, id] (from `STrack.tlbr`). */
  tracks_out: number[][];
}

interface FixtureEnvelope {
  $schema: string;
  generator: { foundationvision_sha: string; numpy: string; scipy: string; python: string };
  bytetrack_config: {
    track_thresh: number;
    track_buffer: number;
    match_thresh: number;
    mot20: boolean;
    frame_rate: number;
  };
  sequence: { n_frames: number; box_size: number; max_time_lost: number };
  frames: FixtureFrame[];
}

const envelope = data as unknown as FixtureEnvelope;

/**
 * Per-frame bbox tolerance. Both implementations run the same cv-xyah Kalman
 * filter, so agreement should be near machine-precision; the sequence's longest
 * predict-only stretch is 30 frames (track D) and drift stays far under this.
 * If this assertion ever bites, the divergence is real — don't loosen it,
 * investigate. Same tolerance as the two sister fixtures.
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

/** Match a vestige Track to its reference counterpart on the SAME frame by closest bbox. */
function findClosestRefRow(track: Track, refRows: number[][]): { row: number[]; delta: number } {
  let best: number[] | null = null;
  let bestDelta = Number.POSITIVE_INFINITY;
  for (const row of refRows) {
    const refBox: BBox = [row[0]!, row[1]!, row[2]!, row[3]!];
    const d = maxBboxDelta(track.bbox, refBox);
    if (d < bestDelta) {
      bestDelta = d;
      best = row;
    }
  }
  if (best === null) throw new Error('empty refRows — guarded by caller');
  return { row: best, delta: bestDelta };
}

describe('ByteTracker — FoundationVision cross-implementation faithfulness', () => {
  it(`reproduces ByteTrack output across all ${envelope.frames.length} frames (sha ${envelope.generator.foundationvision_sha.slice(0, 8)})`, () => {
    const cfg = envelope.bytetrack_config;
    const tracker = new ByteTracker({
      trackThresh: cfg.track_thresh,
      trackBuffer: cfg.track_buffer,
      matchThresh: cfg.match_thresh,
      frameRate: cfg.frame_rate,
      mot20: cfg.mot20,
    });

    /**
     * Lazy `vestigeId → referenceId` mapping, populated on first observation of each
     * vestige id by snapping to the nearest reference bbox in the same frame's output.
     * Once set, the mapping is asserted consistent forever. This is what catches an
     * id switch: if vestige re-uses an id where the reference minted a fresh one (or
     * vice versa), the bijection breaks.
     */
    const idMap = new Map<number, number>();
    /** Reverse map — every reference id seen must trace back to exactly one vestige id. */
    const reverseIdMap = new Map<number, number>();

    for (let i = 0; i < envelope.frames.length; i++) {
      const fixture = envelope.frames[i]!;
      const dets = fixture.detections.map(detFromRow);
      const out = tracker.update(dets);

      // Output cardinality must match exactly. A mismatch means the association, a
      // lifecycle transition, or the export rule diverged. The lifecycle-edge frames
      // documented in the fixture README (tentative promotion, lost re-activation,
      // the max_time_lost boundary, the score-band boundaries) are where this bites.
      expect(
        out.length,
        `frame ${fixture.frame_index}: output count mismatch (reference=${fixture.tracks_out.length}, vestige=${out.length})`,
      ).toBe(fixture.tracks_out.length);

      if (out.length === 0) continue;

      const usedRefIds = new Set<number>();
      for (const track of out) {
        const { row: refRow, delta } = findClosestRefRow(track, fixture.tracks_out);
        expect(
          delta,
          `frame ${fixture.frame_index}: vestige id=${track.id} bbox=[${track.bbox.join(',')}] has no reference bbox within tolerance (closest delta=${delta})`,
        ).toBeLessThanOrEqual(BBOX_TOLERANCE);

        const refId = refRow[4]!;
        expect(
          usedRefIds.has(refId),
          `frame ${fixture.frame_index}: reference id=${refId} was matched to two vestige tracks`,
        ).toBe(false);
        usedRefIds.add(refId);

        const mapped = idMap.get(track.id);
        if (mapped === undefined) {
          const reverseClaimed = reverseIdMap.get(refId);
          expect(
            reverseClaimed,
            `frame ${fixture.frame_index}: reference id=${refId} previously mapped from vestige id=${reverseClaimed}, now appearing for vestige id=${track.id}`,
          ).toBeUndefined();
          idMap.set(track.id, refId);
          reverseIdMap.set(refId, track.id);
        } else {
          expect(
            mapped,
            `frame ${fixture.frame_index}: vestige id=${track.id} previously mapped to reference id=${mapped}, now matched to reference id=${refId}`,
          ).toBe(refId);
        }
      }
    }
  });
});
