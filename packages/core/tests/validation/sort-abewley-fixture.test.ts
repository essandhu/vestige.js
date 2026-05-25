/**
 * Cross-implementation validation: drive {@link SortTracker} through the
 * same synthetic sequence used to produce
 * `packages/core/fixtures/sort-abewley/data.json` (generated against
 * `abewley/sort` at a pinned commit) and assert per-frame output parity.
 *
 * Sister fixture to `ocsort-noahcao-fixture.test.ts`. The mechanism is the
 * same — what differs is the reference (`sort.py` here vs `ocsort.py` there)
 * and the sequence (engineered for SortTracker's narrower default
 * `maxAge=1` occlusion window).
 *
 * See `packages/core/fixtures/sort-abewley/README.md` for the sequence
 * design (which bug-window frames matter, and why).
 */
import { describe, expect, it } from 'vitest';
import data from '../../fixtures/sort-abewley/data.json' with { type: 'json' };
import { SortTracker } from '../../src/trackers/sort.js';
import type { BBox, Detection, Track } from '../../src/types.js';

interface FixtureFrame {
  frame_index: number;
  /** abewley input format: rows of [x1, y1, x2, y2, score]. */
  detections: number[][];
  /** abewley output format: rows of [x1, y1, x2, y2, id]. */
  tracks_out: number[][];
}

interface FixtureEnvelope {
  $schema: string;
  generator: { abewley_sha: string; numpy: string; scipy: string; python: string };
  sort_config: {
    max_age: number;
    min_hits: number;
    iou_threshold: number;
  };
  sequence: { n_frames: number; box_size: number };
  frames: FixtureFrame[];
}

const envelope = data as unknown as FixtureEnvelope;

/**
 * Per-frame bbox tolerance. Cross-implementation Kalman state drift across
 * 60 frames stays well under 1e-3 in practice; if this assertion ever bites,
 * the divergence is real (don't loosen — investigate). Same tolerance as
 * `ocsort-noahcao-fixture.test.ts`.
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

/** Match a vestige Track to its abewley counterpart on the SAME frame by closest bbox. */
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

describe('SortTracker — abewley cross-implementation faithfulness', () => {
  it(`reproduces abewley output across all ${envelope.frames.length} frames (sha ${envelope.generator.abewley_sha.slice(0, 8)})`, () => {
    const cfg = envelope.sort_config;
    const tracker = new SortTracker({
      maxAge: cfg.max_age,
      minHits: cfg.min_hits,
      iouThreshold: cfg.iou_threshold,
    });

    /**
     * Lazy `vestigeId → abewleyId` mapping. Populated on first observation of
     * each vestige ID by snapping to the nearest abewley bbox in the same
     * frame's output. Once set, the mapping is asserted consistent forever.
     */
    const idMap = new Map<number, number>();
    /** Reverse map for symmetry checks — every abewley id seen must trace back to exactly one vestige id. */
    const reverseIdMap = new Map<number, number>();

    for (let i = 0; i < envelope.frames.length; i++) {
      const fixture = envelope.frames[i]!;
      const dets = fixture.detections.map(detFromRow);
      const out = tracker.update(dets);

      // Output cardinality must match exactly. A mismatch means either the
      // association, the lifecycle transitions, or the export rule diverged
      // from the reference. The bug-window frames documented in the fixture
      // README are where this assertion catches the `state === 'confirmed'`
      // vs `hit_streak >= min_hits` divergence.
      expect(
        out.length,
        `frame ${fixture.frame_index}: output count mismatch (abewley=${fixture.tracks_out.length}, vestige=${out.length})`,
      ).toBe(fixture.tracks_out.length);

      if (out.length === 0) continue;

      // Build / verify the id correspondence for this frame.
      const usedRefIds = new Set<number>();
      for (const track of out) {
        const { row: refRow, delta } = findClosestRefRow(track, fixture.tracks_out);
        expect(
          delta,
          `frame ${fixture.frame_index}: vestige id=${track.id} bbox=[${track.bbox.join(',')}] has no abewley bbox within tolerance (closest delta=${delta})`,
        ).toBeLessThanOrEqual(BBOX_TOLERANCE);

        const refId = refRow[4]!;
        expect(
          usedRefIds.has(refId),
          `frame ${fixture.frame_index}: abewley id=${refId} was matched to two vestige tracks`,
        ).toBe(false);
        usedRefIds.add(refId);

        const mapped = idMap.get(track.id);
        if (mapped === undefined) {
          // Reverse-direction sanity: this abewley id must not already be claimed.
          const reverseClaimed = reverseIdMap.get(refId);
          expect(
            reverseClaimed,
            `frame ${fixture.frame_index}: abewley id=${refId} previously mapped from vestige id=${reverseClaimed}, now appearing for vestige id=${track.id}`,
          ).toBeUndefined();
          idMap.set(track.id, refId);
          reverseIdMap.set(refId, track.id);
        } else {
          expect(
            mapped,
            `frame ${fixture.frame_index}: vestige id=${track.id} previously mapped to abewley id=${mapped}, now matched to abewley id=${refId}`,
          ).toBe(refId);
        }
      }
    }
  });
});
