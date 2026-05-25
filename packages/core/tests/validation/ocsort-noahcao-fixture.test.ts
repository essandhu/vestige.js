/**
 * Cross-implementation validation: drive {@link OcSortTracker} through the
 * same synthetic sequence used to produce
 * `packages/core/fixtures/ocsort-noahcao/data.json` (generated against
 * `noahcao/OC_SORT` at a pinned commit) and assert per-frame output parity.
 *
 * This is the strongest defense against silent faithfulness drift: the
 * scenario-based tests in `ocsort-synthetic.test.ts` check what OC-SORT is
 * SUPPOSED to do; this test checks what the canonical reference ACTUALLY does
 * across 60 frames of engineered motion + occlusion. See
 * `packages/core/fixtures/ocsort-noahcao/README.md` for the sequence design
 * and `packages/core/fixtures/README.md` for the regeneration workflow.
 */
import { describe, expect, it } from 'vitest';
import data from '../../fixtures/ocsort-noahcao/data.json' with { type: 'json' };
import { OcSortTracker } from '../../src/trackers/ocsort.js';
import type { BBox, Detection, Track } from '../../src/types.js';

interface FixtureFrame {
  frame_index: number;
  /** noahcao input format: rows of [x1, y1, x2, y2, score]. */
  detections: number[][];
  /** noahcao output format: rows of [x1, y1, x2, y2, id]. */
  tracks_out: number[][];
}

interface FixtureEnvelope {
  $schema: string;
  generator: { noahcao_sha: string; numpy: string; scipy: string; python: string };
  ocsort_config: {
    det_thresh: number;
    max_age: number;
    min_hits: number;
    iou_threshold: number;
    delta_t: number;
    asso_func: 'iou' | 'giou';
    inertia: number;
    use_byte: boolean;
  };
  sequence: { n_frames: number; box_size: number };
  frames: FixtureFrame[];
}

const envelope = data as unknown as FixtureEnvelope;

/**
 * Per-frame bbox tolerance. Cross-implementation Kalman state drift across
 * 60 frames stays well under 1e-3 in practice; if this assertion ever bites,
 * the divergence is real (don't loosen — investigate). Tighter than the
 * fixture envelope's primitive-level `~1e-12` (ARCHITECTURE.md §10.1) because
 * tracker state compounds across frames.
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

/** Match a vestige Track to its noahcao counterpart on the SAME frame by closest bbox. */
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

describe('OcSortTracker — noahcao cross-implementation faithfulness', () => {
  it(`reproduces noahcao output across all ${envelope.frames.length} frames (sha ${envelope.generator.noahcao_sha.slice(0, 8)})`, () => {
    const cfg = envelope.ocsort_config;
    const tracker = new OcSortTracker({
      detThresh: cfg.det_thresh,
      maxAge: cfg.max_age,
      minHits: cfg.min_hits,
      iouThreshold: cfg.iou_threshold,
      deltaT: cfg.delta_t,
      asoFunc: cfg.asso_func,
      inertia: cfg.inertia,
      useByte: cfg.use_byte,
    });

    /**
     * Lazy `vestigeId → noahcaoId` mapping. Populated on first observation of
     * each vestige ID by snapping to the nearest noahcao bbox in the same
     * frame's output. Once set, the mapping is asserted consistent forever.
     */
    const idMap = new Map<number, number>();
    /** Reverse map for symmetry checks — every noahcao id seen must trace back to exactly one vestige id. */
    const reverseIdMap = new Map<number, number>();

    for (let i = 0; i < envelope.frames.length; i++) {
      const fixture = envelope.frames[i]!;
      const dets = fixture.detections.map(detFromRow);
      const out = tracker.update(dets);

      // Output cardinality must match exactly. A mismatch means either the
      // detection partition, the stage-1/2/3 association, or the export rule
      // diverged from the reference.
      expect(
        out.length,
        `frame ${fixture.frame_index}: output count mismatch (noahcao=${fixture.tracks_out.length}, vestige=${out.length})`,
      ).toBe(fixture.tracks_out.length);

      if (out.length === 0) continue;

      // Build / verify the id correspondence for this frame.
      const usedRefIds = new Set<number>();
      for (const track of out) {
        const { row: refRow, delta } = findClosestRefRow(track, fixture.tracks_out);
        expect(
          delta,
          `frame ${fixture.frame_index}: vestige id=${track.id} bbox=[${track.bbox.join(',')}] has no noahcao bbox within tolerance (closest delta=${delta})`,
        ).toBeLessThanOrEqual(BBOX_TOLERANCE);

        const refId = refRow[4]!;
        expect(
          usedRefIds.has(refId),
          `frame ${fixture.frame_index}: noahcao id=${refId} was matched to two vestige tracks`,
        ).toBe(false);
        usedRefIds.add(refId);

        const mapped = idMap.get(track.id);
        if (mapped === undefined) {
          // Reverse-direction sanity: this noahcao id must not already be claimed.
          const reverseClaimed = reverseIdMap.get(refId);
          expect(
            reverseClaimed,
            `frame ${fixture.frame_index}: noahcao id=${refId} previously mapped from vestige id=${reverseClaimed}, now appearing for vestige id=${track.id}`,
          ).toBeUndefined();
          idMap.set(track.id, refId);
          reverseIdMap.set(refId, track.id);
        } else {
          expect(
            mapped,
            `frame ${fixture.frame_index}: vestige id=${track.id} previously mapped to noahcao id=${mapped}, now matched to noahcao id=${refId}`,
          ).toBe(refId);
        }
      }
    }
  });
});
