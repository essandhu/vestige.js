/**
 * Cross-implementation validation of ByteTrack's `remove_duplicate_stracks`
 * (`byte_tracker.py:317-330`), which fires only when a TRACKED track and a LOST track
 * overlap at IoU > 0.85.
 *
 * No other fixture in the repo has ever put two tracks in that position — every sequence
 * keeps its boxes well separated, precisely so ids stay unambiguous. So the function
 * shipped with zero reference-backed coverage, and it was wrong in a way that is not
 * subtle: it deleted the LIVE, JUST-MATCHED track and kept the stale lost one.
 *
 * The reference ranks the two candidates by how long each has been alive AND tracked:
 *
 *     timep = tracked.frame_id - tracked.start_frame
 *     timeq = lost.frame_id    - lost.start_frame
 *     if timep > timeq: drop the lost one, else: drop the tracked one
 *
 * The trap is `frame_id`: it is written only in `activate()` / `re_activate()` /
 * `update()`, so for a LOST track it FREEZES at the frame it was last matched. `timeq` is
 * therefore the track's lifetime *as of when it went lost* — it excludes the lost span.
 * vestige compared `age`, which keeps incrementing every frame including while lost, so
 * the two quantities drift apart by exactly `timeSinceUpdate` and the comparison inverts.
 *
 * The sequence below is built to make that inversion unavoidable: both members of each
 * pair spawn on the SAME frame, so their `age` is EQUAL, `c.age > l.age` is false, and the
 * live track loses. The correct expression is `age - timeSinceUpdate` on both sides, which
 * reconstructs the reference's frozen `frame_id - start_frame` exactly.
 *
 * See `packages/core/fixtures/bytetrack-dedup/gen.py` and ADR-0003 §7.
 */
import { describe, expect, it } from 'vitest';
import data from '../../fixtures/bytetrack-dedup/data.json' with { type: 'json' };
import { ByteTracker } from '../../src/trackers/bytetrack.js';
import type { BBox, Detection, Track } from '../../src/types.js';

interface FixtureFrame {
  frame_index: number;
  detections: number[][];
  tracks_out: number[][];
}

interface FixtureEnvelope {
  generator: { foundationvision_sha: string };
  bytetrack_config: {
    track_thresh: number;
    track_buffer: number;
    match_thresh: number;
    mot20: boolean;
    frame_rate: number;
  };
  sequence: { twin_iou: number };
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

describe(`ByteTracker — duplicate removal (sha ${envelope.generator.foundationvision_sha.slice(0, 8)})`, () => {
  it(`reproduces the reference across all ${envelope.frames.length} frames with near-duplicate twins (IoU ${envelope.sequence.twin_iou})`, () => {
    const c = envelope.bytetrack_config;
    const tracker = new ByteTracker({
      trackThresh: c.track_thresh,
      trackBuffer: c.track_buffer,
      matchThresh: c.match_thresh,
      frameRate: c.frame_rate,
      mot20: c.mot20,
    });

    const idMap = new Map<number, number>();
    const reverseIdMap = new Map<number, number>();

    for (const fixture of envelope.frames) {
      const out: Track[] = tracker.update(fixture.detections.map(detFromRow));
      const refRows = fixture.tracks_out;

      // The single most diagnostic assertion here. When dedup drops the wrong member,
      // vestige emits FEWER tracks than the reference on the frame the live twin is
      // deleted — it silently loses a track that was matched to a real detection.
      expect(
        out.length,
        `frame ${fixture.frame_index}: output count mismatch (reference=${refRows.length}, vestige=${out.length}). A shortfall here means duplicate removal deleted a LIVE, just-matched track.`,
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
          `frame ${fixture.frame_index}: vestige id=${track.id} bbox=[${track.bbox.map((v) => v.toFixed(1)).join(',')}] has no reference bbox within tolerance`,
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
