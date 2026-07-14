/**
 * Pins vestige's behavior for OC-SORT's OCR stage when a track has NEVER been observed,
 * where it DELIBERATELY diverges from `noahcao/OC_SORT`.
 *
 * `KalmanBoxTracker.last_observation` is initialized to the placeholder
 * `[-1, -1, -1, -1, -1]` and is only written on a track's first MATCH — so a track that
 * spawned but was never matched again has no observation at all. The reference's OCR
 * stage builds its track side from `last_boxes[unmatched_trks]` (`ocsort.py:281`), which
 * includes those never-observed tracks **at their placeholder box**, and then scores them
 * with `asso_func`.
 *
 * Under `iou` the placeholder scores 0.0000 against everything (it has zero area), so it
 * can never match and the question never arises. That is why the default config never
 * exposed this. Under `giou` the score is driven by the ENCLOSING box, and a placeholder
 * sitting at (-1, -1) is *near the image origin*:
 *
 *     giou(placeholder, det @ [2,2,82,82])       = 0.4645   -> matches (> 0.3)
 *     giou(placeholder, det @ [300,300,380,380]) = 0.0220
 *     giou(placeholder, det @ [600,600,680,680]) = 0.0069
 *
 * So under giou the reference will "recover" a never-observed track using a detection it
 * has never seen, purely because a placeholder happens to sit near the origin. Executed
 * against the reference at the pinned commit — a track spawned at (600,600), then a lone
 * detection at (2,2) on the next frame:
 *
 *     frame 1: det [600,600,680,680] -> id 1 @ [600,600,680,680]
 *     frame 2: det [2,2,82,82]       -> id 1 @ [2,2,82,82]      <- teleports 840 px
 *
 * The track jumps across the entire frame and keeps its id. We do not reproduce this.
 * OCR exists to recover a track from its last real observation; a track with no
 * observation has nothing to recover FROM, and matching against the placeholder is
 * matching against garbage. vestige excludes never-observed tracks from the OCR pool, so
 * the detection spawns a new track instead.
 *
 * Same posture as `bytetrack-resurrection.test.ts`: the fixture
 * (`ocsort-giou-byte/`) validates where the two implementations agree and keeps its
 * detections out of an origin keep-out zone; this test is what makes the disagreement a
 * decision rather than an accident. If someone later "fixes" the OCR pool to include
 * never-observed tracks, this fails and points at ADR-0007.
 */
import { describe, expect, it } from 'vitest';
import { OcSortTracker } from '../../src/trackers/ocsort.js';
import type { BBox, Detection } from '../../src/types.js';

const det = (bbox: BBox): Detection => ({ bbox, score: 0.9 });

/** Far from the origin: spawns a track that is then never matched again. */
const STRAY: BBox = [600, 600, 680, 680];
/** Near the origin, where the reference's `[-1,-1,-1,-1]` placeholder scores 0.46 under giou. */
const ORIGIN: BBox = [2, 2, 82, 82];

describe('OcSortTracker — OCR never-observed sentinel (deliberate divergence)', () => {
  it('does NOT recover a never-observed track from an origin-adjacent detection under giou', () => {
    // minHits: 1 so tracks are exported immediately and the ids are easy to read.
    const tracker = new OcSortTracker({ asoFunc: 'giou', minHits: 1, iouThreshold: 0.3 });

    const f1 = tracker.update([det(STRAY)]);
    expect(f1.map((t) => t.id)).toEqual([1]);
    expect(f1[0]?.bbox).toEqual(STRAY);

    // The stray detection is gone. The only detection is near the origin, 840px away and
    // wholly unrelated to track 1 — which has never been observed, so it has no last
    // observation to match against.
    //
    // Reference: OCR scores track 1's PLACEHOLDER box against this detection (giou 0.46 >
    // 0.3), MATCHES, and teleports id 1 to [2,2,82,82] — emitting it immediately, because
    // the match bumps its hit_streak to 1.
    // vestige: track 1 is not eligible for OCR, so the detection SPAWNS a new track. A
    // freshly-spawned track has hitStreak = 0, so with minHits = 1 it is not exported on
    // its spawn frame — hence nothing at the origin yet on f2. That absence IS the
    // divergence: the reference emits a track here and we do not.
    const f2 = tracker.update([det(ORIGIN)]);
    expect(
      f2.some((t) => t.bbox[0] < 100),
      'nothing should appear at the origin on the spawn frame; the reference emits id 1 here',
    ).toBe(false);
    expect(
      f2.some((t) => t.id === 1 && t.bbox[0] < 100),
      'track 1 must not teleport',
    ).toBe(false);

    // One more frame: the newly-spawned track is matched, so it is exported — under a NEW
    // id. The reference is still emitting id 1 at this position.
    const f3 = tracker.update([det(ORIGIN)]);
    const atOrigin = f3.find((t) => t.bbox[0] < 100);
    expect(atOrigin, 'the origin detection should now produce a track').toBeDefined();
    expect(
      atOrigin?.id,
      'the origin detection must carry a NEW id, not the resurrected never-observed track 1',
    ).toBe(2);
  });

  it('is unaffected under the default iou asoFunc — the placeholder scores 0 there', () => {
    // Same sequence with `asoFunc: 'iou'`. Here the reference AGREES with us, because the
    // zero-area placeholder scores 0.0000 and cannot clear the threshold. This is why the
    // divergence hid for so long: every fixture ran the default.
    const tracker = new OcSortTracker({ asoFunc: 'iou', minHits: 1, iouThreshold: 0.3 });
    tracker.update([det(STRAY)]);
    tracker.update([det(ORIGIN)]);
    const f3 = tracker.update([det(ORIGIN)]);

    const atOrigin = f3.find((t) => t.bbox[0] < 100);
    expect(atOrigin?.id).toBe(2);
  });

  it('still recovers a track that HAS been observed (OCR is not disabled, just gated)', () => {
    // Guards against "fixing" the divergence by gutting OCR. A track with a real last
    // observation must still be recoverable by it: here the track is stationary, so its
    // Kalman prediction and its last observation coincide, and a detection displaced 120px
    // is invisible to the primary stage (iou 0) but reachable by OCR under giou
    // (normalized giou = 80/(80+120) = 0.400 > 0.3).
    const tracker = new OcSortTracker({ asoFunc: 'giou', minHits: 1, iouThreshold: 0.3 });
    const home: BBox = [600, 600, 680, 680];
    const displaced: BBox = [720, 600, 800, 680]; // +120px in x, no overlap with `home`

    tracker.update([det(home)]);
    tracker.update([det(home)]); // now genuinely observed
    const f3 = tracker.update([det(displaced)]);

    expect(
      f3.map((t) => t.id),
      'an OBSERVED track must still be recovered by OCR under giou, keeping its id',
    ).toEqual([1]);
    expect(f3[0]?.bbox).toEqual(displaced);
  });
});
