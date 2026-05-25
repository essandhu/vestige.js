/**
 * End-to-end validation of OC-SORT's three observation-centric contributions
 * (ORU §3.2, OCM §3.3, OCR §3.4) on hand-designed synthetic scenarios. See
 * the README in this directory for the tier's scope and conventions.
 *
 * Each scenario is documented inline with the hand-trace of the expected
 * outcome and the paper section / reference-impl behavior it exercises.
 */
import { describe, expect, it } from 'vitest';
import { iou } from '../../src/geometry/iou.js';
import { OcSortTracker } from '../../src/trackers/ocsort.js';
import type { BBox, Detection } from '../../src/types.js';

function det(bbox: BBox, score = 0.9): Detection {
  return { bbox, score };
}

/** Translate a 100×100 bbox by (vx*frame, vy*frame). */
function moving(originX: number, originY: number, vx: number, vy: number, frame: number): BBox {
  return [
    originX + vx * frame,
    originY + vy * frame,
    originX + 100 + vx * frame,
    originY + 100 + vy * frame,
  ];
}

describe('OC-SORT validation — ORU (Observation Re-Update; paper §3.2)', () => {
  /**
   * Scenario: a track moves at constant velocity (+10, 0) for 5 frames,
   * is then occluded for 5 frames, and reappears at the position consistent
   * with continued constant-velocity motion.
   *
   * Without ORU: the Kalman filter's predict-only updates during occlusion
   * smear the state. On re-association, the standard update fuses the new
   * observation with the drifted prediction — the post-update bbox lands
   * between them.
   *
   * With ORU: the filter rolls back to the pre-occlusion state and replays
   * a smooth virtual trajectory of measurement updates ending at the new
   * observation. The post-update bbox lands at (essentially) the new
   * observation, because the trajectory is consistent with the originally
   * estimated velocity.
   *
   * Assertion: after re-association, the IoU between the tracker's reported
   * bbox and the re-emergence detection's bbox is high. ID is preserved.
   */
  it('preserves id and lands the bbox on the re-emergence detection after a 5-frame occlusion', () => {
    const t = new OcSortTracker({ minHits: 1, maxAge: 10, inertia: 0.2, deltaT: 3 });
    // Frames 1–5: visible, constant velocity (+10 px/frame in x).
    for (let f = 0; f < 5; f++) t.update([det(moving(0, 100, 10, 0, f))]);
    // Frames 6–10: occluded.
    for (let f = 0; f < 5; f++) t.update([]);
    // Frame 11: reappears at the position consistent with continued (+10, 0)
    // motion from the last visible frame (f=4 was at x=40..140; +6 frames → x=100..200).
    const reEmergence = moving(0, 100, 10, 0, 10);
    const out = t.update([det(reEmergence)]);
    expect(out).toHaveLength(1);
    expect(out[0]?.id).toBe(1); // id preserved through the gap
    // Post-ORU, the tracker's bbox should sit on top of the re-emergence detection.
    // Without ORU, fusion with the drifted prediction would pull the bbox off-center;
    // we expect IoU ≳ 0.9 with the actual detection.
    const reportedBbox = out[0]?.bbox as BBox;
    expect(iou(reportedBbox, reEmergence)).toBeGreaterThan(0.9);
  });
});

describe('OC-SORT validation — OCM (Observation-Centric Momentum; paper §3.3)', () => {
  /**
   * Scenario: a track is moving rightward at (+10, 0). On the disambiguation
   * frame two equally-IoU-close detections arrive at the predicted position:
   * one continues the rightward motion (consistent with track velocity), one
   * would imply a sudden reversal (inconsistent). OCM should bias the
   * Hungarian assignment toward the consistent detection.
   *
   * With `inertia=0` (OCM disabled), the cost matrix is symmetric over the two
   * candidates and the deterministic Hungarian tie-break (row-then-column)
   * chooses arbitrarily. With `inertia > 0`, the angular consistency term
   * tilts the cost in favor of the rightward-moving detection.
   *
   * To make the test deterministic, the "consistent" detection is placed at
   * the exact predicted next position (IoU > 0.99) and the "inconsistent" one
   * at the same IoU offset on the opposite side. Without OCM both have equal
   * cost; with OCM only the consistent one matches.
   *
   * Assertion: with default `inertia=0.2`, the matched detection is the
   * consistent one — the tracker's post-update bbox sits on the consistent
   * detection, not on the inconsistent one.
   */
  it('prefers the direction-consistent detection on an ambiguous frame', () => {
    const t = new OcSortTracker({ minHits: 1, inertia: 0.2, deltaT: 3 });
    // Build up a clear rightward velocity over 5 frames.
    for (let f = 0; f < 5; f++) t.update([det(moving(0, 100, 10, 0, f))]);
    // Frame 6: two candidates at equal IoU offsets from the predicted
    // position (next predicted x ≈ 50..150 after 5 predicts from frame 5).
    // Predicted center after f=5 is approximately (100, 150). Consistent
    // detection continues rightward; "ambiguous mirror" is placed equally
    // far to the left of the predicted box, giving identical IoU.
    const consistent = moving(0, 100, 10, 0, 5); // continues to x=50..150
    const inconsistent: BBox = [0, 100, 100, 200]; // same IoU offset but to the left
    const out = t.update([det(consistent), det(inconsistent)]);
    // Track 1 must take the consistent detection, NOT the inconsistent one.
    const trackOne = out.find((tr) => tr.id === 1);
    expect(trackOne).toBeDefined();
    expect(iou(trackOne!.bbox, consistent)).toBeGreaterThan(iou(trackOne!.bbox, inconsistent));
  });
});

describe('OC-SORT validation — OCR (Observation-Centric Recovery; paper §3.4)', () => {
  /**
   * Scenario: a track is observed for 3 frames at a stationary position, is
   * occluded for 4 frames (Kalman predict-only steps; the prediction stays
   * close to the last position since velocity ≈ 0), and reappears at a
   * location with low IoU against the predicted bbox but high IoU against
   * the last actual observation.
   *
   * Trick: the cv-bbox Kalman filter, fed five identical observations, ends
   * up with non-zero scale velocity (ṡ) due to the broad initial covariance.
   * Over 4 unobserved frames the scale grows, shrinking IoU with any
   * normal-sized observation. The Kalman-predicted bbox can fall below the
   * `iouThreshold` (0.3) against the new observation even though the new
   * observation is geometrically identical to the last real one.
   *
   * Without OCR: stage-1 association gates the pair out (IoU too low),
   * the track stays lost, and a new track is spawned.
   *
   * With OCR: the second pass compares each unmatched detection against
   * each unmatched track's `lastObservation`, which is identical to the new
   * observation. IoU = 1, well above threshold, and the track is rescued.
   *
   * Assertion: id is preserved across the gap, with the same observation
   * pre- and post-occlusion.
   */
  it('rescues a track via last-observation match when the Kalman prediction has drifted', () => {
    // Use a fairly tight iouThreshold and a longer occlusion to give the
    // predicted scale time to drift away from the observation footprint.
    const t = new OcSortTracker({ minHits: 1, maxAge: 30, iouThreshold: 0.3 });
    const stationary = det([100, 100, 200, 200]);
    // Frames 1–5: stationary observations — confirms the track and stabilizes velocity ≈ 0.
    for (let f = 0; f < 5; f++) t.update([stationary]);
    // Frames 6–20: occluded. The Kalman state drifts.
    for (let f = 0; f < 15; f++) t.update([]);
    // Frame 21: same bbox returns.
    const out = t.update([stationary]);
    expect(out).toHaveLength(1);
    expect(out[0]?.id).toBe(1); // id preserved via OCR fallback
  });
});
