import { describe, expect, it } from 'vitest';
import { parseMotChallenge } from '../../src/motchallenge/parse.js';
import {
  MOT17_DISTRACTOR_CLASS_IDS,
  MOT20_DISTRACTOR_CLASS_IDS,
  preprocessMotSequence,
} from '../../src/motchallenge/preprocess.js';

// Hand-traced scenarios against TrackEval's get_preprocessed_seq_data
// (mot_challenge_2d_box.py) — the CONTRIBUTING.md §4.2 reference. The
// trackeval-preproc fixture covers the full pipeline; these isolate one
// branch each.

const rows = (lines: string[]) => parseMotChallenge(lines.join('\n'));

describe('preprocessMotSequence', () => {
  it('removes tracker detections matched to distractor-class gt', () => {
    // Frame 1: pedestrian gt 1 and static_person (class 7) gt 3, each with a
    // tracker box at IoU ≈ 0.9. The distractor's follower must be removed;
    // the distractor itself never appears in the eval gt.
    const gt = rows(['1,1,100,100,60,180,1,1,1', '1,3,800,400,50,160,1,7,1']);
    const tracker = rows(['1,101,103,102,60,180,0.95', '1,104,802,402,50,160,0.85']);

    const [frame] = preprocessMotSequence(gt, tracker);
    expect(frame?.gtIds).toEqual([1]);
    expect(frame?.trackIds).toEqual([101]);
  });

  it('drops zero-marked gt without removing its matched tracker detection', () => {
    // gt 4 is a pedestrian with consider-flag 0: it leaves the eval gt, but
    // its follower stays (and will count as FP downstream) — zero-marked is
    // not a distractor.
    const gt = rows(['1,4,1200,500,60,180,0,1,0.6']);
    const tracker = rows(['1,105,1203,502,60,180,0.8']);

    const [frame] = preprocessMotSequence(gt, tracker);
    expect(frame?.gtIds).toEqual([]);
    expect(frame?.trackIds).toEqual([105]);
  });

  it('drops valid non-evaluated classes without removing tracker detections', () => {
    // A car (class 3) is annotated and considered, but only pedestrians are
    // evaluated: the gt row is dropped and the follower is kept.
    const gt = rows(['1,5,1500,600,120,80,1,3,1']);
    const tracker = rows(['1,107,1502,602,120,80,0.75']);

    const [frame] = preprocessMotSequence(gt, tracker);
    expect(frame?.gtIds).toEqual([]);
    expect(frame?.trackIds).toEqual([107]);
  });

  it('keeps tracker detections whose distractor overlap is below simThreshold', () => {
    // Distractor box [0,0,10,10] vs tracker [4,0,14,10]: IoU = 3/7 ≈ 0.43,
    // below the 0.5 gate → the pair never matches, nothing is removed.
    const gt = rows(['1,3,0,0,10,10,1,8,1']);
    const tracker = rows(['1,101,4,0,10,10,0.9']);

    const [frame] = preprocessMotSequence(gt, tracker);
    expect(frame?.trackIds).toEqual([101]);
  });

  it('assigns each tracker detection optimally before removal', () => {
    // One tracker box overlaps a pedestrian at 0.8546 and a distractor
    // (class 8) at 0.7915 — both above threshold. The assignment pairs it
    // with the pedestrian, so it is NOT removed.
    const gt = rows(['1,1,0,0,100,100,1,1,1', '1,3,10,10,100,100,1,8,1']);
    const tracker = rows(['1,101,4,4,100,100,0.9']);

    const [frame] = preprocessMotSequence(gt, tracker);
    expect(frame?.gtIds).toEqual([1]);
    expect(frame?.trackIds).toEqual([101]);
  });

  it('honors the MOT20 distractor list (non_mot_vehicle)', () => {
    // class 6 is a distractor only on MOT20.
    const gt = rows(['1,9,100,100,60,180,1,6,1']);
    const tracker = rows(['1,101,102,102,60,180,0.9']);

    const mot17 = preprocessMotSequence(gt, tracker, {
      distractorClassIds: MOT17_DISTRACTOR_CLASS_IDS,
    });
    const mot20 = preprocessMotSequence(gt, tracker, {
      distractorClassIds: MOT20_DISTRACTOR_CLASS_IDS,
    });
    expect(mot17[0]?.trackIds).toEqual([101]);
    expect(mot20[0]?.trackIds).toEqual([]);
  });

  it('throws on invalid gt classes — but only on frames with tracker output', () => {
    // TrackEval validates classes inside the matching branch: a frame with an
    // empty tracker side is not validated. Mirror that exactly.
    const badGt = rows(['1,1,0,0,10,10,1,14,1']);
    expect(() => preprocessMotSequence(badGt, rows(['1,101,0,0,10,10,0.9']))).toThrow(
      /invalid.*class/i,
    );
    expect(() => preprocessMotSequence(badGt, [])).not.toThrow();
  });

  it('materializes numFrames frames, defaulting to the max frame present', () => {
    const gt = rows(['3,1,0,0,10,10,1,1,1']);
    expect(preprocessMotSequence(gt, [])).toHaveLength(3);
    expect(preprocessMotSequence(gt, [], { numFrames: 5 })).toHaveLength(5);
    expect(preprocessMotSequence(gt, [], { numFrames: 5 })[4]?.gtIds).toEqual([]);
  });

  it('preserves within-frame entry order on both sides', () => {
    const gt = rows(['1,2,0,0,10,10,1,1,1', '1,1,50,0,10,10,1,1,1']);
    const tracker = rows(['1,202,1,0,10,10,0.9', '1,201,51,0,10,10,0.9']);

    const [frame] = preprocessMotSequence(gt, tracker);
    expect(frame?.gtIds).toEqual([2, 1]);
    expect(frame?.trackIds).toEqual([202, 201]);
    expect(frame?.gtBoxes[0]).toEqual([0, 0, 10, 10]);
  });
});
