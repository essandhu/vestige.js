import { describe, expect, it } from 'vitest';
import type { Detection, Track } from '../../src/core.js';
import { SortTracker } from '../../src/core.js';
import { clearMot } from '../../src/metrics/clearmot.js';
import { hota } from '../../src/metrics/hota.js';
import { identity } from '../../src/metrics/identity.js';
import { formatMotChallenge, parseMotChallenge } from '../../src/motchallenge/parse.js';
import { evalFramesFromEntries, runTracker, tracksToMotEntries } from '../../src/runner.js';

const det = (x: number, y: number, w: number, h: number, score = 0.9): Detection => ({
  bbox: [x, y, x + w, y + h],
  score,
});

const fakeTrack = (id: number, x: number, y: number, w: number, h: number): Track => ({
  id,
  bbox: [x, y, x + w, y + h],
  score: 0.9,
  age: 1,
  hits: 1,
  timeSinceUpdate: 0,
  state: 'confirmed',
});

describe('runTracker', () => {
  it('produces one output frame per input frame', () => {
    // A stationary detection through a default SortTracker: the warmup export
    // rule (sort.py `frame_count <= min_hits`) emits the track from frame 1,
    // and a zero-innovation Kalman update keeps the bbox exactly stationary.
    const frames = Array.from({ length: 3 }, () => [det(10, 20, 30, 40)]);
    const out = runTracker(new SortTracker(), frames);

    expect(out).toHaveLength(3);
    for (const tracks of out) {
      expect(tracks).toHaveLength(1);
      expect(tracks[0]?.id).toBe(1);
      const bbox = tracks[0]?.bbox ?? [0, 0, 0, 0];
      expect(bbox[0]).toBeCloseTo(10, 9);
      expect(bbox[1]).toBeCloseTo(20, 9);
      expect(bbox[2]).toBeCloseTo(40, 9);
      expect(bbox[3]).toBeCloseTo(60, 9);
    }
  });

  it('passes empty detection frames through to the tracker', () => {
    const out = runTracker(new SortTracker(), [[det(0, 0, 10, 10)], [], []]);
    expect(out).toHaveLength(3);
    expect(out[1]).toEqual([]);
  });
});

describe('tracksToMotEntries', () => {
  it('converts per-frame tracks to 1-based MOT entries in ltwh form', () => {
    const entries = tracksToMotEntries([
      [fakeTrack(1, 10, 20, 30, 40)],
      [],
      [fakeTrack(1, 12, 22, 30, 40), fakeTrack(2, 50, 60, 10, 10)],
    ]);

    expect(entries).toHaveLength(3);
    expect(entries[0]).toEqual({
      frame: 1,
      id: 1,
      bbox: [10, 20, 40, 60],
      score: 0.9,
      classId: -1,
      visibility: -1,
    });
    expect(entries[1]?.frame).toBe(3);
    expect(entries[2]?.frame).toBe(3);
    expect(entries[2]?.id).toBe(2);
  });

  it('formats to the MOTChallenge result-file form end to end', () => {
    const text = formatMotChallenge(tracksToMotEntries([[fakeTrack(1, 10, 20, 30, 40)]]));
    expect(text).toBe('1,1,10,20,30,40,0.9,-1,-1,-1\n');
  });
});

describe('evalFramesFromEntries', () => {
  it('aligns gt and tracker entries by frame, including one-sided frames', () => {
    const gt = parseMotChallenge('1,1,0,0,10,10,1,1,1\n3,1,5,0,10,10,1,1,1\n');
    const tracker = parseMotChallenge('2,9,0,0,10,10,0.9\n3,9,5,0,10,10,0.9\n');

    const frames = evalFramesFromEntries(gt, tracker);
    expect(frames).toHaveLength(3);
    expect(frames[0]?.gtIds).toEqual([1]);
    expect(frames[0]?.trackIds).toEqual([]);
    expect(frames[1]?.gtIds).toEqual([]);
    expect(frames[1]?.trackIds).toEqual([9]);
    expect(frames[2]?.gtIds).toEqual([1]);
    expect(frames[2]?.trackIds).toEqual([9]);
    expect(frames[2]?.gtBoxes[0]).toEqual([5, 0, 15, 10]);
  });

  it('honors an explicit numFrames with trailing empty frames', () => {
    const gt = parseMotChallenge('1,1,0,0,10,10,1,1,1\n');
    const frames = evalFramesFromEntries(gt, [], 3);
    expect(frames).toHaveLength(3);
    expect(frames[2]?.gtIds).toEqual([]);
    expect(frames[2]?.trackIds).toEqual([]);
  });

  it('feeds the metrics: a tracker that echoes gt scores perfectly', () => {
    // End-to-end glue check: gt of 2 objects × 3 frames, tracker output equal
    // to gt with its own ids. Every metric should report a perfect score.
    const gtText = [1, 2, 3]
      .flatMap((f) => [`${f},1,0,0,10,10,1,1,1`, `${f},2,50,0,10,10,1,1,1`])
      .join('\n');
    const trText = [1, 2, 3]
      .flatMap((f) => [`${f},8,0,0,10,10,0.9`, `${f},9,50,0,10,10,0.9`])
      .join('\n');
    const frames = evalFramesFromEntries(parseMotChallenge(gtText), parseMotChallenge(trText));

    expect(clearMot(frames).mota).toBe(1);
    expect(identity(frames).idf1).toBe(1);
    expect(hota(frames).hota).toBe(1);
  });
});
