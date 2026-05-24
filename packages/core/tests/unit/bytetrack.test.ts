import { describe, expect, it } from 'vitest';
import { ByteTracker } from '../../src/trackers/bytetrack.js';
import type { BBox, Detection } from '../../src/types.js';

function det(bbox: BBox, score = 0.9, classId?: number): Detection {
  return classId === undefined ? { bbox, score } : { bbox, score, classId };
}

/**
 * Drive a `ByteTracker` for N frames, returning the user-facing track list
 * from each frame. `detsByFrame[t]` is the detection list for frame t.
 */
function run(tracker: ByteTracker, detsByFrame: ReadonlyArray<ReadonlyArray<Detection>>) {
  return detsByFrame.map((d) => tracker.update(d));
}

describe('ByteTracker — defaults', () => {
  it('trackThresh defaults to 0.5', () => {
    expect(new ByteTracker().trackThresh).toBeCloseTo(0.5, 12);
  });

  it('trackBuffer defaults to 30', () => {
    expect(new ByteTracker().trackBuffer).toBe(30);
  });

  it('matchThresh defaults to 0.8', () => {
    expect(new ByteTracker().matchThresh).toBeCloseTo(0.8, 12);
  });

  it('frameRate defaults to 30', () => {
    expect(new ByteTracker().frameRate).toBe(30);
  });

  it('mot20 defaults to false', () => {
    expect(new ByteTracker().mot20).toBe(false);
  });

  it('detThresh = trackThresh + 0.1', () => {
    expect(new ByteTracker().detThresh).toBeCloseTo(0.6, 12);
    expect(new ByteTracker({ trackThresh: 0.7 }).detThresh).toBeCloseTo(0.8, 12);
  });

  it('maxAge = floor(frameRate/30 * trackBuffer)', () => {
    expect(new ByteTracker({ frameRate: 30, trackBuffer: 30 }).maxAge).toBe(30);
    expect(new ByteTracker({ frameRate: 60, trackBuffer: 30 }).maxAge).toBe(60);
    expect(new ByteTracker({ frameRate: 15, trackBuffer: 30 }).maxAge).toBe(15);
    expect(new ByteTracker({ frameRate: 25, trackBuffer: 30 }).maxAge).toBe(25); // floor(0.833 * 30)
  });
});

describe('ByteTracker — frame 1 instant activation', () => {
  it('emits the first frame detection immediately (mirrors STrack.activate(frame_id=1))', () => {
    const t = new ByteTracker();
    const out = t.update([det([10, 20, 110, 220], 0.9)]);
    expect(out).toHaveLength(1);
    expect(out[0]?.id).toBe(1);
    expect(out[0]?.state).toBe('confirmed');
  });

  it('assigns sequential ids on the first frame in detection order', () => {
    const t = new ByteTracker();
    const out = t.update([det([0, 0, 10, 10], 0.9), det([100, 100, 200, 200], 0.9)]);
    const ids = out.map((tr) => tr.id).sort((a, b) => a - b);
    expect(ids).toEqual([1, 2]);
  });
});

describe('ByteTracker — single-chance tentative on frames ≥ 2', () => {
  it('a track spawned on frame 2 is NOT output that frame (tentative)', () => {
    const t = new ByteTracker();
    // Frame 1 already establishes the tracker is past frame 1.
    t.update([det([0, 0, 50, 50], 0.9)]);
    // Frame 2: spawn a brand-new detection in a new location → tentative, not output.
    const out = t.update([det([0, 0, 50, 50], 0.9), det([500, 500, 600, 600], 0.9)]);
    const ids = out.map((tr) => tr.id).sort((a, b) => a - b);
    // Only the frame-1 track (id=1) appears; the new id=2 is tentative.
    expect(ids).toEqual([1]);
  });

  it('a tentative track confirms on its very next match (one-chance rule)', () => {
    const t = new ByteTracker();
    t.update([det([0, 0, 50, 50], 0.9)]); // frame 1: id=1 confirmed
    t.update([det([0, 0, 50, 50], 0.9), det([500, 500, 600, 600], 0.9)]); // frame 2: id=2 tentative
    const out = t.update([det([0, 0, 50, 50], 0.9), det([500, 500, 600, 600], 0.9)]); // frame 3
    const ids = out.map((tr) => tr.id).sort((a, b) => a - b);
    expect(ids).toEqual([1, 2]);
    expect(out.find((tr) => tr.id === 2)?.state).toBe('confirmed');
  });

  it('a tentative track that fails to match on its next frame is removed immediately', () => {
    const t = new ByteTracker();
    t.update([det([0, 0, 50, 50], 0.9)]); // frame 1: id=1
    t.update([det([0, 0, 50, 50], 0.9), det([500, 500, 600, 600], 0.9)]); // frame 2: id=2 tentative
    // Frame 3: id=2's region has no detection. Tentative → removed.
    t.update([det([0, 0, 50, 50], 0.9)]);
    // Frame 4: a brand-new detection at the old (500..600) region should get id=3, not id=2.
    const out = t.update([det([0, 0, 50, 50], 0.9), det([500, 500, 600, 600], 0.9)]);
    const ids = out.map((tr) => tr.id).sort((a, b) => a - b);
    // id=2 must not reappear (it was removed); id=3 is the new tentative (not output).
    expect(ids).toEqual([1]);
  });
});

describe('ByteTracker — stage 2 low-score association', () => {
  it('a confirmed track that only has a LOW-score detection this frame still matches via stage 2', () => {
    const t = new ByteTracker({ trackThresh: 0.5 });
    // Frame 1: id=1 confirmed at high score.
    const r1 = t.update([det([0, 0, 100, 100], 0.9)]);
    expect(r1).toHaveLength(1);
    // Frame 2: the box is now low-confidence (e.g. partial occlusion). Stage 1
    // sees no high-score det, stage 2 picks the low-score one up.
    const r2 = t.update([det([0, 0, 100, 100], 0.3)]);
    expect(r2).toHaveLength(1);
    expect(r2[0]?.id).toBe(1);
    expect(r2[0]?.score).toBeCloseTo(0.3, 12);
  });

  it('keeps an id stable across a low-score occlusion frame', () => {
    const t = new ByteTracker();
    const settled = det([0, 0, 100, 100], 0.9);
    const occluded = det([0, 0, 100, 100], 0.2); // below trackThresh
    t.update([settled]); // frame 1: id=1
    t.update([settled]); // frame 2
    t.update([settled]); // frame 3
    const r = t.update([occluded]); // frame 4: low-score → stage 2
    expect(r).toHaveLength(1);
    expect(r[0]?.id).toBe(1);
  });

  it('does not spawn a new track from a low-score detection (det_thresh gate)', () => {
    const t = new ByteTracker(); // detThresh = 0.6
    const out = t.update([det([0, 0, 50, 50], 0.55)]);
    // 0.55 > trackThresh=0.5 so it IS high-score, but 0.55 < detThresh=0.6 so
    // it cannot spawn a track. With no existing tracks, output is empty.
    expect(out).toEqual([]);
  });
});

describe('ByteTracker — low-score floor (≤ 0.1 discarded)', () => {
  it('detections with score ≤ 0.1 do not participate in any stage', () => {
    const t = new ByteTracker();
    // Frame 1: confirmed track at id=1.
    t.update([det([0, 0, 100, 100], 0.9)]);
    // Frame 2: only a score=0.05 detection. Discarded entirely; track misses → lost.
    const out = t.update([det([0, 0, 100, 100], 0.05)]);
    expect(out).toEqual([]);
    // The track should now be `lost` (still in the map, retained for trackBuffer).
    expect(t.getLostTracks()).toHaveLength(1);
    expect(t.getLostTracks()[0]?.id).toBe(1);
  });
});

describe('ByteTracker — lost-track retention (trackBuffer)', () => {
  it('keeps a lost track for `maxAge` frames and re-acquires its id', () => {
    const t = new ByteTracker({ trackBuffer: 5, frameRate: 30 }); // maxAge=5
    const d = det([0, 0, 100, 100], 0.9);
    t.update([d]); // frame 1: id=1 confirmed
    t.update([d]); // frame 2: still confirmed
    t.update([]); // frame 3: miss 1 → lost
    t.update([]); // frame 4: miss 2 → lost
    expect(t.getLostTracks()).toHaveLength(1);
    const out = t.update([d]); // frame 5: re-match
    expect(out).toHaveLength(1);
    expect(out[0]?.id).toBe(1);
  });

  it('removes a lost track once tsu > maxAge', () => {
    const t = new ByteTracker({ trackBuffer: 2, frameRate: 30 }); // maxAge=2
    const d = det([0, 0, 100, 100], 0.9);
    t.update([d]); // frame 1: id=1
    t.update([d]); // frame 2
    t.update([]); // miss 1 → lost, tsu=1
    t.update([]); // miss 2 → lost, tsu=2
    t.update([]); // miss 3 → tsu=3 > maxAge=2 → removed
    expect(t.getActiveTracks()).toHaveLength(0);
    expect(t.getLostTracks()).toHaveLength(0);
  });
});

describe('ByteTracker — two non-overlapping detections stay distinct', () => {
  it('two far-apart detections produce two distinct, stable ids', () => {
    const t = new ByteTracker();
    const a = det([0, 0, 50, 50], 0.9);
    const b = det([500, 500, 600, 600], 0.9);
    const frames = run(t, [
      [a, b],
      [a, b],
      [a, b],
      [a, b],
      [a, b],
    ]);
    for (const f of frames) {
      expect(f).toHaveLength(2);
      expect(new Set(f.map((tr) => tr.id)).size).toBe(2);
    }
    const idsByFrame = frames.map((f) =>
      [...f].sort((p, q) => p.bbox[0] - q.bbox[0]).map((tr) => tr.id),
    );
    for (let i = 1; i < idsByFrame.length; i++) {
      expect(idsByFrame[i]).toEqual(idsByFrame[0]);
    }
  });
});

describe('ByteTracker — matchThresh gating', () => {
  it('rejects stage-1 matches whose IoU is below the threshold', () => {
    // matchThresh=0.3 (cost ≤ 0.3, i.e. IoU ≥ 0.7) → a disjoint det cannot rescue the track.
    const t = new ByteTracker({ matchThresh: 0.3, trackBuffer: 30 });
    t.update([det([0, 0, 100, 100], 0.9)]); // frame 1: id=1 confirmed
    // Frame 2: a high-score detection nowhere near id=1. Stage 1 gated out;
    // id=1 misses → lost. New detection above detThresh → tentative id=2.
    t.update([det([1000, 1000, 1100, 1100], 0.9)]);
    expect(t.getLostTracks().map((tr) => tr.id)).toEqual([1]);
  });
});

describe('ByteTracker — payload + classId + score preserved', () => {
  it('payload flows through update()', () => {
    type P = { label: string };
    const t = new ByteTracker<P>();
    const out = t.update([{ bbox: [0, 0, 10, 10], score: 0.9, payload: { label: 'cat' } }]);
    expect(out[0]?.payload).toEqual({ label: 'cat' });
  });

  it('classId flows through update()', () => {
    const t = new ByteTracker();
    const out = t.update([det([0, 0, 10, 10], 0.9, 7)]);
    expect(out[0]?.classId).toBe(7);
  });

  it('score reflects the most-recent matched detection', () => {
    const t = new ByteTracker();
    t.update([det([0, 0, 100, 100], 0.9)]);
    const out = t.update([det([0, 0, 100, 100], 0.7)]);
    expect(out[0]?.score).toBeCloseTo(0.7, 12);
  });
});

describe('ByteTracker — degenerate inputs', () => {
  it('handles empty detections (no crash, no output)', () => {
    const t = new ByteTracker();
    expect(t.update([])).toEqual([]);
    expect(t.frameIndex).toBe(1);
  });

  it('handles many empty frames in a row', () => {
    const t = new ByteTracker();
    for (let i = 0; i < 100; i++) t.update([]);
    expect(t.frameIndex).toBe(100);
    expect(t.getActiveTracks()).toHaveLength(0);
  });

  it('handles a frame with only low-score detections (all dropped)', () => {
    const t = new ByteTracker();
    const out = t.update([det([0, 0, 50, 50], 0.05), det([100, 100, 200, 200], 0.08)]);
    expect(out).toEqual([]);
    expect(t.frameIndex).toBe(1);
  });
});

describe('ByteTracker — Track shape', () => {
  it('reports timeSinceUpdate=0 for tracks matched this frame', () => {
    const t = new ByteTracker();
    const out = t.update([det([0, 0, 100, 100], 0.9)]);
    expect(out[0]?.timeSinceUpdate).toBe(0);
  });

  it('reports state="confirmed" on the very first frame of a track`s life', () => {
    // Frame-1 deviation from BaseTracker — see class JSDoc.
    const t = new ByteTracker();
    const out = t.update([det([0, 0, 100, 100], 0.9)]);
    expect(out[0]?.state).toBe('confirmed');
  });

  it('age increments every frame the track is alive', () => {
    const t = new ByteTracker({ trackBuffer: 30 });
    const d = det([0, 0, 100, 100], 0.9);
    t.update([d]); // age=0
    t.update([d]); // age=1
    const out = t.update([d]); // age=2
    expect(out[0]?.age).toBe(2);
  });
});

describe('ByteTracker.reset', () => {
  it('clears tracks, restarts ids at 1, resets frame counter', () => {
    const t = new ByteTracker();
    t.update([det([0, 0, 10, 10], 0.9), det([100, 100, 200, 200], 0.9)]);
    t.reset();
    expect(t.frameIndex).toBe(0);
    const out = t.update([det([0, 0, 10, 10], 0.9)]);
    expect(out[0]?.id).toBe(1);
  });
});
