import { describe, expect, it } from 'vitest';
import { SortTracker } from '../../src/trackers/sort.js';
import type { BBox, Detection } from '../../src/types.js';

function det(bbox: BBox, score = 0.9, classId?: number): Detection {
  return classId === undefined ? { bbox, score } : { bbox, score, classId };
}

/**
 * Drive a `SortTracker` for N frames, returning the user-facing track list
 * from each frame. `detsByFrame[t]` is the detection list for frame t.
 */
function run(tracker: SortTracker, detsByFrame: ReadonlyArray<ReadonlyArray<Detection>>) {
  return detsByFrame.map((d) => tracker.update(d));
}

describe('SortTracker — defaults', () => {
  it('iouThreshold default is 0.3', () => {
    const t = new SortTracker();
    expect(t.iouThreshold).toBeCloseTo(0.3, 12);
  });

  it('respects overridden iouThreshold', () => {
    const t = new SortTracker({ iouThreshold: 0.5 });
    expect(t.iouThreshold).toBeCloseTo(0.5, 12);
  });
});

describe('SortTracker — first frame and ID assignment', () => {
  it('emits the first detection as a track with id=1 during the warmup', () => {
    const t = new SortTracker(); // minHits=3 by default → warmup covers frame 1
    const out = t.update([det([10, 20, 110, 220])]);
    expect(out).toHaveLength(1);
    expect(out[0]?.id).toBe(1);
  });

  it('assigns sequential ids in detection order', () => {
    const t = new SortTracker();
    const out = t.update([det([0, 0, 10, 10]), det([100, 100, 200, 200])]);
    const ids = out.map((tr) => tr.id).sort((a, b) => a - b);
    expect(ids).toEqual([1, 2]);
  });

  it('does not reuse ids after a track is removed', () => {
    const t = new SortTracker({ maxAge: 0, minHits: 1 });
    t.update([det([0, 0, 10, 10])]); // id=1 created and confirmed
    t.update([]); // miss → lost → removed (maxAge=0)
    const out = t.update([det([0, 0, 10, 10])]); // new track
    expect(out[0]?.id).toBe(2);
  });
});

describe('SortTracker — warmup output rule (sort.py frame_count <= min_hits)', () => {
  it('outputs every matched track during the first minHits frames', () => {
    const t = new SortTracker({ minHits: 3 });
    const det1 = det([10, 10, 50, 50]);
    // frame 1: new track outputs immediately (warmup)
    const r1 = t.update([det1]);
    expect(r1).toHaveLength(1);
    // frame 2: same detection matches the predicted track
    const r2 = t.update([det1]);
    expect(r2).toHaveLength(1);
    // frame 3: still in warmup; output continues
    const r3 = t.update([det1]);
    expect(r3).toHaveLength(1);
  });

  it('after warmup, tentative tracks created on a single frame are NOT output', () => {
    const t = new SortTracker({ minHits: 3 });
    // Run 3 frames of one detection to clear the warmup window.
    const settled = det([10, 10, 50, 50]);
    t.update([settled]);
    t.update([settled]);
    t.update([settled]); // track 1 now confirmed
    // Frame 4: new detection at a different location → new tentative track.
    const out = t.update([settled, det([200, 200, 240, 240])]);
    // Only the confirmed track is output; the new tentative one is not.
    expect(out).toHaveLength(1);
    expect(out[0]?.id).toBe(1);
  });
});

describe('SortTracker — stable detection keeps the same id', () => {
  it('a stationary detection keeps its id across 10 frames', () => {
    const t = new SortTracker();
    const d = det([100, 100, 200, 200]);
    const frames = run(
      t,
      Array.from({ length: 10 }, () => [d]),
    );
    const ids = frames.map((f) => f[0]?.id);
    expect(new Set(ids).size).toBe(1);
    expect(ids[0]).toBe(1);
  });

  it('a slowly moving detection keeps its id (Kalman tracks the motion)', () => {
    const t = new SortTracker();
    // Box translates by (2, 1) per frame; 100×100 box → IoU well above 0.3.
    const detsByFrame: Detection[][] = [];
    for (let f = 0; f < 10; f++) {
      detsByFrame.push([det([100 + 2 * f, 100 + f, 200 + 2 * f, 200 + f])]);
    }
    const frames = run(t, detsByFrame);
    const ids = frames.map((f) => f[0]?.id).filter((id): id is number => id !== undefined);
    expect(new Set(ids).size).toBe(1);
  });
});

describe('SortTracker — two non-overlapping detections stay distinct', () => {
  it('two far-apart detections produce two distinct, stable ids', () => {
    const t = new SortTracker();
    const a = det([0, 0, 50, 50]);
    const b = det([500, 500, 600, 600]);
    const frames = run(t, [
      [a, b],
      [a, b],
      [a, b],
      [a, b],
      [a, b],
    ]);
    // Within each frame: 2 distinct ids.
    for (const f of frames) {
      expect(f).toHaveLength(2);
      expect(new Set(f.map((tr) => tr.id)).size).toBe(2);
    }
    // Across frames: the id of each box is stable.
    const idsByFrame = frames.map((f) =>
      [...f].sort((p, q) => p.bbox[0] - q.bbox[0]).map((tr) => tr.id),
    );
    for (let i = 1; i < idsByFrame.length; i++) {
      expect(idsByFrame[i]).toEqual(idsByFrame[0]);
    }
  });
});

describe('SortTracker — iouThreshold gating', () => {
  it('does not match a detection whose IoU with every track is below the threshold', () => {
    const t = new SortTracker({ iouThreshold: 0.5, minHits: 1, maxAge: 30 });
    // Frame 1: spawn a track at (0, 0, 100, 100).
    t.update([det([0, 0, 100, 100])]);
    // Frame 2: detection at (200, 200, 300, 300) — zero IoU with the track.
    const out = t.update([det([200, 200, 300, 300])]);
    // The old track loses its match (→ lost) and a NEW track is spawned for
    // the new detection. The new one is tentative (won't appear in output
    // because warmup already passed), so the result depends on whether the
    // old confirmed track outputs. It missed → tsu=1 → not output.
    expect(out).toHaveLength(0);
  });
});

describe('SortTracker — lifecycle on missed frames', () => {
  it('confirmed track survives maxAge frames of misses, then disappears', () => {
    const t = new SortTracker({ minHits: 1, maxAge: 2 });
    t.update([det([0, 0, 10, 10])]); // confirmed
    t.update([]); // miss 1 → lost, tsu=1
    t.update([]); // miss 2 → lost, tsu=2
    t.update([]); // miss 3 → removed (tsu=3 > maxAge=2)
    expect(t.getActiveTracks()).toHaveLength(0);
    expect(t.getLostTracks()).toHaveLength(0);
  });

  it('confirmed → lost → re-found preserves the id', () => {
    const t = new SortTracker({ minHits: 1, maxAge: 10 });
    const d = det([0, 0, 100, 100]);
    t.update([d]); // confirmed id=1
    t.update([]); // → lost
    t.update([]); // still lost
    const out = t.update([d]); // re-found
    expect(out).toHaveLength(1);
    expect(out[0]?.id).toBe(1);
  });
});

describe('SortTracker — payload + classId preserved', () => {
  it('payload flows through update()', () => {
    type P = { label: string };
    const t = new SortTracker<P>({ minHits: 1 });
    const out = t.update([{ bbox: [0, 0, 10, 10], score: 0.9, payload: { label: 'cat' } }]);
    expect(out[0]?.payload).toEqual({ label: 'cat' });
  });

  it('classId flows through update()', () => {
    const t = new SortTracker({ minHits: 1 });
    const out = t.update([det([0, 0, 10, 10], 0.9, 7)]);
    expect(out[0]?.classId).toBe(7);
  });
});

describe('SortTracker — degenerate inputs', () => {
  it('handles empty detections (no crash, no output)', () => {
    const t = new SortTracker();
    expect(t.update([])).toEqual([]);
    expect(t.frameIndex).toBe(1);
  });

  it('handles many empty frames in a row', () => {
    const t = new SortTracker();
    for (let i = 0; i < 100; i++) t.update([]);
    expect(t.frameIndex).toBe(100);
    expect(t.getActiveTracks()).toHaveLength(0);
  });
});

describe('SortTracker — Track shape', () => {
  it('reports timeSinceUpdate=0 for tracks matched this frame', () => {
    const t = new SortTracker({ minHits: 1 });
    const out = t.update([det([0, 0, 10, 10])]);
    expect(out[0]?.timeSinceUpdate).toBe(0);
  });

  it('reports state="confirmed" once minHits is reached', () => {
    const t = new SortTracker({ minHits: 3 });
    const d = det([10, 10, 50, 50]);
    t.update([d]); // tentative (warmup output)
    t.update([d]); // tentative
    const out = t.update([d]); // 3rd hit → confirmed
    expect(out[0]?.state).toBe('confirmed');
  });

  it('hits counts every match, age counts every alive frame', () => {
    const t = new SortTracker({ minHits: 1, maxAge: 10 });
    const d = det([10, 10, 50, 50]);
    t.update([d]); // age=0, hits=1
    t.update([d]); // age=1, hits=2
    const out = t.update([d]); // age=2, hits=3
    expect(out[0]?.hits).toBe(3);
    expect(out[0]?.age).toBe(2);
  });
});

describe('SortTracker.reset', () => {
  it('clears tracks and restarts id numbering at 1', () => {
    const t = new SortTracker({ minHits: 1 });
    t.update([det([0, 0, 10, 10]), det([100, 100, 200, 200])]);
    t.reset();
    expect(t.frameIndex).toBe(0);
    const out = t.update([det([0, 0, 10, 10])]);
    expect(out[0]?.id).toBe(1);
  });
});
