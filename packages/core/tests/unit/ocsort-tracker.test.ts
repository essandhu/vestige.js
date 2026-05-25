import { describe, expect, it } from 'vitest';
import { OcSortTracker } from '../../src/trackers/ocsort.js';
import type { BBox, Detection } from '../../src/types.js';

function det(bbox: BBox, score = 0.9, classId?: number): Detection {
  return classId === undefined ? { bbox, score } : { bbox, score, classId };
}

/** Drive a tracker N frames, returning the user-facing track list per frame. */
function run(tracker: OcSortTracker, detsByFrame: ReadonlyArray<ReadonlyArray<Detection>>) {
  return detsByFrame.map((d) => tracker.update(d));
}

describe('OcSortTracker — defaults', () => {
  it('detThresh defaults to 0.6 (ARCHITECTURE.md §6.4)', () => {
    expect(new OcSortTracker().detThresh).toBeCloseTo(0.6, 12);
  });

  it('maxAge defaults to 30', () => {
    expect(new OcSortTracker().maxAge).toBe(30);
  });

  it('minHits defaults to 3', () => {
    expect(new OcSortTracker().minHits).toBe(3);
  });

  it('iouThreshold defaults to 0.3', () => {
    expect(new OcSortTracker().iouThreshold).toBeCloseTo(0.3, 12);
  });

  it('deltaT defaults to 3', () => {
    expect(new OcSortTracker().deltaT).toBe(3);
  });

  it("asoFunc defaults to 'iou'", () => {
    expect(new OcSortTracker().asoFunc).toBe('iou');
  });

  it('inertia defaults to 0.2', () => {
    expect(new OcSortTracker().inertia).toBeCloseTo(0.2, 12);
  });

  it('useByte defaults to false', () => {
    expect(new OcSortTracker().useByte).toBe(false);
  });

  it('every option is overridable', () => {
    const t = new OcSortTracker({
      detThresh: 0.5,
      maxAge: 60,
      minHits: 1,
      iouThreshold: 0.4,
      deltaT: 5,
      asoFunc: 'giou',
      inertia: 0.3,
      useByte: true,
    });
    expect(t.detThresh).toBeCloseTo(0.5, 12);
    expect(t.maxAge).toBe(60);
    expect(t.minHits).toBe(1);
    expect(t.iouThreshold).toBeCloseTo(0.4, 12);
    expect(t.deltaT).toBe(5);
    expect(t.asoFunc).toBe('giou');
    expect(t.inertia).toBeCloseTo(0.3, 12);
    expect(t.useByte).toBe(true);
  });
});

describe('OcSortTracker — first frame and ID assignment', () => {
  it('emits the first detection as a track with id=1 during the warmup', () => {
    const t = new OcSortTracker(); // minHits=3 → warmup covers frame 1
    const out = t.update([det([10, 20, 110, 220])]);
    expect(out).toHaveLength(1);
    expect(out[0]?.id).toBe(1);
  });

  it('assigns sequential ids in detection order', () => {
    const t = new OcSortTracker();
    const out = t.update([det([0, 0, 50, 50]), det([500, 500, 600, 600])]);
    const ids = out.map((tr) => tr.id).sort((a, b) => a - b);
    expect(ids).toEqual([1, 2]);
  });

  it('does not spawn a track when the detection score is below detThresh', () => {
    const t = new OcSortTracker({ detThresh: 0.6, minHits: 1 });
    const out = t.update([det([0, 0, 50, 50], 0.5)]); // 0.5 < 0.6
    expect(out).toHaveLength(0);
    expect(t.getActiveTracks()).toHaveLength(0);
    expect(t.getLostTracks()).toHaveLength(0);
  });

  it('does not reuse ids after a track is removed', () => {
    const t = new OcSortTracker({ maxAge: 0, minHits: 1 });
    t.update([det([0, 0, 50, 50])]); // spawn id=1, warmup output
    t.update([]); // miss → tsu=1 > maxAge=0 → removed in same frame
    t.update([det([0, 0, 50, 50])]); // fresh spawn → id=2
    t.update([det([0, 0, 50, 50])]); // hits=1 ≥ minHits=1 → confirmed
    const out = t.getActiveTracks();
    expect(out).toHaveLength(1);
    expect(out[0]?.id).toBe(2);
  });
});

describe('OcSortTracker — warmup output rule', () => {
  it('outputs every matched track during the first minHits frames', () => {
    const t = new OcSortTracker({ minHits: 3 });
    const d = det([10, 10, 110, 110]);
    expect(t.update([d])).toHaveLength(1); // frame 1: warmup
    expect(t.update([d])).toHaveLength(1); // frame 2: warmup
    expect(t.update([d])).toHaveLength(1); // frame 3: warmup (still ≤ minHits)
  });

  it('after warmup, a brand-new tentative track is NOT output the frame it spawns', () => {
    const t = new OcSortTracker({ minHits: 3 });
    const settled = det([10, 10, 110, 110]);
    t.update([settled]);
    t.update([settled]);
    t.update([settled]); // through warmup
    t.update([settled]); // frame 4: still confirmed id=1
    const out = t.update([settled, det([500, 500, 600, 600])]);
    // Confirmed track 1 always; the new tentative track 2 is NOT output (warmup expired).
    expect(out.map((tr) => tr.id).sort()).toEqual([1]);
  });
});

describe('OcSortTracker — stable detection keeps the same id', () => {
  it('a stationary detection keeps id across 10 frames', () => {
    const t = new OcSortTracker();
    const d = det([100, 100, 200, 200]);
    const frames = run(
      t,
      Array.from({ length: 10 }, () => [d]),
    );
    const ids = frames.flatMap((f) => f.map((tr) => tr.id));
    expect(new Set(ids).size).toBe(1);
    expect(ids[0]).toBe(1);
  });

  it('a slowly moving detection keeps its id', () => {
    const t = new OcSortTracker();
    const detsByFrame: Detection[][] = [];
    for (let f = 0; f < 10; f++) {
      detsByFrame.push([det([100 + 2 * f, 100 + f, 200 + 2 * f, 200 + f])]);
    }
    const frames = run(t, detsByFrame);
    const ids = frames.flatMap((f) => f.map((tr) => tr.id));
    expect(new Set(ids).size).toBe(1);
  });
});

describe('OcSortTracker — iouThreshold gating', () => {
  it('a detection with zero IoU to every track does not match', () => {
    const t = new OcSortTracker({ iouThreshold: 0.3, minHits: 1, maxAge: 30 });
    t.update([det([0, 0, 100, 100])]); // spawn id=1, warmup output
    t.update([det([0, 0, 100, 100])]); // hits=1, confirmed
    const out = t.update([det([1000, 1000, 1100, 1100])]);
    // The far-away detection spawns id=2 (tentative, warmup-expired → no output);
    // existing track 1 is unmatched and transitions to lost (tsu=1, not output).
    expect(out).toHaveLength(0);
  });
});

describe('OcSortTracker — lifecycle on missed frames (ADR-0003 §4)', () => {
  it('advances timeSinceUpdate by exactly 1 per missed frame for a confirmed-then-lost track', () => {
    const t = new OcSortTracker({ minHits: 1, maxAge: 10 });
    const d = det([0, 0, 100, 100]);
    t.update([d]); // frame 1: spawn (tentative; warmup output)
    t.update([d]); // frame 2: hits=1 → confirmed
    t.update([]); // miss 1 → lost, tsu must be exactly 1
    expect(t.getLostTracks()).toHaveLength(1);
    expect(t.getLostTracks()[0]?.timeSinceUpdate).toBe(1);
    t.update([]); // miss 2
    expect(t.getLostTracks()[0]?.timeSinceUpdate).toBe(2);
    t.update([]); // miss 3
    expect(t.getLostTracks()[0]?.timeSinceUpdate).toBe(3);
  });

  it('confirmed → lost → re-found preserves the id', () => {
    const t = new OcSortTracker({ minHits: 1, maxAge: 10 });
    const d = det([0, 0, 100, 100]);
    t.update([d]); // spawn
    t.update([d]); // confirmed id=1
    t.update([]); // → lost
    t.update([]); // still lost
    const out = t.update([d]); // re-found
    expect(out).toHaveLength(1);
    expect(out[0]?.id).toBe(1);
  });

  it('confirmed → lost → removed after maxAge+1 misses', () => {
    const t = new OcSortTracker({ minHits: 1, maxAge: 2 });
    const d = det([0, 0, 100, 100]);
    t.update([d]); // tentative
    t.update([d]); // confirmed
    t.update([]); // miss 1 → lost, tsu=1
    t.update([]); // miss 2 → lost, tsu=2
    t.update([]); // miss 3 → removed (tsu=3 > maxAge=2)
    expect(t.getActiveTracks()).toHaveLength(0);
    expect(t.getLostTracks()).toHaveLength(0);
  });
});

describe('OcSortTracker — OCM hyperparameter wiring (paper §3.3)', () => {
  // Surface-level smoke: changing `inertia` and `deltaT` must not throw and
  // must not break the basic ID-preservation contract. The validation/ tier
  // tests the actual directional cost effect on engineered scenarios.
  it('inertia=0 (OCM disabled) still tracks a moving box', () => {
    const t = new OcSortTracker({ minHits: 1, inertia: 0 });
    const frames: Detection[][] = [];
    for (let f = 0; f < 5; f++) {
      frames.push([det([10 * f, 0, 100 + 10 * f, 100])]);
    }
    const out = run(t, frames);
    const ids = out.flatMap((f) => f.map((tr) => tr.id));
    expect(new Set(ids).size).toBe(1);
  });

  it('deltaT larger than the track age falls back gracefully', () => {
    const t = new OcSortTracker({ minHits: 1, deltaT: 100 });
    const out = t.update([det([0, 0, 100, 100])]);
    expect(out).toHaveLength(1);
    expect(out[0]?.id).toBe(1);
  });
});

describe('OcSortTracker — useByte (optional ByteTrack-style stage)', () => {
  it('useByte=false: a low-score detection cannot rescue an unmatched track', () => {
    const t = new OcSortTracker({ minHits: 1, maxAge: 10, detThresh: 0.6, useByte: false });
    const high = det([0, 0, 100, 100], 0.9);
    const low = det([0, 0, 100, 100], 0.4); // between LOW_SCORE_FLOOR (0.1) and detThresh (0.6)
    t.update([high]); // spawn id=1, warmup
    t.update([high]); // confirmed
    const out = t.update([low]); // primary stage drops it (score < detThresh)
    // With useByte=false, no BYTE rescue → track 1 unmatched → not output.
    expect(out).toHaveLength(0);
    // The track should still exist as lost.
    expect(t.getLostTracks()).toHaveLength(1);
  });

  it('useByte=true: a low-score detection at the same location can rescue an unmatched track', () => {
    const t = new OcSortTracker({ minHits: 1, maxAge: 10, detThresh: 0.6, useByte: true });
    const high = det([0, 0, 100, 100], 0.9);
    const low = det([0, 0, 100, 100], 0.4);
    t.update([high]);
    t.update([high]); // confirmed id=1
    const out = t.update([low]);
    // BYTE stage matches the low-score detection to track 1 → still output, id preserved.
    expect(out).toHaveLength(1);
    expect(out[0]?.id).toBe(1);
  });
});

describe('OcSortTracker — asoFunc selector', () => {
  it("asoFunc='giou' is accepted and produces the same id for a clean stationary case", () => {
    const t = new OcSortTracker({ asoFunc: 'giou', minHits: 1 });
    const d = det([100, 100, 200, 200]);
    t.update([d]);
    const out = t.update([d]);
    expect(out[0]?.id).toBe(1);
  });
});

describe('OcSortTracker — payload + classId preservation', () => {
  it('payload flows through update()', () => {
    type P = { label: string };
    const t = new OcSortTracker<P>({ minHits: 1 });
    const out = t.update([{ bbox: [0, 0, 100, 100], score: 0.9, payload: { label: 'dog' } }]);
    expect(out[0]?.payload).toEqual({ label: 'dog' });
  });

  it('classId flows through update()', () => {
    const t = new OcSortTracker({ minHits: 1 });
    const out = t.update([det([0, 0, 100, 100], 0.9, 7)]);
    expect(out[0]?.classId).toBe(7);
  });
});

describe('OcSortTracker — degenerate inputs', () => {
  it('handles empty detections (no crash, no output)', () => {
    const t = new OcSortTracker();
    expect(t.update([])).toEqual([]);
    expect(t.frameIndex).toBe(1);
  });

  it('handles many empty frames in a row', () => {
    const t = new OcSortTracker();
    for (let i = 0; i < 100; i++) t.update([]);
    expect(t.frameIndex).toBe(100);
    expect(t.getActiveTracks()).toHaveLength(0);
  });
});

describe('OcSortTracker — Track shape', () => {
  it('reports timeSinceUpdate=0 for tracks matched this frame', () => {
    const t = new OcSortTracker({ minHits: 1 });
    const out = t.update([det([0, 0, 100, 100])]);
    expect(out[0]?.timeSinceUpdate).toBe(0);
  });

  it('hits counts every match, age counts every alive frame', () => {
    const t = new OcSortTracker({ minHits: 1, maxAge: 10 });
    const d = det([10, 10, 110, 110]);
    t.update([d]); // age=0, hits=0 (spawn isn't a hit)
    t.update([d]); // age=1, hits=1
    t.update([d]); // age=2, hits=2
    const out = t.update([d]); // age=3, hits=3
    expect(out[0]?.hits).toBe(3);
    expect(out[0]?.age).toBe(3);
  });
});

describe('OcSortTracker.reset', () => {
  it('clears tracks and restarts id numbering at 1', () => {
    const t = new OcSortTracker({ minHits: 1 });
    t.update([det([0, 0, 100, 100]), det([500, 500, 600, 600])]);
    t.reset();
    expect(t.frameIndex).toBe(0);
    const out = t.update([det([0, 0, 100, 100])]);
    expect(out[0]?.id).toBe(1);
  });
});
