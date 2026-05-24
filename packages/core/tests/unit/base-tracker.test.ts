import { describe, expect, it } from 'vitest';
import type { KalmanState } from '../../src/filters/kalman.js';
import {
  type AssociationResult,
  BaseTracker,
  type InternalTrack,
} from '../../src/trackers/base.js';
import type { BBox, Detection } from '../../src/types.js';

/**
 * Tests for {@link BaseTracker}'s lifecycle machinery in isolation from any
 * motion model. The fake subclass below uses a no-op Kalman state and a
 * scripted association function so each test can drive the state machine
 * deterministically.
 */

const NO_STATE: KalmanState = {
  mean: new Float64Array(0),
  covariance: new Float64Array(0),
};

/**
 * Minimal {@link BaseTracker} subclass for testing the lifecycle. Motion is a
 * no-op (predicted bbox = last-known bbox). Association is injected per
 * instance via {@link FakeTracker.setAssociate}.
 */
class FakeTracker extends BaseTracker {
  predictCalls = 0;
  updateCalls = 0;
  initCalls = 0;

  private associateFn: (
    detections: ReadonlyArray<Detection>,
    associable: ReadonlyArray<InternalTrack>,
  ) => AssociationResult = () => ({ matched: [], unmatchedTracks: [], unmatchedDetections: [] });

  constructor(opts?: { minHits?: number; maxAge?: number }) {
    super({ minHits: opts?.minHits ?? 3, maxAge: opts?.maxAge ?? 1 });
  }

  setAssociate(fn: typeof this.associateFn): void {
    this.associateFn = fn;
  }

  /** Snapshot of every track in the map (including lost), for white-box assertions. */
  snapshot(): InternalTrack[] {
    return [...this.tracks.values()];
  }

  protected predictTrack(_track: InternalTrack): void {
    this.predictCalls++;
    // No-op motion: predicted bbox stays at the last-known location.
  }

  protected updateTrack(track: InternalTrack, detection: Detection): void {
    this.updateCalls++;
    track.bbox = detection.bbox;
    track.lastDetection = detection;
  }

  protected initTrack(detection: Detection): InternalTrack {
    this.initCalls++;
    // hits = 0, hitStreak = 0 to match sort.py — the spawn detection is NOT
    // a hit. See `BaseTracker.initTrack` JSDoc.
    return {
      id: 0,
      state: 'tentative',
      age: 0,
      hits: 0,
      hitStreak: 0,
      timeSinceUpdate: 0,
      kalmanState: NO_STATE,
      bbox: detection.bbox,
      lastDetection: detection,
    };
  }

  protected associate(
    detections: ReadonlyArray<Detection>,
    associable: ReadonlyArray<InternalTrack>,
  ): AssociationResult {
    return this.associateFn(detections, associable);
  }
}

function det(bbox: BBox, score = 0.9): Detection {
  return { bbox, score };
}

function matchAll(
  detections: ReadonlyArray<Detection>,
  associable: ReadonlyArray<InternalTrack>,
): AssociationResult {
  const matched: Array<[number, number]> = [];
  const n = Math.min(detections.length, associable.length);
  for (let i = 0; i < n; i++) matched.push([i, i]);
  const unmatchedTracks: number[] = [];
  for (let i = n; i < associable.length; i++) unmatchedTracks.push(i);
  const unmatchedDetections: number[] = [];
  for (let i = n; i < detections.length; i++) unmatchedDetections.push(i);
  return { matched, unmatchedTracks, unmatchedDetections };
}

function matchNone(
  detections: ReadonlyArray<Detection>,
  associable: ReadonlyArray<InternalTrack>,
): AssociationResult {
  return {
    matched: [],
    unmatchedTracks: associable.map((_, i) => i),
    unmatchedDetections: detections.map((_, i) => i),
  };
}

describe('BaseTracker.update — frame index', () => {
  it('starts at 0 before any update call', () => {
    const t = new FakeTracker();
    expect(t.frameIndex).toBe(0);
  });

  it('advances by 1 on each update', () => {
    const t = new FakeTracker();
    t.update([]);
    expect(t.frameIndex).toBe(1);
    t.update([]);
    expect(t.frameIndex).toBe(2);
  });
});

describe('BaseTracker.update — predict / update / init wiring', () => {
  it('calls initTrack for each unmatched detection', () => {
    const t = new FakeTracker();
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10]), det([20, 20, 30, 30])]);
    expect(t.initCalls).toBe(2);
  });

  it('does not call predictTrack on the frame a track is created', () => {
    const t = new FakeTracker();
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]);
    expect(t.predictCalls).toBe(0);
  });

  it('calls predictTrack for every existing track on the next frame', () => {
    const t = new FakeTracker();
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10]), det([20, 20, 30, 30])]);
    t.predictCalls = 0;
    t.setAssociate(matchAll);
    t.update([det([0, 0, 10, 10]), det([20, 20, 30, 30])]);
    expect(t.predictCalls).toBe(2);
  });

  it('calls updateTrack for each matched pair', () => {
    const t = new FakeTracker();
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]);
    t.updateCalls = 0;
    t.setAssociate(matchAll);
    t.update([det([0, 0, 10, 10])]);
    expect(t.updateCalls).toBe(1);
  });
});

describe('BaseTracker.update — track IDs', () => {
  it('assigns id=1 to the first created track', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 30 });
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]);
    // Track is tentative after spawn (hits=0); default exportConfirmed
    // doesn't output it, so check via snapshot.
    expect(t.snapshot()[0]?.id).toBe(1);
  });

  it('assigns monotonically increasing ids across frames', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 30 });
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]); // spawn id=1
    t.update([det([20, 20, 30, 30])]); // unmatched (track 1's only detection is itself) → spawn id=2
    const ids = t
      .snapshot()
      .map((tr) => tr.id)
      .sort((a, b) => a - b);
    expect(ids).toEqual([1, 2]);
  });

  it('does not reuse the id of a removed track', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 0 });
    // Frame 1: spawn track id=1 (tentative, hits=0).
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]);
    // Frame 2: miss → tentative + tsu(1) > maxAge(0) → removed.
    t.update([]);
    expect(t.snapshot()).toHaveLength(0);
    // Frame 3: spawn id=2 (not a recycled 1).
    t.update([det([50, 50, 60, 60])]);
    expect(t.snapshot()[0]?.id).toBe(2);
  });
});

describe('BaseTracker.update — tentative lifecycle', () => {
  it('promotes tentative → confirmed after minHits consecutive matches', () => {
    const t = new FakeTracker({ minHits: 3, maxAge: 5 });
    // Frame 1: spawn (hits=0).
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]);
    expect(t.snapshot()[0]?.state).toBe('tentative');
    // Frames 2–4: match. Per sort.py convention, hits=0 at init so the
    // 3rd match-after-spawn (= frame 4) is what trips minHits.
    t.setAssociate(matchAll);
    t.update([det([0, 0, 10, 10])]); // hits=1
    expect(t.snapshot()[0]?.state).toBe('tentative');
    t.update([det([0, 0, 10, 10])]); // hits=2
    expect(t.snapshot()[0]?.state).toBe('tentative');
    t.update([det([0, 0, 10, 10])]); // hits=3 ≥ minHits → confirmed
    expect(t.snapshot()[0]?.state).toBe('confirmed');
  });

  it('keeps a tentative track alive for maxAge frames of misses, then removes it', () => {
    const t = new FakeTracker({ minHits: 3, maxAge: 2 });
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]); // spawn, tsu=0
    expect(t.snapshot()[0]?.state).toBe('tentative');
    t.update([]); // tsu=1, alive
    expect(t.snapshot()[0]?.state).toBe('tentative');
    t.update([]); // tsu=2 == maxAge, alive
    expect(t.snapshot()[0]?.state).toBe('tentative');
    t.update([]); // tsu=3 > maxAge → removed
    expect(t.snapshot()).toHaveLength(0);
  });
});

describe('BaseTracker.update — confirmed → lost → removed', () => {
  it('transitions confirmed → lost on a miss', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 5 });
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]); // spawn (tentative)
    t.setAssociate(matchAll);
    t.update([det([0, 0, 10, 10])]); // hits=1 → confirmed
    expect(t.snapshot()[0]?.state).toBe('confirmed');
    t.setAssociate(matchNone);
    t.update([]); // miss
    expect(t.snapshot()[0]?.state).toBe('lost');
  });

  it('keeps a lost track for maxAge frames, then removes it', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 2 });
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]); // spawn
    t.setAssociate(matchAll);
    t.update([det([0, 0, 10, 10])]); // confirmed
    t.setAssociate(matchNone);
    t.update([]); // miss 1 → lost, tsu=1
    expect(t.snapshot()[0]?.state).toBe('lost');
    t.update([]); // miss 2 → still lost, tsu=2 (= maxAge)
    expect(t.snapshot()[0]?.state).toBe('lost');
    t.update([]); // miss 3 → removed (tsu=3 > maxAge=2)
    expect(t.snapshot()).toHaveLength(0);
  });

  it('promotes a lost track back to confirmed when re-matched', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 5 });
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]); // spawn
    t.setAssociate(matchAll);
    t.update([det([0, 0, 10, 10])]); // confirmed
    t.setAssociate(matchNone);
    t.update([]); // → lost
    expect(t.snapshot()[0]?.state).toBe('lost');
    t.setAssociate(matchAll);
    t.update([det([0, 0, 10, 10])]); // re-found
    expect(t.snapshot()[0]?.state).toBe('confirmed');
  });

  it('resets timeSinceUpdate to 0 on re-match', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 5 });
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]); // spawn
    t.setAssociate(matchAll);
    t.update([det([0, 0, 10, 10])]); // confirmed
    t.setAssociate(matchNone);
    t.update([]); // tsu=1
    t.update([]); // tsu=2
    expect(t.snapshot()[0]?.timeSinceUpdate).toBe(2);
    t.setAssociate(matchAll);
    t.update([det([0, 0, 10, 10])]);
    expect(t.snapshot()[0]?.timeSinceUpdate).toBe(0);
  });
});

describe('BaseTracker.update — counters', () => {
  it('increments hits on every match, never on a miss', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 5 });
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]); // hits=0 (init; spawn detection is NOT a hit)
    t.setAssociate(matchAll);
    t.update([det([0, 0, 10, 10])]); // hits=1
    t.update([det([0, 0, 10, 10])]); // hits=2
    t.setAssociate(matchNone);
    t.update([]); // miss, hits stays at 2
    expect(t.snapshot()[0]?.hits).toBe(2);
  });

  it('resets hitStreak to 0 on a miss', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 5 });
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]); // hitStreak=0 (init)
    t.setAssociate(matchAll);
    t.update([det([0, 0, 10, 10])]); // hitStreak=1
    t.update([det([0, 0, 10, 10])]); // hitStreak=2
    t.setAssociate(matchNone);
    t.update([]); // miss → hitStreak=0
    expect(t.snapshot()[0]?.hitStreak).toBe(0);
  });

  it('increments age every frame the track is alive (matched or not)', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 5 });
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]); // age starts at 0; predict not called on creation frame
    t.update([]); // age=1
    t.update([]); // age=2
    expect(t.snapshot()[0]?.age).toBe(2);
  });
});

describe('BaseTracker.exportConfirmed (default rule)', () => {
  it('returns confirmed tracks matched this frame', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 5 });
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]); // spawn (tentative; default rule does not output)
    t.setAssociate(matchAll);
    const out = t.update([det([0, 0, 10, 10])]); // hits=1 → confirmed
    expect(out).toHaveLength(1);
    expect(out[0]?.state).toBe('confirmed');
  });

  it('excludes confirmed tracks that did not match this frame', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 5 });
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]);
    t.setAssociate(matchAll);
    t.update([det([0, 0, 10, 10])]); // confirmed
    t.setAssociate(matchNone);
    const out = t.update([]); // lost this frame
    expect(out).toHaveLength(0);
  });

  it('excludes tentative tracks (default rule, no warmup)', () => {
    const t = new FakeTracker({ minHits: 3, maxAge: 5 });
    t.setAssociate(matchNone);
    const out = t.update([det([0, 0, 10, 10])]);
    expect(out).toHaveLength(0);
  });
});

describe('BaseTracker — public Track materialization', () => {
  it('carries score, classId, and payload through from the matched detection', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 5 });
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]); // spawn with dummy detection
    t.setAssociate(matchAll);
    const out = t.update([
      { bbox: [0, 0, 10, 10], score: 0.75, classId: 42, payload: { tag: 'cat' } },
    ]); // match → updateTrack writes lastDetection from this frame → confirmed
    expect(out[0]?.score).toBe(0.75);
    expect(out[0]?.classId).toBe(42);
    expect(out[0]?.payload).toEqual({ tag: 'cat' });
  });

  it('reflects the post-update bbox (from the matched detection, not the prior prediction)', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 5 });
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]); // spawn
    t.setAssociate(matchAll);
    t.update([det([0, 0, 10, 10])]); // confirm
    const out = t.update([det([5, 5, 15, 15])]); // detection at new location
    expect(out[0]?.bbox).toEqual([5, 5, 15, 15]);
  });
});

describe('BaseTracker.getActiveTracks / getLostTracks', () => {
  it('getActiveTracks returns only confirmed tracks', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 5 });
    // Spawn two tracks; confirm both (matchAll on frame 2).
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10]), det([20, 20, 30, 30])]);
    t.setAssociate(matchAll);
    t.update([det([0, 0, 10, 10]), det([20, 20, 30, 30])]); // both → confirmed
    // Frame 3: only track 0 matches; track 1 → lost.
    t.setAssociate((dets, _ass) => ({
      matched: [[0, 0]],
      unmatchedTracks: [1],
      unmatchedDetections: dets.slice(1).map((_, i) => i + 1),
    }));
    t.update([det([0, 0, 10, 10])]);
    const active = t.getActiveTracks();
    expect(active.map((tr) => tr.state)).toEqual(['confirmed']);
  });

  it('getLostTracks returns only lost tracks', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 5 });
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10]), det([20, 20, 30, 30])]);
    t.setAssociate(matchAll);
    t.update([det([0, 0, 10, 10]), det([20, 20, 30, 30])]); // both confirmed
    t.setAssociate((dets, _ass) => ({
      matched: [[0, 0]],
      unmatchedTracks: [1],
      unmatchedDetections: dets.slice(1).map((_, i) => i + 1),
    }));
    t.update([det([0, 0, 10, 10])]);
    const lost = t.getLostTracks();
    expect(lost.map((tr) => tr.state)).toEqual(['lost']);
  });
});

describe('BaseTracker.reset', () => {
  it('clears tracks, resets frame counter, resets id counter', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 5 });
    t.setAssociate(matchNone);
    t.update([det([0, 0, 10, 10])]);
    t.update([det([20, 20, 30, 30])]);
    expect(t.frameIndex).toBe(2);
    t.reset();
    expect(t.frameIndex).toBe(0);
    expect(t.getActiveTracks()).toHaveLength(0);
    expect(t.getLostTracks()).toHaveLength(0);
    t.update([det([0, 0, 10, 10])]);
    expect(t.snapshot()[0]?.id).toBe(1); // nextId reset
  });
});

describe('BaseTracker.update — input safety', () => {
  it('does not mutate the input detections array', () => {
    const t = new FakeTracker({ minHits: 1, maxAge: 5 });
    t.setAssociate(matchNone);
    const dets = [det([0, 0, 10, 10]), det([20, 20, 30, 30])];
    const snapshot = dets.slice();
    t.update(dets);
    expect(dets).toEqual(snapshot);
  });

  it('handles an empty detection frame on an empty tracker', () => {
    const t = new FakeTracker();
    const out = t.update([]);
    expect(out).toEqual([]);
    expect(t.frameIndex).toBe(1);
  });
});
