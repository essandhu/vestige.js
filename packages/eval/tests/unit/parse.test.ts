import { describe, expect, it } from 'vitest';
import type { MotEntry } from '../../src/motchallenge/parse.js';
import {
  detectionsByFrame,
  filterGt,
  formatMotChallenge,
  parseMotChallenge,
} from '../../src/motchallenge/parse.js';

describe('parseMotChallenge', () => {
  it('parses a 9-column gt row', () => {
    const [e] = parseMotChallenge('1,1,10,20,30,40,1,1,0.83\n');
    expect(e).toEqual({
      frame: 1,
      id: 1,
      bbox: [10, 20, 40, 60],
      score: 1,
      classId: 1,
      visibility: 0.83,
    });
  });

  it('parses a 10-column det row, ignoring the unused world-coordinate tail', () => {
    const [e] = parseMotChallenge('2,-1,5.5,6.25,10,20,0.97,-1,-1,-1\n');
    expect(e).toEqual({
      frame: 2,
      id: -1,
      bbox: [5.5, 6.25, 15.5, 26.25],
      score: 0.97,
      classId: -1,
      visibility: -1,
    });
  });

  it('defaults classId and visibility to -1 on 7-column rows', () => {
    const [e] = parseMotChallenge('1,2,0,0,10,10,0.5\n');
    expect(e?.classId).toBe(-1);
    expect(e?.visibility).toBe(-1);
  });

  it('accepts CRLF line endings and skips blank lines', () => {
    const entries = parseMotChallenge('1,1,0,0,10,10,1,1,1\r\n\r\n2,1,0,0,10,10,1,1,1\r\n');
    expect(entries).toHaveLength(2);
    expect(entries[0]?.frame).toBe(1);
    expect(entries[1]?.frame).toBe(2);
  });

  it('returns an empty array for empty text', () => {
    expect(parseMotChallenge('')).toEqual([]);
    expect(parseMotChallenge('\n')).toEqual([]);
  });

  it('throws on rows with fewer than 7 fields or non-numeric fields', () => {
    expect(() => parseMotChallenge('1,2,3\n')).toThrow(/malformed/i);
    expect(() => parseMotChallenge('1,a,0,0,10,10,1\n')).toThrow(/malformed/i);
  });
});

describe('formatMotChallenge', () => {
  const entry = (overrides: Partial<MotEntry>): MotEntry => ({
    frame: 1,
    id: 1,
    bbox: [10, 20, 40, 60],
    score: 0.9,
    classId: -1,
    visibility: -1,
    ...overrides,
  });

  it('emits the 10-column MOTChallenge result form', () => {
    expect(formatMotChallenge([entry({})])).toBe('1,1,10,20,30,40,0.9,-1,-1,-1\n');
  });

  it('rounds float noise to at most 6 decimals', () => {
    const noisy = entry({ bbox: [10.0000000001, 20, 40.0000000001, 60.1234567891] });
    expect(formatMotChallenge([noisy])).toBe('1,1,10,20,30,40.123457,0.9,-1,-1,-1\n');
  });

  it('round-trips through parseMotChallenge', () => {
    const entries = [
      entry({ frame: 3, id: 12, bbox: [1.5, 2.25, 11.5, 22.25], score: 0.654321 }),
      entry({ frame: 4, id: 13, classId: 1, visibility: 0.5 }),
    ];
    const parsed = parseMotChallenge(formatMotChallenge(entries));
    expect(parsed).toHaveLength(2);
    expect(parsed[0]?.frame).toBe(3);
    expect(parsed[0]?.id).toBe(12);
    expect(parsed[0]?.bbox[2]).toBeCloseTo(11.5, 6);
    expect(parsed[0]?.score).toBeCloseTo(0.654321, 6);
    expect(parsed[1]?.classId).toBe(1);
    expect(parsed[1]?.visibility).toBeCloseTo(0.5, 6);
  });
});

describe('filterGt', () => {
  const rows = parseMotChallenge(
    [
      '1,1,0,0,10,10,1,1,0.9', // considered, class 1, vis 0.9
      '1,2,0,0,10,10,0,1,0.9', // consider-flag 0
      '1,3,0,0,10,10,1,7,0.9', // class 7
      '1,4,0,0,10,10,1,1,0.1', // low visibility
      '1,5,0,0,10,10,1,1,-1', // absent visibility marker
    ].join('\n'),
  );

  it('keeps everything when no filters are set', () => {
    expect(filterGt(rows, {})).toHaveLength(5);
  });

  it('drops consider-flag-0 entries with requireConsidered', () => {
    const kept = filterGt(rows, { requireConsidered: true });
    expect(kept.map((e) => e.id)).toEqual([1, 3, 4, 5]);
  });

  it('filters by class list', () => {
    const kept = filterGt(rows, { classIds: [1] });
    expect(kept.map((e) => e.id)).toEqual([1, 2, 4, 5]);
  });

  it('filters by minimum visibility, dropping the -1 absent marker', () => {
    const kept = filterGt(rows, { minVisibility: 0.5 });
    expect(kept.map((e) => e.id)).toEqual([1, 2, 3]);
  });

  it('combines filters (the standard MOT17 pedestrian setting)', () => {
    const kept = filterGt(rows, { requireConsidered: true, classIds: [1] });
    expect(kept.map((e) => e.id)).toEqual([1, 4, 5]);
  });
});

describe('detectionsByFrame', () => {
  it('groups by frame with empty arrays for gaps', () => {
    const dets = detectionsByFrame(parseMotChallenge('1,-1,0,0,10,10,0.9\n3,-1,5,5,10,10,0.8\n'));
    expect(dets).toHaveLength(3);
    expect(dets[0]).toHaveLength(1);
    expect(dets[1]).toEqual([]);
    expect(dets[2]).toHaveLength(1);
    expect(dets[0]?.[0]).toEqual({ bbox: [0, 0, 10, 10], score: 0.9 });
  });

  it('attaches classId when present', () => {
    const dets = detectionsByFrame(parseMotChallenge('1,-1,0,0,10,10,0.9,2,-1,-1\n'));
    expect(dets[0]?.[0]?.classId).toBe(2);
  });

  it('honors an explicit numFrames', () => {
    const entries = parseMotChallenge('1,-1,0,0,10,10,0.9\n2,-1,0,0,10,10,0.9\n');
    expect(detectionsByFrame(entries, 4)).toHaveLength(4);
    expect(detectionsByFrame(entries, 1)).toHaveLength(1);
  });
});
