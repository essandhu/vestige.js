import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { loadMotSequence } from '../../src/motchallenge/load.js';
import { parseSeqinfo } from '../../src/motchallenge/seqinfo.js';

const FULL_INI = [
  '[Sequence]',
  'name=MOT17-04-FRCNN',
  'imDir=img1',
  'frameRate=30',
  'seqLength=1050',
  'imWidth=1920',
  'imHeight=1080',
  'imExt=.jpg',
  '',
].join('\n');

describe('parseSeqinfo', () => {
  it('parses the standard MOTChallenge seqinfo.ini', () => {
    expect(parseSeqinfo(FULL_INI)).toEqual({
      name: 'MOT17-04-FRCNN',
      seqLength: 1050,
      frameRate: 30,
      imWidth: 1920,
      imHeight: 1080,
    });
  });

  it('accepts CRLF endings, comments, and spacing around =', () => {
    const ini = '[Sequence]\r\n; a comment\r\nname = SYN-01\r\nseqLength = 12\r\n';
    expect(parseSeqinfo(ini)).toEqual({ name: 'SYN-01', seqLength: 12 });
  });

  it('leaves optional fields undefined when absent', () => {
    const info = parseSeqinfo('[Sequence]\nname=X\nseqLength=5\n');
    expect(info.frameRate).toBeUndefined();
    expect(info.imWidth).toBeUndefined();
    expect(info.imHeight).toBeUndefined();
  });

  it('throws on missing name or missing/invalid seqLength', () => {
    expect(() => parseSeqinfo('[Sequence]\nseqLength=5\n')).toThrow(/missing|invalid/i);
    expect(() => parseSeqinfo('[Sequence]\nname=X\n')).toThrow(/missing|invalid/i);
    expect(() => parseSeqinfo('[Sequence]\nname=X\nseqLength=abc\n')).toThrow(/missing|invalid/i);
  });
});

describe('loadMotSequence', () => {
  const dir = mkdtempSync(join(tmpdir(), 'vestige-mot-'));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  it('loads seqinfo + gt, and returns null for an absent det file', () => {
    writeFileSync(join(dir, 'seqinfo.ini'), FULL_INI);
    mkdirSync(join(dir, 'gt'), { recursive: true });
    writeFileSync(join(dir, 'gt', 'gt.txt'), '1,1,10,20,30,40,1,1,1\n');

    const seq = loadMotSequence(dir);
    expect(seq.seqInfo.name).toBe('MOT17-04-FRCNN');
    expect(seq.seqInfo.seqLength).toBe(1050);
    expect(seq.gt).toHaveLength(1);
    expect(seq.gt?.[0]?.bbox).toEqual([10, 20, 40, 60]);
    expect(seq.detections).toBeNull();
  });

  it('throws a specific error when seqinfo.ini is absent', () => {
    const empty = mkdtempSync(join(tmpdir(), 'vestige-mot-empty-'));
    try {
      expect(() => loadMotSequence(empty)).toThrow(/seqinfo/i);
    } finally {
      rmSync(empty, { recursive: true, force: true });
    }
  });
});
