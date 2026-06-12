import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { MotEntry } from './parse.js';
import { parseMotChallenge } from './parse.js';
import type { SeqInfo } from './seqinfo.js';
import { parseSeqinfo } from './seqinfo.js';

/**
 * Read and parse one MOTChallenge CSV file (`gt.txt`, `det.txt`, or a tracker
 * result file) from disk. Node-only — this module is the fs touchpoint in the
 * package, kept apart from the pure parsing/metric code (ARCHITECTURE.md §3:
 * eval may use Node APIs; core may not).
 */
export function loadMotFile(path: string): MotEntry[] {
  return parseMotChallenge(readFileSync(path, 'utf8'));
}

/** A MOTChallenge sequence directory loaded into memory. */
export interface MotSequence {
  readonly seqInfo: SeqInfo;
  /** Parsed `gt/gt.txt`, or null when the sequence ships no ground truth (test split). */
  readonly gt: MotEntry[] | null;
  /** Parsed `det/det.txt`, or null when the sequence ships no public detections. */
  readonly detections: MotEntry[] | null;
}

/**
 * Load a standard MOTChallenge sequence directory
 * (`<dir>/seqinfo.ini`, `<dir>/gt/gt.txt`, `<dir>/det/det.txt`).
 * `seqinfo.ini` is required (`/seqinfo/i` on absence); gt and det files are
 * each optional and come back null when absent.
 */
export function loadMotSequence(dir: string): MotSequence {
  const seqinfoPath = join(dir, 'seqinfo.ini');
  if (!existsSync(seqinfoPath)) {
    throw new Error(`no seqinfo.ini found in sequence directory ${dir}`);
  }
  const seqInfo = parseSeqinfo(readFileSync(seqinfoPath, 'utf8'));

  const gtPath = join(dir, 'gt', 'gt.txt');
  const detPath = join(dir, 'det', 'det.txt');
  return {
    seqInfo,
    gt: existsSync(gtPath) ? loadMotFile(gtPath) : null,
    detections: existsSync(detPath) ? loadMotFile(detPath) : null,
  };
}
