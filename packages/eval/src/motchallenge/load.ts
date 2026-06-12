import { readFileSync } from 'node:fs';
import type { MotEntry } from './parse.js';
import { parseMotChallenge } from './parse.js';

/**
 * Read and parse one MOTChallenge CSV file (`gt.txt`, `det.txt`, or a tracker
 * result file) from disk. Node-only — this is the one fs touchpoint in the
 * package, kept apart from the pure parsing/metric code (ARCHITECTURE.md §3:
 * eval may use Node APIs; core may not).
 */
export function loadMotFile(path: string): MotEntry[] {
  return parseMotChallenge(readFileSync(path, 'utf8'));
}
