/**
 * Parsed `seqinfo.ini` metadata shipped with every MOTChallenge sequence
 * directory. Only the fields the eval pipeline consumes are surfaced.
 */
export interface SeqInfo {
  /** Sequence name, e.g. `MOT17-04-FRCNN`. */
  readonly name: string;
  /** Number of frames in the sequence. */
  readonly seqLength: number;
  /** Frames per second; undefined when absent. */
  readonly frameRate?: number;
  /** Image width in pixels; undefined when absent. */
  readonly imWidth?: number;
  /** Image height in pixels; undefined when absent. */
  readonly imHeight?: number;
}

/**
 * Parse `seqinfo.ini` text. Accepts the standard single-`[Sequence]`-section
 * INI subset MOTChallenge ships: `key=value` lines, `;` comments, LF or CRLF
 * endings; unknown keys and other sections are ignored.
 *
 * `name` and `seqLength` are required — missing or non-numeric `seqLength`
 * throws (`/seqinfo/i` and `/missing|invalid/i`).
 */
export function parseSeqinfo(_text: string): SeqInfo {
  throw new Error('not implemented');
}
