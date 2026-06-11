import type { BBox, Detection } from '../core.js';

/**
 * One row of a MOTChallenge CSV file (`gt/gt.txt`, `det/det.txt`, or a tracker
 * result file). Column semantics follow the MOTChallenge devkit
 * (Dendorfer et al. 2020):
 *
 * `frame, id, bbLeft, bbTop, bbWidth, bbHeight, conf, classId, visibility`
 *
 * The bbox is converted to the library-canonical xyxy form on parse
 * (`x2 = left + width`, `y2 = top + height`). Note the column-7 semantics
 * differ by file kind: in `gt.txt` it is the 0/1 "consider this entry" flag;
 * in detection and tracker files it is a confidence score. Both land in
 * `score` untouched.
 */
export interface MotEntry {
  /** 1-based frame number. */
  readonly frame: number;
  /** Identity; `-1` in raw detection files. */
  readonly id: number;
  /** Box in xyxy, converted from the file's left/top/width/height. */
  readonly bbox: BBox;
  /** Column 7: confidence score, or the consider-flag in gt files. */
  readonly score: number;
  /** Column 8: object class; `-1` when absent. */
  readonly classId: number;
  /** Column 9: visibility ratio in [0, 1]; `-1` when absent. */
  readonly visibility: number;
}

/**
 * Parse MOTChallenge CSV text into entries, in file order.
 *
 * Accepts LF or CRLF line endings and skips blank lines. Rows must have at
 * least 7 comma-separated numeric fields (frame…conf); columns 8–9 default to
 * `-1` when absent, and any columns past 9 (e.g. the unused `x,y,z` of 3D-form
 * det files) are ignored. A row with fewer than 7 fields or a non-numeric
 * field throws (`/malformed/i`).
 */
export function parseMotChallenge(_text: string): MotEntry[] {
  throw new Error('not implemented');
}

/**
 * Render entries back to MOTChallenge CSV text: LF line endings, a trailing
 * newline, one 10-column row per entry —
 * `frame,id,left,top,width,height,score,classId,visibility,-1`. For tracker
 * output (where `classId` and `visibility` are `-1`) this is exactly the
 * MOTChallenge result-file form `…,conf,-1,-1,-1`.
 *
 * The xyxy bbox is converted back to left/top/width/height. Numeric fields
 * are rounded to at most 6 decimal places so Kalman-filtered float noise
 * doesn't bloat result files; `parseMotChallenge(formatMotChallenge(e))`
 * round-trips frame/id exactly and geometry to within 1e-6.
 */
export function formatMotChallenge(_entries: ReadonlyArray<MotEntry>): string {
  throw new Error('not implemented');
}

/**
 * Options for {@link filterGt}. Every filter is opt-in; calling with `{}`
 * keeps all entries (explicit cost over magic ergonomics — ARCHITECTURE.md
 * §2.6). The standard MOT17/MOT20 pedestrian evaluation setting is
 * `{ requireConsidered: true, classIds: [1] }`.
 */
export interface GtFilterOptions {
  /** Drop entries whose consider-flag (`score` column of gt files) is 0. */
  readonly requireConsidered?: boolean;
  /** Keep only entries whose `classId` is in this list. */
  readonly classIds?: ReadonlyArray<number>;
  /**
   * Keep only entries with `visibility >= minVisibility`. Entries with the
   * absent-marker `visibility === -1` are dropped by any threshold > -1.
   */
  readonly minVisibility?: number;
}

/** Apply {@link GtFilterOptions} to parsed gt entries; returns a new array. */
export function filterGt(_entries: ReadonlyArray<MotEntry>, _options: GtFilterOptions): MotEntry[] {
  throw new Error('not implemented');
}

/**
 * Group raw detection entries into a per-frame `Detection[][]` ready to feed
 * a tracker (frame `f` lands at index `f - 1`; frames with no detections are
 * empty arrays). `numFrames` extends or truncates the result; it defaults to
 * the maximum frame number present.
 *
 * `score` is carried through, and `classId` is attached when it isn't the
 * absent-marker `-1`.
 */
export function detectionsByFrame(
  _entries: ReadonlyArray<MotEntry>,
  _numFrames?: number,
): Detection[][] {
  throw new Error('not implemented');
}
