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
export function parseMotChallenge(text: string): MotEntry[] {
  const out: MotEntry[] = [];
  const lines = text.split('\n');
  for (let lineNo = 0; lineNo < lines.length; lineNo++) {
    const line = (lines[lineNo] ?? '').trim();
    if (line === '') continue;

    const fields = line.split(',');
    if (fields.length < 7) {
      throw new Error(
        `malformed MOTChallenge line ${lineNo + 1}: expected at least 7 fields, got ${fields.length}`,
      );
    }

    const values: number[] = [];
    const numericFields = Math.min(fields.length, 9);
    for (let i = 0; i < numericFields; i++) {
      const field = (fields[i] ?? '').trim();
      const value = Number(field);
      if (field === '' || !Number.isFinite(value)) {
        throw new Error(`malformed MOTChallenge line ${lineNo + 1}: non-numeric field '${field}'`);
      }
      values.push(value);
    }

    const [frame = 0, id = 0, left = 0, top = 0, width = 0, height = 0, score = 0] = values;
    out.push({
      frame,
      id,
      bbox: [left, top, left + width, top + height],
      score,
      classId: values[7] ?? -1,
      visibility: values[8] ?? -1,
    });
  }
  return out;
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
export function formatMotChallenge(entries: ReadonlyArray<MotEntry>): string {
  let out = '';
  for (const e of entries) {
    const [x1, y1, x2, y2] = e.bbox;
    const fields = [e.frame, e.id, x1, y1, x2 - x1, y2 - y1, e.score, e.classId, e.visibility];
    out += `${fields.map(formatNumber).join(',')},-1\n`;
  }
  return out;
}

function formatNumber(value: number): string {
  return String(Math.round(value * 1e6) / 1e6);
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
export function filterGt(entries: ReadonlyArray<MotEntry>, options: GtFilterOptions): MotEntry[] {
  const { requireConsidered = false, classIds, minVisibility } = options;
  return entries.filter((e) => {
    if (requireConsidered && e.score === 0) return false;
    if (classIds !== undefined && !classIds.includes(e.classId)) return false;
    if (minVisibility !== undefined && e.visibility < minVisibility) return false;
    return true;
  });
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
  entries: ReadonlyArray<MotEntry>,
  numFrames?: number,
): Detection[][] {
  let maxFrame = 0;
  for (const e of entries) maxFrame = Math.max(maxFrame, e.frame);
  const n = numFrames ?? maxFrame;

  const out: Detection[][] = Array.from({ length: n }, () => []);
  for (const e of entries) {
    const frame = out[e.frame - 1];
    if (frame === undefined) continue;
    const det: Detection = { bbox: e.bbox, score: e.score };
    if (e.classId !== -1) det.classId = e.classId;
    frame.push(det);
  }
  return out;
}
