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
export function parseSeqinfo(text: string): SeqInfo {
  const values = new Map<string, string>();
  for (const raw of text.split('\n')) {
    const line = raw.trim();
    if (line === '' || line.startsWith(';') || line.startsWith('[')) continue;
    const eq = line.indexOf('=');
    if (eq === -1) continue;
    values.set(line.slice(0, eq).trim(), line.slice(eq + 1).trim());
  }

  const name = values.get('name');
  if (name === undefined || name === '') {
    throw new Error('malformed seqinfo.ini: missing name');
  }
  const seqLength = optionalNumber(values, 'seqLength');
  if (seqLength === undefined) {
    throw new Error('malformed seqinfo.ini: missing or invalid seqLength');
  }

  const info: { -readonly [K in keyof SeqInfo]: SeqInfo[K] } = { name, seqLength };
  const frameRate = optionalNumber(values, 'frameRate');
  const imWidth = optionalNumber(values, 'imWidth');
  const imHeight = optionalNumber(values, 'imHeight');
  if (frameRate !== undefined) info.frameRate = frameRate;
  if (imWidth !== undefined) info.imWidth = imWidth;
  if (imHeight !== undefined) info.imHeight = imHeight;
  return info;
}

function optionalNumber(values: Map<string, string>, key: string): number | undefined {
  const raw = values.get(key);
  if (raw === undefined || raw === '') return undefined;
  const value = Number(raw);
  return Number.isFinite(value) ? value : undefined;
}
