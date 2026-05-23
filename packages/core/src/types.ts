/**
 * Canonical bounding box format: [x1, y1, x2, y2] (top-left, bottom-right) in pixel coordinates.
 * Readonly tuple to prevent accidental mutation of detector outputs.
 */
export type BBox = readonly [number, number, number, number];

/**
 * Track lifecycle phase.
 *
 * - `tentative`: insufficient hits for confirmation; not reported externally.
 * - `confirmed`: passed the min-hits threshold; publicly exposed by `update()`.
 * - `lost`: previously confirmed; currently unmatched; retained for re-association.
 * - `removed`: scheduled for cleanup; will not appear in subsequent updates.
 */
export type TrackState = 'tentative' | 'confirmed' | 'lost' | 'removed';

/**
 * A single detection from an upstream detector.
 *
 * `TPayload` lets users thread arbitrary per-detection metadata (embeddings,
 * masks, etc.) through the tracker without the library inspecting it.
 */
export interface Detection<TPayload = unknown> {
  bbox: BBox;
  score: number;
  classId?: number;
  payload?: TPayload;
}

/**
 * A Detection augmented with tracker-assigned identity and lifecycle state.
 * Consumers iterate one shape; the tracker's job is to add `id` + lifecycle info.
 */
export interface Track<TPayload = unknown> extends Detection<TPayload> {
  id: number;
  age: number;
  hits: number;
  timeSinceUpdate: number;
  state: TrackState;
}

/**
 * Per-frame interface shared by every tracker implementation.
 * Constructor options differ per tracker (see §2.2 of ARCHITECTURE.md);
 * the runtime contract does not.
 */
export interface Tracker<TPayload = unknown> {
  update(detections: ReadonlyArray<Detection<TPayload>>): Track<TPayload>[];
  getActiveTracks(): Track<TPayload>[];
  getLostTracks(): Track<TPayload>[];
  reset(): void;
  readonly frameIndex: number;
}
