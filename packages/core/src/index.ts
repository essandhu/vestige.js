export type { BBox, Detection, Track, Tracker, TrackState } from './types.js';

// Trackers are exported here as they land:
//   export { SortTracker } from './trackers/sort.js';
//   export { ByteTracker } from './trackers/bytetrack.js';
//   export { OcSortTracker } from './trackers/ocsort.js';
//   export { BotSortTracker } from './trackers/botsort.js';
//
// Geometry primitives are exposed via the `vestige.js/geometry` subpath
// (see ARCHITECTURE.md §11.7), not from this root entry.
