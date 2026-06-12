export type {
  CmcProvider,
  GrayFrame,
  SparseOpticalFlowCmcOptions,
} from './plugins/cmc.js';
export { SparseOpticalFlowCmc, warpBBox } from './plugins/cmc.js';
export type { ByteTrackerOptions } from './trackers/bytetrack.js';
export { ByteTracker } from './trackers/bytetrack.js';
export type { OcSortAsoFunc, OcSortTrackerOptions } from './trackers/ocsort.js';
export { OcSortTracker } from './trackers/ocsort.js';
export type { SortTrackerOptions } from './trackers/sort.js';
export { SortTracker } from './trackers/sort.js';
export type { BBox, Detection, Track, Tracker, TrackState } from './types.js';

// Additional trackers are exported here as they land:
//   export { BotSortTracker } from './trackers/botsort.js';
//
// Geometry primitives are exposed via the `vestige.js/geometry` subpath
// (see ARCHITECTURE.md §11.7), not from this root entry.
