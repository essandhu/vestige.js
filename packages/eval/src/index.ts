/**
 * MOTChallenge evaluation harness for vestige.js (ARCHITECTURE.md §10).
 *
 * - `motchallenge/` — gt/det/result file parsing, formatting, gt filtering.
 * - `metrics/` — CLEAR-MOT (MOTA/MOTP/…), ID (IDF1), and HOTA, each computed
 *   per the `JonathonLuiten/TrackEval` reference (CONTRIBUTING.md §4.2).
 * - `runner` — drives a `Tracker` over a sequence and glues the two together.
 */
export * from './metrics/index.js';
export * from './motchallenge/index.js';
export { evalFramesFromEntries, runTracker, tracksToMotEntries } from './runner.js';
