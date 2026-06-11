/**
 * Single bridge point for the eval package's dependency on the core package.
 *
 * Imports resolve straight into the core *source* (not the built `vestige.js`
 * dist) so `pnpm test` / `pnpm typecheck` work in a fresh checkout without a
 * prior `pnpm build`, and so eval can reach internals (`solveLsap`) that the
 * published package deliberately does not export yet. This is an
 * intra-monorepo convenience for a private package; nothing here widens the
 * public API of `vestige.js`. See `docs/decisions/0004-eval-package.md`.
 */
export { iou, iouMatrix } from '../../core/src/geometry/iou.js';
export type { LsapResult } from '../../core/src/solvers/hungarian.js';
export { solveLsap } from '../../core/src/solvers/hungarian.js';
export { SortTracker } from '../../core/src/trackers/sort.js';
export type { BBox, Detection, Track, Tracker } from '../../core/src/types.js';
