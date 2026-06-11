/**
 * Camera Motion Compensation plugin interface and the default from-scratch
 * sparse-optical-flow implementation (ARCHITECTURE.md §8.2.5).
 *
 * The pipeline mirrors BoT-SORT's `sparseOptFlow` GMC mode
 * (`NirAharon/BoT-SORT`, `tracker/gmc.py`): Shi-Tomasi corner selection
 * (Shi & Tomasi, CVPR 1994 — `cv2.goodFeaturesToTrack`), pyramidal
 * Lucas-Kanade tracking (Bouguet 2000 — `cv2.calcOpticalFlowPyrLK`), and a
 * RANSAC 4-DOF similarity fit (`cv2.estimateAffinePartial2D`). Everything is
 * implemented in pure TS; the building blocks are exported individually so
 * the numerics are unit-testable in isolation.
 */

import type { BBox } from '../types.js';

/**
 * Estimates camera motion between consecutive frames. Used by
 * `BotSortTracker` to correct track predictions for camera movement before
 * association. See ARCHITECTURE.md §8.2.5.
 */
export interface CmcProvider {
  /**
   * Estimate the affine warp from the previous frame to the current frame.
   * Returns a 2×3 affine matrix as a 6-element row-major Float64Array
   * `[a, b, tx, c, d, ty]` (a point `[x, y]` maps to
   * `[a·x + b·y + tx, c·x + d·y + ty]`), or `null` if estimation failed
   * (e.g. insufficient trackable features).
   *
   * Frames are `unknown` to keep the interface environment-agnostic;
   * implementations document the concrete frame type they accept.
   */
  estimate(prevFrame: unknown, currFrame: unknown): Promise<Float64Array | null>;
}

/**
 * The concrete frame shape {@link SparseOpticalFlowCmc} consumes: a single-
 * channel (grayscale) intensity buffer in row-major order, `width × height`
 * pixels. Both browser (`ImageData` converted to luma) and Node (decoded
 * buffer) sources reduce to this shape; the conversion is the caller's
 * responsibility — core stays free of DOM and `fs` dependencies.
 */
export interface GrayFrame {
  /** Row-major intensity samples; `data[y * width + x]`. Length `width · height`. */
  readonly data: Uint8Array | Uint8ClampedArray | Float32Array | Float64Array;
  readonly width: number;
  readonly height: number;
}

/**
 * Options for {@link shiTomasiCorners}. Defaults follow BoT-SORT's
 * `goodFeaturesToTrack` call (`gmc.py`: maxCorners 1000, qualityLevel 0.01,
 * minDistance 1, blockSize 3).
 */
export interface ShiTomasiOptions {
  /** Maximum number of corners returned, strongest first. Default 1000. */
  readonly maxCorners?: number;
  /**
   * Fraction of the strongest corner's response below which candidates are
   * rejected. Default 0.01.
   */
  readonly qualityLevel?: number;
  /** Minimum Euclidean distance between returned corners, in pixels. Default 1. */
  readonly minDistance?: number;
  /** Side length of the structure-tensor integration window. Default 3 (must be odd). */
  readonly blockSize?: number;
}

/**
 * Options for {@link trackPointsLk}. Defaults follow OpenCV's
 * `calcOpticalFlowPyrLK` (winSize 21, maxLevel 3, 30 iterations, eps 0.01).
 */
export interface LkOptions {
  /** Side length of the tracking window, in pixels. Default 21 (must be odd). */
  readonly winSize?: number;
  /**
   * Number of pyramid levels above the base image. Levels whose integration
   * window no longer fits are skipped automatically. Default 3.
   */
  readonly maxLevel?: number;
  /** Iteration cap per pyramid level. Default 30. */
  readonly maxIterations?: number;
  /** Convergence threshold on the per-iteration displacement norm. Default 0.01. */
  readonly epsilon?: number;
}

/** Options for {@link estimateSimilarity}. */
export interface SimilarityRansacOptions {
  /** Inlier reprojection-distance threshold, in pixels. Default 3. */
  readonly ransacThreshold?: number;
  /** Number of RANSAC minimal-sample iterations. Default 200. */
  readonly ransacIterations?: number;
  /**
   * Minimum point pairs required to attempt (and inliers required to accept)
   * an estimate; below it the fit returns `null`. Default 4.
   */
  readonly minPoints?: number;
  /**
   * PRNG seed for RANSAC sampling. Fixed by default (42) so estimation is
   * deterministic — same inputs, bit-identical output (ARCHITECTURE.md §2.5).
   */
  readonly seed?: number;
}

/** Options for {@link SparseOpticalFlowCmc}: the union of all stage options. */
export interface SparseOpticalFlowCmcOptions
  extends ShiTomasiOptions,
    LkOptions,
    SimilarityRansacOptions {}

/**
 * Sample a {@link GrayFrame} at a fractional position by bilinear
 * interpolation. Coordinates are clamped to the valid sample rectangle
 * `[0, width − 1] × [0, height − 1]`, so out-of-range reads return the
 * nearest edge value rather than NaN.
 */
export function sampleBilinear(_frame: GrayFrame, _x: number, _y: number): number {
  throw new Error('not implemented');
}

/**
 * Shi-Tomasi "good features to track": corners are pixels whose structure
 * tensor (summed over `blockSize²`, gradients by central difference) has a
 * large minimum eigenvalue. Candidates below `qualityLevel · maxResponse`
 * are dropped; survivors are returned strongest-first after greedy
 * `minDistance` suppression, capped at `maxCorners`.
 *
 * Returns `[x, y]` pixel positions. A frame with no intensity variation
 * returns an empty array. Deterministic: ties in response break toward the
 * smaller pixel index (ARCHITECTURE.md §2.5).
 */
export function shiTomasiCorners(
  _frame: GrayFrame,
  _options?: ShiTomasiOptions,
): Array<readonly [number, number]> {
  throw new Error('not implemented');
}

/**
 * Pyramidal Lucas-Kanade: track `points` from `prev` to `curr`
 * (Bouguet 2000). Each point gets the iterative LK refinement at every
 * pyramid level from coarsest to finest, doubling the displacement on the
 * way down.
 *
 * Returns one entry per input point: the tracked `[x, y]` in `curr`, or
 * `null` when tracking failed (integration window outside the frame,
 * degenerate gradient structure, or divergence out of bounds).
 */
export function trackPointsLk(
  _prev: GrayFrame,
  _curr: GrayFrame,
  _points: ReadonlyArray<readonly [number, number]>,
  _options?: LkOptions,
): Array<readonly [number, number] | null> {
  throw new Error('not implemented');
}

/**
 * Robustly fit a 4-DOF similarity transform (uniform scale + rotation +
 * translation) mapping `src[i] → dst[i]`, the same model as OpenCV's
 * `estimateAffinePartial2D`: RANSAC over 2-point minimal samples, then a
 * least-squares refit on the inliers of the best hypothesis.
 *
 * Returns the 2×3 row-major matrix `[a, −b, tx, b, a, ty]`, or `null` when
 * `src.length < minPoints`, the points are degenerate (coincident), or the
 * best hypothesis has fewer than `minPoints` inliers. Deterministic for
 * fixed `seed`.
 *
 * @param src points in the previous frame
 * @param dst tracked positions in the current frame, parallel to `src`
 */
export function estimateSimilarity(
  _src: ReadonlyArray<readonly [number, number]>,
  _dst: ReadonlyArray<readonly [number, number]>,
  _options?: SimilarityRansacOptions,
): Float64Array | null {
  throw new Error('not implemented');
}

/**
 * Apply a 2×3 affine matrix (the {@link CmcProvider.estimate} output shape)
 * to an xyxy bbox by warping its two defining corners. The result is
 * re-normalized so `x1 ≤ x2`, `y1 ≤ y2` even under reflections.
 *
 * This is the helper `BotSortTracker` uses to carry predicted track boxes
 * into the current frame's coordinates before association.
 */
export function warpBBox(_matrix: Float64Array, _bbox: BBox): BBox {
  throw new Error('not implemented');
}

/**
 * Default {@link CmcProvider}: sparse optical flow over {@link GrayFrame}s.
 *
 * Pipeline per `estimate(prev, curr)` call:
 *
 * 1. {@link shiTomasiCorners} on `prev`.
 * 2. {@link trackPointsLk} of those corners into `curr`.
 * 3. {@link estimateSimilarity} on the successfully tracked pairs.
 *
 * Returns `null` (a documented "estimation failed" outcome, not an error)
 * when any stage yields fewer than `minPoints` usable points — e.g. flat,
 * textureless frames. Throws (`/grayscale/i`) when an argument is not a
 * {@link GrayFrame}, when `data.length !== width · height`, or when the two
 * frames' dimensions differ.
 *
 * The provider is stateless across calls and deterministic: the same frame
 * pair always produces the bit-identical matrix (ARCHITECTURE.md §2.5).
 */
export class SparseOpticalFlowCmc implements CmcProvider {
  constructor(_options: SparseOpticalFlowCmcOptions = {}) {}

  estimate(_prevFrame: unknown, _currFrame: unknown): Promise<Float64Array | null> {
    throw new Error('not implemented');
  }
}
