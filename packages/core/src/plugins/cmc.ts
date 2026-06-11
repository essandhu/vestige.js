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
export function sampleBilinear(frame: GrayFrame, x: number, y: number): number {
  const { data, width, height } = frame;
  const cx = x < 0 ? 0 : x > width - 1 ? width - 1 : x;
  const cy = y < 0 ? 0 : y > height - 1 ? height - 1 : y;
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const x1 = x0 + 1 < width ? x0 + 1 : x0;
  const y1 = y0 + 1 < height ? y0 + 1 : y0;
  const fx = cx - x0;
  const fy = cy - y0;

  const v00 = data[y0 * width + x0] ?? 0;
  const v10 = data[y0 * width + x1] ?? 0;
  const v01 = data[y1 * width + x0] ?? 0;
  const v11 = data[y1 * width + x1] ?? 0;
  const top = v00 + fx * (v10 - v00);
  const bottom = v01 + fx * (v11 - v01);
  return top + fy * (bottom - top);
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
  frame: GrayFrame,
  options?: ShiTomasiOptions,
): Array<readonly [number, number]> {
  const maxCorners = options?.maxCorners ?? 1000;
  const qualityLevel = options?.qualityLevel ?? 0.01;
  const minDistance = options?.minDistance ?? 1;
  const blockSize = options?.blockSize ?? 3;
  const { data, width, height } = frame;

  // Central-difference gradients; the one-pixel border stays zero.
  const ix = new Float64Array(width * height);
  const iy = new Float64Array(width * height);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      ix[i] = ((data[i + 1] ?? 0) - (data[i - 1] ?? 0)) / 2;
      iy[i] = ((data[i + width] ?? 0) - (data[i - width] ?? 0)) / 2;
    }
  }

  // Minimum eigenvalue of the structure tensor summed over the block window.
  const r = (blockSize - 1) >> 1;
  const margin = r + 1;
  const response = new Float64Array(width * height);
  let maxResponse = 0;
  for (let y = margin; y < height - margin; y++) {
    for (let x = margin; x < width - margin; x++) {
      let sxx = 0;
      let sxy = 0;
      let syy = 0;
      for (let wy = -r; wy <= r; wy++) {
        const row = (y + wy) * width + x;
        for (let wx = -r; wx <= r; wx++) {
          const gx = ix[row + wx] ?? 0;
          const gy = iy[row + wx] ?? 0;
          sxx += gx * gx;
          sxy += gx * gy;
          syy += gy * gy;
        }
      }
      const diff = sxx - syy;
      const lambdaMin = (sxx + syy - Math.sqrt(diff * diff + 4 * sxy * sxy)) / 2;
      response[y * width + x] = lambdaMin;
      if (lambdaMin > maxResponse) maxResponse = lambdaMin;
    }
  }
  if (maxResponse <= 0) return [];

  // Strongest-first with deterministic index tie-break, then greedy
  // min-distance suppression. CMC runs once per frame, outside the per-track
  // hot paths, so the O(candidates · accepted) loop is acceptable.
  const threshold = maxResponse * qualityLevel;
  const candidates: number[] = [];
  for (let i = 0; i < response.length; i++) {
    if ((response[i] ?? 0) >= threshold) candidates.push(i);
  }
  candidates.sort((a, b) => (response[b] ?? 0) - (response[a] ?? 0) || a - b);

  const minDistSq = minDistance * minDistance;
  const corners: Array<readonly [number, number]> = [];
  for (const i of candidates) {
    if (corners.length >= maxCorners) break;
    const x = i % width;
    const y = (i - x) / width;
    let suppressed = false;
    for (const [ax, ay] of corners) {
      const dx = ax - x;
      const dy = ay - y;
      if (dx * dx + dy * dy < minDistSq) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) corners.push([x, y]);
  }
  return corners;
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
  prev: GrayFrame,
  curr: GrayFrame,
  points: ReadonlyArray<readonly [number, number]>,
  options?: LkOptions,
): Array<readonly [number, number] | null> {
  const winSize = options?.winSize ?? 21;
  const maxLevel = options?.maxLevel ?? 3;
  const maxIterations = options?.maxIterations ?? 30;
  const epsilon = options?.epsilon ?? 0.01;

  const r = (winSize - 1) >> 1;
  const prevPyramid = buildPyramid(prev, maxLevel, r);
  const currPyramid = buildPyramid(curr, maxLevel, r);
  const epsSq = epsilon * epsilon;

  return points.map(([px, py]) =>
    trackOnePoint(prevPyramid, currPyramid, px, py, r, maxIterations, epsSq),
  );
}

/**
 * Build the Gaussian image pyramid for pyramidal LK: level 0 is the input,
 * each level above is a 5-tap-Gaussian-blurred 2× decimation. Levels too
 * small to host an LK integration window (radius `r`, plus the bilinear and
 * gradient margins) are not built — coarse levels degrade gracefully rather
 * than failing every border point.
 */
function buildPyramid(frame: GrayFrame, maxLevel: number, r: number): GrayFrame[] {
  const levels: GrayFrame[] = [frame];
  const minDim = 2 * (r + 2);
  for (let level = 1; level <= maxLevel; level++) {
    const src = levels[levels.length - 1];
    if (src === undefined) break;
    const width = Math.ceil(src.width / 2);
    const height = Math.ceil(src.height / 2);
    if (width < minDim || height < minDim) break;
    levels.push(downsample(src, width, height));
  }
  return levels;
}

/** 5-tap Gaussian ([1,4,6,4,1]/16, separable) blur + 2× decimation. */
function downsample(src: GrayFrame, width: number, height: number): GrayFrame {
  const out = new Float32Array(width * height);
  const sw = src.width;
  const sh = src.height;
  const data = src.data;
  for (let y = 0; y < height; y++) {
    const sy = 2 * y;
    for (let x = 0; x < width; x++) {
      const sx = 2 * x;
      let acc = 0;
      for (let ky = -2; ky <= 2; ky++) {
        let yy = sy + ky;
        if (yy < 0) yy = 0;
        else if (yy >= sh) yy = sh - 1;
        const row = yy * sw;
        const wy = GAUSS5[ky + 2] ?? 0;
        for (let kx = -2; kx <= 2; kx++) {
          let xx = sx + kx;
          if (xx < 0) xx = 0;
          else if (xx >= sw) xx = sw - 1;
          acc += wy * (GAUSS5[kx + 2] ?? 0) * (data[row + xx] ?? 0);
        }
      }
      out[y * width + x] = acc / 256;
    }
  }
  return { data: out, width, height };
}

const GAUSS5 = [1, 4, 6, 4, 1];

/**
 * Iterative LK refinement of one point across the pyramid, coarse → fine
 * (Bouguet 2000 §2). The displacement found at each level seeds the next
 * finer level (×2). A level whose integration window falls outside the frame
 * is skipped; failure is only declared if the base level is unusable, the
 * gradient structure is degenerate, or the result leaves the frame.
 */
function trackOnePoint(
  prevPyramid: ReadonlyArray<GrayFrame>,
  currPyramid: ReadonlyArray<GrayFrame>,
  px: number,
  py: number,
  r: number,
  maxIterations: number,
  epsSq: number,
): readonly [number, number] | null {
  const win = 2 * r + 1;
  const winArea = win * win;
  const prevVals = new Float64Array(winArea);
  const gradX = new Float64Array(winArea);
  const gradY = new Float64Array(winArea);

  let dx = 0;
  let dy = 0;
  for (let level = prevPyramid.length - 1; level >= 0; level--) {
    const prevL = prevPyramid[level];
    const currL = currPyramid[level];
    if (prevL === undefined || currL === undefined) return null;

    const scale = 1 / (1 << level);
    const cx = px * scale;
    const cy = py * scale;

    // The window (±r) plus the bilinear+gradient margin (±1) must fit.
    const usable =
      cx - r - 1 >= 0 &&
      cx + r + 1 <= prevL.width - 1 &&
      cy - r - 1 >= 0 &&
      cy + r + 1 <= prevL.height - 1;
    if (!usable) {
      if (level === 0) return null;
      dx *= 2;
      dy *= 2;
      continue;
    }

    // Template values and spatial gradients from the previous frame,
    // computed once per level (Bouguet's constant-gradient approximation).
    let sxx = 0;
    let sxy = 0;
    let syy = 0;
    let k = 0;
    for (let wy = -r; wy <= r; wy++) {
      for (let wx = -r; wx <= r; wx++) {
        const sx = cx + wx;
        const sy = cy + wy;
        prevVals[k] = sampleBilinear(prevL, sx, sy);
        const gx = (sampleBilinear(prevL, sx + 1, sy) - sampleBilinear(prevL, sx - 1, sy)) / 2;
        const gy = (sampleBilinear(prevL, sx, sy + 1) - sampleBilinear(prevL, sx, sy - 1)) / 2;
        gradX[k] = gx;
        gradY[k] = gy;
        sxx += gx * gx;
        sxy += gx * gy;
        syy += gy * gy;
        k++;
      }
    }
    const det = sxx * syy - sxy * sxy;
    if (det < 1e-9) return null;

    for (let iter = 0; iter < maxIterations; iter++) {
      let bx = 0;
      let by = 0;
      k = 0;
      for (let wy = -r; wy <= r; wy++) {
        for (let wx = -r; wx <= r; wx++) {
          const diff = (prevVals[k] ?? 0) - sampleBilinear(currL, cx + dx + wx, cy + dy + wy);
          bx += diff * (gradX[k] ?? 0);
          by += diff * (gradY[k] ?? 0);
          k++;
        }
      }
      const stepX = (syy * bx - sxy * by) / det;
      const stepY = (sxx * by - sxy * bx) / det;
      dx += stepX;
      dy += stepY;
      if (stepX * stepX + stepY * stepY <= epsSq) break;
    }

    if (level > 0) {
      dx *= 2;
      dy *= 2;
    }
  }

  const base = prevPyramid[0];
  if (base === undefined) return null;
  const qx = px + dx;
  const qy = py + dy;
  if (qx < 0 || qx > base.width - 1 || qy < 0 || qy > base.height - 1) return null;
  return [qx, qy];
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
  src: ReadonlyArray<readonly [number, number]>,
  dst: ReadonlyArray<readonly [number, number]>,
  options?: SimilarityRansacOptions,
): Float64Array | null {
  const ransacThreshold = options?.ransacThreshold ?? 3;
  const ransacIterations = options?.ransacIterations ?? 200;
  const minPoints = options?.minPoints ?? 4;
  const seed = options?.seed ?? 42;

  if (src.length !== dst.length) {
    throw new Error(
      `estimateSimilarity: src and dst are not parallel arrays (${src.length} vs ${dst.length})`,
    );
  }
  const n = src.length;
  if (n < minPoints) return null;

  const thresholdSq = ransacThreshold * ransacThreshold;
  const random = mulberry32(seed);
  let bestInliers: number[] | null = null;

  for (let iter = 0; iter < ransacIterations; iter++) {
    const i = Math.floor(random() * n);
    const j = Math.floor(random() * n);
    if (i === j) continue;
    const model = similarityFromTwoPairs(src[i], dst[i], src[j], dst[j]);
    if (model === null) continue;

    const [a, b, tx, ty] = model;
    const inliers: number[] = [];
    for (let k = 0; k < n; k++) {
      const [x, y] = src[k] ?? [0, 0];
      const [u, v] = dst[k] ?? [0, 0];
      const ex = a * x - b * y + tx - u;
      const ey = b * x + a * y + ty - v;
      if (ex * ex + ey * ey <= thresholdSq) inliers.push(k);
    }
    if (bestInliers === null || inliers.length > bestInliers.length) bestInliers = inliers;
  }

  if (bestInliers === null || bestInliers.length < minPoints) return null;
  return fitSimilarityLeastSquares(src, dst, bestInliers);
}

/**
 * Exact 4-DOF similarity from two point correspondences via the complex
 * ratio `(q2 − q1)/(p2 − p1)`. Returns `[a, b, tx, ty]`, or null when the
 * source points coincide.
 */
function similarityFromTwoPairs(
  p1: readonly [number, number] | undefined,
  q1: readonly [number, number] | undefined,
  p2: readonly [number, number] | undefined,
  q2: readonly [number, number] | undefined,
): readonly [number, number, number, number] | null {
  if (p1 === undefined || q1 === undefined || p2 === undefined || q2 === undefined) return null;
  const dpx = p2[0] - p1[0];
  const dpy = p2[1] - p1[1];
  const den = dpx * dpx + dpy * dpy;
  if (den < 1e-12) return null;

  const dqx = q2[0] - q1[0];
  const dqy = q2[1] - q1[1];
  const a = (dqx * dpx + dqy * dpy) / den;
  const b = (dqy * dpx - dqx * dpy) / den;
  const tx = q1[0] - (a * p1[0] - b * p1[1]);
  const ty = q1[1] - (b * p1[0] + a * p1[1]);
  return [a, b, tx, ty];
}

/**
 * Closed-form least-squares similarity over the inlier set (the standard
 * mean-centered complex regression). Returns the 2×3 matrix, or null when
 * the inlier sources are all coincident.
 */
function fitSimilarityLeastSquares(
  src: ReadonlyArray<readonly [number, number]>,
  dst: ReadonlyArray<readonly [number, number]>,
  inliers: ReadonlyArray<number>,
): Float64Array | null {
  const n = inliers.length;
  let mx = 0;
  let my = 0;
  let mu = 0;
  let mv = 0;
  for (const k of inliers) {
    const [x, y] = src[k] ?? [0, 0];
    const [u, v] = dst[k] ?? [0, 0];
    mx += x;
    my += y;
    mu += u;
    mv += v;
  }
  mx /= n;
  my /= n;
  mu /= n;
  mv /= n;

  let sNorm = 0;
  let sa = 0;
  let sb = 0;
  for (const k of inliers) {
    const [x, y] = src[k] ?? [0, 0];
    const [u, v] = dst[k] ?? [0, 0];
    const cx = x - mx;
    const cy = y - my;
    const cu = u - mu;
    const cv = v - mv;
    sNorm += cx * cx + cy * cy;
    sa += cx * cu + cy * cv;
    sb += cx * cv - cy * cu;
  }
  if (sNorm < 1e-12) return null;

  const a = sa / sNorm;
  const b = sb / sNorm;
  const tx = mu - (a * mx - b * my);
  const ty = mv - (b * mx + a * my);
  return Float64Array.from([a, -b, tx, b, a, ty]);
}

/**
 * Deterministic 32-bit PRNG (mulberry32). Quality is ample for RANSAC index
 * sampling, and a fixed seed keeps {@link estimateSimilarity} bit-reproducible
 * (ARCHITECTURE.md §2.5).
 */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Apply a 2×3 affine matrix (the {@link CmcProvider.estimate} output shape)
 * to an xyxy bbox by warping its two defining corners. The result is
 * re-normalized so `x1 ≤ x2`, `y1 ≤ y2` even under reflections.
 *
 * This is the helper `BotSortTracker` uses to carry predicted track boxes
 * into the current frame's coordinates before association.
 */
export function warpBBox(matrix: Float64Array, bbox: BBox): BBox {
  const a = matrix[0] ?? 1;
  const b = matrix[1] ?? 0;
  const tx = matrix[2] ?? 0;
  const c = matrix[3] ?? 0;
  const d = matrix[4] ?? 1;
  const ty = matrix[5] ?? 0;
  const [x1, y1, x2, y2] = bbox;

  const u1 = a * x1 + b * y1 + tx;
  const v1 = c * x1 + d * y1 + ty;
  const u2 = a * x2 + b * y2 + tx;
  const v2 = c * x2 + d * y2 + ty;
  return [Math.min(u1, u2), Math.min(v1, v2), Math.max(u1, u2), Math.max(v1, v2)];
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
  private readonly options: SparseOpticalFlowCmcOptions;

  constructor(options: SparseOpticalFlowCmcOptions = {}) {
    this.options = options;
  }

  async estimate(prevFrame: unknown, currFrame: unknown): Promise<Float64Array | null> {
    const prev = toGrayFrame(prevFrame, 'prevFrame');
    const curr = toGrayFrame(currFrame, 'currFrame');
    if (prev.width !== curr.width || prev.height !== curr.height) {
      throw new Error(
        `SparseOpticalFlowCmc: frame sizes differ (${prev.width}×${prev.height} vs ${curr.width}×${curr.height})`,
      );
    }

    const minPoints = this.options.minPoints ?? 4;
    const corners = shiTomasiCorners(prev, this.options);
    if (corners.length < minPoints) return null;

    const tracked = trackPointsLk(prev, curr, corners, this.options);
    const src: Array<readonly [number, number]> = [];
    const dst: Array<readonly [number, number]> = [];
    for (let i = 0; i < corners.length; i++) {
      const q = tracked[i];
      const p = corners[i];
      if (q !== null && q !== undefined && p !== undefined) {
        src.push(p);
        dst.push(q);
      }
    }
    if (src.length < minPoints) return null;

    return estimateSimilarity(src, dst, this.options);
  }
}

function toGrayFrame(value: unknown, which: 'prevFrame' | 'currFrame'): GrayFrame {
  if (typeof value === 'object' && value !== null) {
    const frame = value as Partial<GrayFrame>;
    if (
      isPixelBuffer(frame.data) &&
      typeof frame.width === 'number' &&
      typeof frame.height === 'number' &&
      Number.isInteger(frame.width) &&
      Number.isInteger(frame.height) &&
      frame.width > 0 &&
      frame.height > 0 &&
      frame.data.length === frame.width * frame.height
    ) {
      return frame as GrayFrame;
    }
  }
  throw new Error(
    `SparseOpticalFlowCmc: ${which} is not a grayscale frame — expected { data, width, height } with data.length === width · height`,
  );
}

function isPixelBuffer(data: unknown): data is GrayFrame['data'] {
  return (
    data instanceof Uint8Array ||
    data instanceof Uint8ClampedArray ||
    data instanceof Float32Array ||
    data instanceof Float64Array
  );
}
