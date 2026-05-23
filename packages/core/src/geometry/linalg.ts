/**
 * Linear algebra primitives for the Kalman filter and cost-matrix machinery.
 *
 * All matrices are row-major Float64Array. Sizes are passed explicitly rather
 * than inferred — this matches the "fixed-size unrolled routines" performance
 * strategy described in ARCHITECTURE.md §5.3 while keeping the contract generic.
 *
 * Hot paths (8x8 cov updates, 4x8 measurement updates) may eventually be
 * specialized; the public contract should remain identical.
 */

/**
 * Row-major matrix multiplication `C = A * B`.
 * A is (m × k), B is (k × n), C is (m × n).
 *
 * If `out` is provided it must be at least `m * n` long; otherwise a new
 * Float64Array is allocated.
 */
export function matMul(
  _a: Float64Array,
  _b: Float64Array,
  _m: number,
  _k: number,
  _n: number,
  _out?: Float64Array,
): Float64Array {
  throw new Error('not implemented');
}

/**
 * Row-major transpose. A is (m × n), result is (n × m).
 */
export function transpose(
  _a: Float64Array,
  _m: number,
  _n: number,
  _out?: Float64Array,
): Float64Array {
  throw new Error('not implemented');
}

/**
 * Matrix-vector product `y = A * x`. A is (m × n), x is length n, y is length m.
 */
export function matVec(
  _a: Float64Array,
  _x: Float64Array,
  _m: number,
  _n: number,
  _out?: Float64Array,
): Float64Array {
  throw new Error('not implemented');
}

/**
 * Outer product `M = x * yᵀ`. x is length m, y is length n, M is (m × n).
 */
export function outerProduct(
  _x: Float64Array,
  _y: Float64Array,
  _out?: Float64Array,
): Float64Array {
  throw new Error('not implemented');
}

/**
 * In-place addition `A := A + B`. Returns A.
 */
export function addInPlace(_a: Float64Array, _b: Float64Array): Float64Array {
  throw new Error('not implemented');
}

/**
 * In-place subtraction `A := A - B`. Returns A.
 */
export function subInPlace(_a: Float64Array, _b: Float64Array): Float64Array {
  throw new Error('not implemented');
}

/**
 * Cholesky decomposition of a symmetric positive-definite matrix A (n × n, row-major).
 * Returns the lower-triangular factor L such that A = L * Lᵀ, also as row-major Float64Array.
 * Upper-triangular cells of the result are zero.
 *
 * Throws if A is not positive definite (a non-positive diagonal arises during factorization).
 */
export function cholesky(_a: Float64Array, _n: number): Float64Array {
  throw new Error('not implemented');
}

/**
 * Solve `A x = b` for x, given the Cholesky factor L of A (A = L Lᵀ).
 * L is (n × n) lower-triangular row-major; b is length n; returns x of length n.
 *
 * This is the numerically-stable inversion path used by the Kalman filter
 * innovation-covariance solve (ARCHITECTURE.md §5.3). Do not compute A⁻¹ explicitly.
 */
export function choleskySolve(_L: Float64Array, _b: Float64Array, _n: number): Float64Array {
  throw new Error('not implemented');
}
