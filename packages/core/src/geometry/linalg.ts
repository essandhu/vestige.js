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

// biome-ignore-all lint/style/noNonNullAssertion: indices in these loops are
// bounded by the size parameters and the input length contracts. Asserting at
// each read avoids `number | undefined` from `noUncheckedIndexedAccess` in
// tight numerical loops without runtime cost.

/**
 * Row-major matrix multiplication `C = A * B`.
 * A is (m × k), B is (k × n), C is (m × n).
 *
 * If `out` is provided it must be at least `m * n` long; otherwise a new
 * Float64Array is allocated.
 */
export function matMul(
  a: Float64Array,
  b: Float64Array,
  m: number,
  k: number,
  n: number,
  out?: Float64Array,
): Float64Array {
  const c = out ?? new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    const aRow = i * k;
    const cRow = i * n;
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) {
        s += a[aRow + p]! * b[p * n + j]!;
      }
      c[cRow + j] = s;
    }
  }

  return c;
}

/**
 * Row-major transpose. A is (m × n), result is (n × m).
 */
export function transpose(a: Float64Array, m: number, n: number, out?: Float64Array): Float64Array {
  const t = out ?? new Float64Array(n * m);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      t[j * m + i] = a[i * n + j]!;
    }
  }

  return t;
}

/**
 * Matrix-vector product `y = A * x`. A is (m × n), x is length n, y is length m.
 */
export function matVec(
  a: Float64Array,
  x: Float64Array,
  m: number,
  n: number,
  out?: Float64Array,
): Float64Array {
  const y = out ?? new Float64Array(m);
  for (let i = 0; i < m; i++) {
    const row = i * n;
    let s = 0;
    for (let j = 0; j < n; j++) {
      s += a[row + j]! * x[j]!;
    }
    y[i] = s;
  }

  return y;
}

/**
 * Outer product `M = x * yᵀ`. x is length m, y is length n, M is (m × n).
 */
export function outerProduct(x: Float64Array, y: Float64Array, out?: Float64Array): Float64Array {
  const m = x.length;
  const n = y.length;
  const M = out ?? new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    const xi = x[i]!;
    const row = i * n;
    for (let j = 0; j < n; j++) {
      M[row + j] = xi * y[j]!;
    }
  }

  return M;
}

/**
 * In-place addition `A := A + B`. Returns A.
 */
export function addInPlace(a: Float64Array, b: Float64Array): Float64Array {
  for (let i = 0; i < a.length; i++) {
    a[i] = a[i]! + b[i]!;
  }

  return a;
}

/**
 * In-place subtraction `A := A - B`. Returns A.
 */
export function subInPlace(a: Float64Array, b: Float64Array): Float64Array {
  for (let i = 0; i < a.length; i++) {
    a[i] = a[i]! - b[i]!;
  }

  return a;
}

/**
 * Cholesky decomposition of a symmetric positive-definite matrix A (n × n, row-major).
 * Returns the lower-triangular factor L such that A = L * Lᵀ, also as row-major Float64Array.
 * Upper-triangular cells of the result are zero.
 *
 * Throws if A is not positive definite (a non-positive diagonal arises during factorization).
 */
export function cholesky(a: Float64Array, n: number): Float64Array {
  const L = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    const Lj = j * n;
    let diag = a[Lj + j]!;
    for (let k = 0; k < j; k++) {
      const v = L[Lj + k]!;
      diag -= v * v;
    }
    if (diag <= 0) {
      throw new Error('cholesky: matrix is not positive-definite');
    }
    const ljj = Math.sqrt(diag);
    L[Lj + j] = ljj;
    for (let i = j + 1; i < n; i++) {
      const Li = i * n;
      let s = a[Li + j]!;
      for (let k = 0; k < j; k++) {
        s -= L[Li + k]! * L[Lj + k]!;
      }
      L[Li + j] = s / ljj;
    }
  }

  return L;
}

/**
 * Solve `A x = b` for x, given the Cholesky factor L of A (A = L Lᵀ).
 * L is (n × n) lower-triangular row-major; b is length n; returns x of length n.
 *
 * This is the numerically-stable inversion path used by the Kalman filter
 * innovation-covariance solve (ARCHITECTURE.md §5.3). Do not compute A⁻¹ explicitly.
 */
export function choleskySolve(L: Float64Array, b: Float64Array, n: number): Float64Array {
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const row = i * n;
    let s = b[i]!;
    for (let k = 0; k < i; k++) {
      s -= L[row + k]! * y[k]!;
    }
    y[i] = s / L[row + i]!;
  }

  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i]!;
    for (let k = i + 1; k < n; k++) {
      s -= L[k * n + i]! * x[k]!;
    }
    x[i] = s / L[i * n + i]!;
  }

  return x;
}
