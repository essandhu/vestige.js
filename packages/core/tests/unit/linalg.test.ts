import { describe, expect, it } from 'vitest';
import {
  addInPlace,
  cholesky,
  choleskySolve,
  matMul,
  matVec,
  outerProduct,
  subInPlace,
  transpose,
} from '../../src/geometry/linalg.js';

function f(...xs: number[]): Float64Array {
  return new Float64Array(xs);
}

function expectCloseArray(actual: Float64Array, expected: ArrayLike<number>, eps = 1e-10) {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i] as number, -Math.log10(eps));
  }
}

// Suites are skipped until linalg.ts is implemented on `feature/linalg`. Un-skip them as
// the first commit of that branch so the impl PR shows the red → green arc.

describe.skip('matMul', () => {
  it('2x2 * 2x2 sanity', () => {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]  =>  A*B = [[19,22],[43,50]]
    const a = f(1, 2, 3, 4);
    const b = f(5, 6, 7, 8);
    expectCloseArray(matMul(a, b, 2, 2, 2), [19, 22, 43, 50]);
  });

  it('identity is a left and right identity', () => {
    const I2 = f(1, 0, 0, 1);
    const a = f(7, 8, 9, 10);
    expectCloseArray(matMul(I2, a, 2, 2, 2), [7, 8, 9, 10]);
    expectCloseArray(matMul(a, I2, 2, 2, 2), [7, 8, 9, 10]);
  });

  it('non-square: 2x3 * 3x2', () => {
    // A = [[1,2,3],[4,5,6]]  (2x3)
    // B = [[7,8],[9,10],[11,12]] (3x2)
    // A*B = [[58,64],[139,154]]
    const a = f(1, 2, 3, 4, 5, 6);
    const b = f(7, 8, 9, 10, 11, 12);
    expectCloseArray(matMul(a, b, 2, 3, 2), [58, 64, 139, 154]);
  });

  it('writes into the provided out buffer when given', () => {
    const out = new Float64Array(4);
    const result = matMul(f(1, 0, 0, 1), f(2, 3, 4, 5), 2, 2, 2, out);
    expect(result).toBe(out);
    expectCloseArray(out, [2, 3, 4, 5]);
  });
});

describe.skip('transpose', () => {
  it('swaps dimensions correctly', () => {
    // A = [[1,2,3],[4,5,6]]  =>  Aᵀ = [[1,4],[2,5],[3,6]]
    expectCloseArray(transpose(f(1, 2, 3, 4, 5, 6), 2, 3), [1, 4, 2, 5, 3, 6]);
  });

  it('double transpose is identity', () => {
    const a = f(1, 2, 3, 4, 5, 6);
    const t = transpose(a, 2, 3);
    expectCloseArray(transpose(t, 3, 2), Array.from(a));
  });
});

describe.skip('matVec', () => {
  it('y = A * x', () => {
    // A = [[1,2,3],[4,5,6]]  (2x3),  x = [1,1,1]  =>  y = [6,15]
    expectCloseArray(matVec(f(1, 2, 3, 4, 5, 6), f(1, 1, 1), 2, 3), [6, 15]);
  });
});

describe.skip('outerProduct', () => {
  it('produces an m x n matrix in row-major order', () => {
    // x = [1,2], y = [3,4,5]  =>  M = [[3,4,5],[6,8,10]]
    expectCloseArray(outerProduct(f(1, 2), f(3, 4, 5)), [3, 4, 5, 6, 8, 10]);
  });
});

describe.skip('addInPlace / subInPlace', () => {
  it('mutates A and returns A', () => {
    const a = f(1, 2, 3, 4);
    const b = f(10, 20, 30, 40);
    const r = addInPlace(a, b);
    expect(r).toBe(a);
    expectCloseArray(a, [11, 22, 33, 44]);
  });

  it('subInPlace mirrors addInPlace', () => {
    const a = f(11, 22, 33, 44);
    const b = f(10, 20, 30, 40);
    subInPlace(a, b);
    expectCloseArray(a, [1, 2, 3, 4]);
  });
});

describe.skip('cholesky', () => {
  it('returns I for the identity matrix', () => {
    const I3 = f(1, 0, 0, 0, 1, 0, 0, 0, 1);
    expectCloseArray(cholesky(I3, 3), [1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });

  it('factorizes a 2x2 SPD matrix correctly', () => {
    // A = [[4,2],[2,3]]  =>  L = [[2,0],[1, sqrt(2)]]
    const A = f(4, 2, 2, 3);
    const L = cholesky(A, 2);
    expect(L[0]).toBeCloseTo(2, 10);
    expect(L[1]).toBeCloseTo(0, 10); // upper triangle is zero
    expect(L[2]).toBeCloseTo(1, 10);
    expect(L[3]).toBeCloseTo(Math.sqrt(2), 10);
  });

  it('L * Lᵀ reconstructs the input', () => {
    // A 3x3 SPD: [[25,15,-5],[15,18,0],[-5,0,11]]
    // L = [[5,0,0],[3,3,0],[-1,1,3]]
    const A = f(25, 15, -5, 15, 18, 0, -5, 0, 11);
    const L = cholesky(A, 3);
    const Lt = transpose(L, 3, 3);
    const reconstructed = matMul(L, Lt, 3, 3, 3);
    expectCloseArray(reconstructed, Array.from(A));
  });

  it('throws on a non-positive-definite matrix with an informative message', () => {
    // A negative-definite (or indefinite) matrix should throw.
    // The error message must mention positive-definiteness so this test
    // doesn't pass trivially against the `not implemented` stub.
    const A = f(1, 2, 2, 1); // eigenvalues 3, -1 — indefinite
    expect(() => cholesky(A, 2)).toThrow(/positive[\s-]?definite/i);
  });
});

describe.skip('choleskySolve', () => {
  it('solves A x = b given the Cholesky factor of A', () => {
    // A = [[4,2],[2,3]], b = [6, 5]  =>  x = [1, 1]
    const A = f(4, 2, 2, 3);
    const b = f(6, 5);
    const L = cholesky(A, 2);
    const x = choleskySolve(L, b, 2);
    expectCloseArray(x, [1, 1], 1e-9);
  });

  it('returns b itself (numerically) when A = I', () => {
    const I = f(1, 0, 0, 1);
    const L = cholesky(I, 2);
    const b = f(3, 4);
    expectCloseArray(choleskySolve(L, b, 2), [3, 4]);
  });
});
