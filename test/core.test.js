import { test } from "node:test";
import assert from "node:assert/strict";
import { solve, rank } from "../src/index.js";

const EPS = 1e-9;

function approxEqual(actual, expected) {
  assert.ok(Array.isArray(actual), "result should be an array");
  assert.equal(actual.length, expected.length, "length mismatch");
  for (let i = 0; i < expected.length; i++) {
    assert.ok(
      Math.abs(actual[i] - expected[i]) < EPS,
      `index ${i}: expected ${expected[i]}, got ${actual[i]}`
    );
  }
}

test("solve: 2x2 unique system", () => {
  const A = [[2, 1], [1, -1]];
  const b = [3, 0];
  approxEqual(solve(A, b), [1, 1]);
});

test("solve: 3x3 identity-ish system", () => {
  const A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const b = [5, 7, 9];
  approxEqual(solve(A, b), [5, 7, 9]);
});

test("solve: requires pivoting to avoid zero pivot", () => {
  // First pivot is zero without row swapping; partial pivoting must catch this.
  const A = [[0, 2], [1, 3]];
  const b = [4, 5];
  approxEqual(solve(A, b), [-1, 2]);
});

test("solve: singular system (free variables) returns null", () => {
  // Rows are linearly dependent: row2 = 2 * row1.
  const A = [[1, 2], [2, 4]];
  const b = [3, 6];
  assert.equal(solve(A, b), null);
});

test("solve: inconsistent system returns null", () => {
  // row2 = 2*row1 for A, but b does not satisfy the same relation.
  const A = [[1, 2], [2, 4]];
  const b = [3, 7];
  assert.equal(solve(A, b), null);
});

test("solve: does not mutate input matrices", () => {
  const A = [[2, 1], [1, -1]];
  const b = [3, 0];
  const Acopy = A.map((r) => r.slice());
  const bcopy = b.slice();
  solve(A, b);
  assert.deepEqual(A, Acopy, "A was mutated");
  assert.deepEqual(b, bcopy, "b was mutated");
});

test("solve: dimension mismatch throws", () => {
  assert.throws(() => solve([[1, 2]], [1, 2]), /Dimension mismatch/);
});

test("solve: ragged matrix throws", () => {
  assert.throws(() => solve([[1, 2], [3]], [1, 1]), /expected 2/);
});

test("solve: wide system (more unknowns than equations) returns null", () => {
  // rank < n, so no unique solution exists.
  const A = [[1, 2, 3], [4, 5, 6]];
  const b = [1, 2];
  assert.equal(solve(A, b), null);
});

test("solve: tall system (more equations, consistent) returns solution", () => {
  // The third equation is a linear combination of the first two.
  const A = [[1, 0], [0, 1], [1, 1]];
  const b = [2, 3, 5];
  approxEqual(solve(A, b), [2, 3]);
});

test("rank: full-rank square matrix", () => {
  assert.equal(rank([[1, 0], [0, 1]]), 2);
});

test("rank: rank-deficient matrix", () => {
  assert.equal(rank([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 2);
});

test("rank: zero matrix", () => {
  assert.equal(rank([[0, 0], [0, 0]]), 0);
});
