/**
 * Partial-pivoting Gaussian elimination for dense linear systems.
 *
 * Implemented from scratch with zero dependencies. Partial pivoting
 * (selecting the row with the largest absolute value in the pivot column
 * at each step) is chosen over complete pivoting because it is substantially
 * cheaper while still providing numerical stability for non-pathological inputs.
 */

const EPS = 1e-12;

/**
 * Performs partial-pivoting Gaussian elimination on an augmented
 * matrix [A|b] in place. Returns the row-reduced matrix and the rank.
 * We do NOT fully reduce to RREF — we stop at row echelon form so that
 * the same routine can feed both `solve` (back-substitution) and `rank`.
 */
function eliminate(matrix, m, n) {
  let rank = 0;
  let row = 0;
  for (let col = 0; col < n && row < m; col++) {
    // Partial pivoting: find the row (at or below `row`) with the largest
    // absolute value in this column. Swapping it into the pivot position
    // avoids dividing by a near-zero number, which would amplify error.
    let pivotRow = row;
    let maxAbs = Math.abs(matrix[row][col]);
    for (let r = row + 1; r < m; r++) {
      const v = Math.abs(matrix[r][col]);
      if (v > maxAbs) {
        maxAbs = v;
        pivotRow = r;
      }
    }

    if (maxAbs < EPS) {
      // Column is effectively all zeros below the current rank; skip it.
      continue;
    }

    if (pivotRow !== row) {
      const tmp = matrix[row];
      matrix[row] = matrix[pivotRow];
      matrix[pivotRow] = tmp;
    }

    const pivotVal = matrix[row][col];
    for (let r = row + 1; r < m; r++) {
      const factor = matrix[r][col] / pivotVal;
      if (factor === 0) continue;
      for (let c = col; c <= n; c++) {
        matrix[r][c] -= factor * matrix[row][c];
      }
    }

    rank++;
    row++;
  }
  return rank;
}

/**
 * Solves a system of linear equations Ax = b via Gaussian elimination with
 * partial pivoting, followed by back-substitution.
 *
 * @param A  m x n coefficient matrix (m rows, n cols).
 * @param b  Right-hand side vector of length m.
 * @returns   A length-n solution vector if there is a unique solution;
 *            `null` if the system is inconsistent or has free variables.
 */
export function solve(A, b) {
  const m = A.length;
  if (m === 0) return null;
  const n = A[0].length;

  if (b.length !== m) {
    throw new Error(`Dimension mismatch: A has ${m} rows but b has length ${b.length}`);
  }

  // Build the augmented matrix [A | b] as a fresh copy so the caller's
  // arrays are not mutated. We only ever read from A and b after this.
  const aug = new Array(m);
  for (let i = 0; i < m; i++) {
    const row = A[i];
    if (row.length !== n) {
      throw new Error(`Row ${i} has length ${row.length}, expected ${n}`);
    }
    aug[i] = row.slice(0); // copy coefficients
    aug[i].push(b[i]);     // append RHS
  }

  const computedRank = eliminate(aug, m, n);

  // A unique solution exists only when rank equals the number of unknowns.
  // If rank < n, either the system is inconsistent or it has free variables
  // — either way there is no unique vector to return.
  if (computedRank < n) {
    return null;
  }

  // Back-substitution. Since rank === n and the matrix is in row echelon
  // form, row i has its leading non-zero entry at column i (value 1 in the
  // original pivot before elimination, but we divide by the pivot instead
  // of normalizing rows to save a pass over the matrix).
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = aug[i][n];
    for (let j = i + 1; j < n; j++) {
      sum -= aug[i][j] * x[j];
    }
    const pivot = aug[i][i];
    x[i] = sum / pivot;
  }

  return x;
}

/**
 * Computes the rank of a matrix (the number of linearly independent rows
 * or columns) via Gaussian elimination with partial pivoting.
 *
 * @param A  m x n matrix.
 * @returns   The rank as a non-negative integer.
 */
export function rank(A) {
  const m = A.length;
  if (m === 0) return 0;
  const n = A[0].length;

  const aug = new Array(m);
  for (let i = 0; i < m; i++) {
    const row = A[i];
    if (row.length !== n) {
      throw new Error(`Row ${i} has length ${row.length}, expected ${n}`);
    }
    aug[i] = row.slice(0);
  }

  return eliminate(aug, m, n);
}
