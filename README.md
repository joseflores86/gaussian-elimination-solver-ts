# gaussian-elimination-solver-ts

Solves systems of linear equations and computes matrix ranks via partial pivoting Gaussian elimination.

```ts
import { solve, rank } from "gaussian-elimination-solver-ts";

// Ax = b for a 2x2 system
const x = solve([[2, 1], [1, -1]], [3, 0]);
console.log(x); // [1, 1]  -> 2*1 + 1*1 = 3, 1*1 - 1*1 = 0

const r = rank([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
console.log(r); // 2
```

## Why

In the browser or server, pulling in a dense linear-algebra package to solve a handful of equations is wasteful. This is a single-file dependency-free solver that handles square, wide, tall, singular, and inconsistent systems. Back-substitution returns the unique solution when one exists; otherwise it returns `null` so the caller can branch cleanly.

The trade-off is that all input matrices must be dense JavaScript arrays of arrays. Sparse or symbolic matrices are not supported — use a different tool if those matter to you.

## Edge cases

- Inconsistent systems (no solution) return `null`.
- Systems with free variables (infinitely many solutions) also return `null`, because there is no unique vector to return. `rank` still works for these.
- Partial pivoting is used to mitigate floating-point error; a pivot with magnitude below `1e-12` is treated as zero.
