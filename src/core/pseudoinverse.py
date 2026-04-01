from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, gmres, lgmres, minres


ArrayLike = np.ndarray


@dataclass
class SolverInfo:
    iterations: int = 0
    info: int = 0


class PseudoinverseSolver:
    """Solve (H - sigma_sq I)^+ b with iterative or dense methods."""

    def __init__(self, method: str = "minres", tol: float = 1e-8, max_iter: int = 1000):
        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        self._last_info = SolverInfo()

    def solve(
        self,
        H: Callable[[ArrayLike], ArrayLike] | ArrayLike,
        sigma_sq: float,
        b: ArrayLike,
        null_vector: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        b = np.asarray(b, dtype=np.complex128)
        n = b.shape[0]

        if null_vector is not None:
            null_vector = np.asarray(null_vector, dtype=np.complex128)
            null_vector = null_vector / max(np.linalg.norm(null_vector), 1e-15)
            b = b - np.vdot(null_vector, b) * null_vector

        if self.method == "svd" or n < 64:
            matrix = self._materialize_matrix(H, n)
            shifted = matrix - sigma_sq * np.eye(n, dtype=np.complex128)
            x = np.linalg.pinv(shifted) @ b
            self._last_info = SolverInfo(iterations=1, info=0)
            return self._project_null(x, null_vector)

        op = self._build_operator(H, sigma_sq, n)
        iterations = {"count": 0}

        def callback(_: ArrayLike) -> None:
            iterations["count"] += 1

        # Use Krylov methods compatible with the lifted real operator.
        if np.iscomplexobj(b):
            real_op = self._build_real_block_operator(op, n)
            real_b = self._complex_to_real(b)
            solver = self._select_solver(is_complex=True)
            real_x, info = solver(real_op, real_b, rtol=self.tol, maxiter=self.max_iter, callback=callback)
            x = self._real_to_complex(np.asarray(real_x, dtype=np.float64))
        else:
            solver = self._select_solver(is_complex=False)
            x, info = solver(op, b, rtol=self.tol, maxiter=self.max_iter, callback=callback)
            x = np.asarray(x, dtype=np.complex128)
        self._last_info = SolverInfo(iterations=iterations["count"], info=info)
        return self._project_null(x, null_vector)

    def get_iteration_count(self) -> int:
        return self._last_info.iterations

    def _build_operator(self, H: Callable[[ArrayLike], ArrayLike] | ArrayLike, sigma_sq: float, n: int) -> LinearOperator:
        if callable(H):
            matvec = lambda x: np.asarray(H(x), dtype=np.complex128) - sigma_sq * x
        else:
            matrix = np.asarray(H, dtype=np.complex128)
            matvec = lambda x: matrix @ x - sigma_sq * x
        return LinearOperator((n, n), matvec=matvec, dtype=np.complex128)

    def _materialize_matrix(self, H: Callable[[ArrayLike], ArrayLike] | ArrayLike, n: int) -> ArrayLike:
        if not callable(H):
            return np.asarray(H, dtype=np.complex128)
        eye = np.eye(n, dtype=np.complex128)
        return np.column_stack([H(eye[:, i]) for i in range(n)])

    def _build_real_block_operator(self, operator: LinearOperator, n: int) -> LinearOperator:
        def matvec(x: np.ndarray) -> np.ndarray:
            complex_x = self._real_to_complex(x)
            complex_y = np.asarray(operator @ complex_x, dtype=np.complex128)
            return self._complex_to_real(complex_y)

        return LinearOperator((2 * n, 2 * n), matvec=matvec, dtype=np.float64)

    def _select_solver(self, is_complex: bool):
        if is_complex:
            if self.method == "lgmres":
                return lgmres
            return gmres
        if self.method == "cg":
            return cg
        if self.method == "gmres":
            return gmres
        if self.method == "lgmres":
            return lgmres
        return minres

    @staticmethod
    def _complex_to_real(x: np.ndarray) -> np.ndarray:
        return np.concatenate([np.real(x), np.imag(x)]).astype(np.float64, copy=False)

    @staticmethod
    def _real_to_complex(x: np.ndarray) -> np.ndarray:
        half = x.shape[0] // 2
        return x[:half] + 1j * x[half:]

    def _project_null(self, x: ArrayLike, null_vector: Optional[ArrayLike]) -> ArrayLike:
        if null_vector is None:
            return x
        return x - np.vdot(null_vector, x) * null_vector
