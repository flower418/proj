from __future__ import annotations

import numpy as np


def smallest_singular_triplet(A: np.ndarray, z: complex):
    A = np.asarray(A, dtype=np.complex128)
    M = z * np.eye(A.shape[0], dtype=np.complex128) - A
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    idx = int(np.argmin(S))
    return float(S[idx]), U[:, idx], Vh[idx, :].conj().T
