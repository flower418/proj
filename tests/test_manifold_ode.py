import numpy as np

from src.core.manifold_ode import ManifoldODE
from src.solvers.rk4 import rk4_triplet_step
from src.utils.svd import smallest_singular_triplet


class TestManifoldODE:
    def setup_method(self):
        np.random.seed(42)
        self.n = 12
        self.A = np.random.randn(self.n, self.n) + 1j * np.random.randn(self.n, self.n)
        self.z = 0.5 + 0.1j
        self.epsilon, self.u, self.v = smallest_singular_triplet(self.A, self.z)
        self.ode = ManifoldODE(self.A, self.epsilon)

    def test_dz_ds_unit_norm(self):
        dz_ds = self.ode.compute_dz_ds(self.z, self.u, self.v)
        assert np.isclose(np.abs(dz_ds), 1.0, atol=1e-10)

    def test_dv_orthogonal_to_v(self):
        dz_ds = self.ode.compute_dz_ds(self.z, self.u, self.v)
        dv_ds = self.ode.compute_dv_ds(self.z, self.u, self.v, dz_ds)
        assert np.isclose(np.real(np.vdot(self.v, dv_ds)), 0.0, atol=1e-8)

    def test_du_orthogonal_to_u(self):
        dz_ds = self.ode.compute_dz_ds(self.z, self.u, self.v)
        du_ds = self.ode.compute_du_ds(self.z, self.u, self.v, dz_ds)
        assert np.isclose(np.real(np.vdot(self.u, du_ds)), 0.0, atol=1e-8)

    def test_cache_correctness(self):
        M1 = self.ode._get_M(self.z)
        M2 = self.ode._get_M(self.z)
        assert np.array_equal(M1, M2)
        assert M1 is not M2
        M3 = self.ode._get_M(self.z + 1e-8)
        assert not np.array_equal(M1, M3)

    def test_sigma_conservation_short_step(self):
        z_next, u_next, v_next = rk4_triplet_step(
            self.ode.get_full_derivatives,
            self.z,
            self.u,
            self.v,
            1e-3,
        )
        u_next = u_next / np.linalg.norm(u_next)
        v_next = v_next / np.linalg.norm(v_next)
        sigma_next, _, _ = smallest_singular_triplet(self.A, z_next)
        assert abs(sigma_next - self.epsilon) < 1e-3


def test_large_matrix_derivatives():
    np.random.seed(7)
    n = 80
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    z = 0.2 + 0.15j
    epsilon, u, v = smallest_singular_triplet(A, z)
    ode = ManifoldODE(A, epsilon)
    dz_ds, du_ds, dv_ds = ode.get_full_derivatives(z, u, v)
    assert np.isfinite(np.abs(dz_ds))
    assert np.all(np.isfinite(np.real(du_ds)))
    assert np.all(np.isfinite(np.real(dv_ds)))
