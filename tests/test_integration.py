import numpy as np

from src.core.contour_tracker import ContourTracker
from src.core.manifold_ode import ManifoldODE
from src.utils.svd import smallest_singular_triplet


def test_tracker_runs_without_controller():
    np.random.seed(1)
    A = np.random.randn(8, 8) + 1j * np.random.randn(8, 8)
    tracker = ContourTracker(A=A, epsilon=0.1, ode_system=ManifoldODE(A, 0.1))
    result = tracker.track(z0=0.3 + 0.1j, max_steps=5)
    assert len(result["trajectory"]) >= 2
    assert len(result["step_sizes"]) >= 1


def test_closure_detection():
    np.random.seed(2)
    A = np.random.randn(6, 6) + 1j * np.random.randn(6, 6)
    tracker = ContourTracker(A=A, epsilon=0.1, ode_system=ManifoldODE(A, 0.1), closure_tol=1e-2)
    assert tracker.check_closure(0.1 + 0.1j, 0.1005 + 0.1005j, current_step=12)
    assert not tracker.check_closure(0.1 + 0.1j, 0.1005 + 0.1005j, current_step=3)


def test_restart_improves_residual():
    np.random.seed(3)
    A = np.random.randn(8, 8) + 1j * np.random.randn(8, 8)
    z = 0.25 + 0.15j
    epsilon, u, v = smallest_singular_triplet(A, z)
    tracker = ContourTracker(A=A, epsilon=epsilon, ode_system=ManifoldODE(A, epsilon))
    u_bad = u + 0.05 * (np.random.randn(*u.shape) + 1j * np.random.randn(*u.shape))
    v_bad = v + 0.05 * (np.random.randn(*v.shape) + 1j * np.random.randn(*v.shape))
    u_bad /= np.linalg.norm(u_bad)
    v_bad /= np.linalg.norm(v_bad)
    M = z * np.eye(A.shape[0], dtype=np.complex128) - A
    residual_before = np.linalg.norm(M @ v_bad - epsilon * u_bad)
    _, u_restart, v_restart = tracker.exact_svd_restart(z)
    residual_after = np.linalg.norm(M @ v_restart - epsilon * u_restart)
    assert residual_after <= residual_before
