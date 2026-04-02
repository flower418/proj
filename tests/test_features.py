import numpy as np

from src.nn.features import extract_features


def test_extract_features_shape():
    np.random.seed(0)
    A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    u = np.ones(4, dtype=np.complex128) / 2.0
    v = np.ones(4, dtype=np.complex128) / 2.0
    features = extract_features(0.1 + 0.2j, u, v, A, epsilon=0.1)
    assert features.shape == (10,)
    assert features.dtype == np.float32
