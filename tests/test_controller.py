import numpy as np

from src.nn.controller import NNController


def test_controller_predict_shapes():
    model = NNController()
    state = np.zeros(8, dtype=np.float32)
    ds = model.predict(state)
    assert isinstance(ds, float)
    assert ds > 0.0
