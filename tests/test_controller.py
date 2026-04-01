import numpy as np

from src.nn.controller import NNController


def test_controller_predict_shapes():
    model = NNController()
    state = np.zeros(7, dtype=np.float32)
    ds, need_restart = model.predict(state)
    assert ds > 0.0
    assert isinstance(need_restart, bool)
