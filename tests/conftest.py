import os

import pytest
from accelerate.state import AcceleratorState, GradientState


@pytest.fixture(autouse=True)
def isolate_accelerate_singletons():
    """Prevent one Trainer's device/precision state from leaking to another."""
    accelerate_environment = {
        key: value for key, value in os.environ.items() if key.startswith("ACCELERATE_")
    }
    yield
    for key in tuple(os.environ):
        if key.startswith("ACCELERATE_") and key not in accelerate_environment:
            del os.environ[key]
    os.environ.update(accelerate_environment)
    GradientState._reset_state()
    AcceleratorState._reset_state(reset_partial_state=True)
