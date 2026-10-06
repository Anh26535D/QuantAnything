import os
import sys

import pytest

# Make the project root importable when pytest is run from anywhere.
project_root = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from quantization import native  # noqa: E402


@pytest.fixture(params=["native", "numpy"])
def backend(request, monkeypatch):
    """Runs a test with the C++ kernels and with the NumPy fallbacks."""
    if request.param == "native":
        if not native.available():
            pytest.skip(f"native kernels unavailable: {native.load_error()}")
    else:
        monkeypatch.setattr(native, "_lib", None)
    return request.param
