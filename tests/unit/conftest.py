"""Unit test configuration and fixtures.

This conftest provides test isolation safeguards to detect when mock
patches leak between tests. Use @patch decorators (not sys.modules
replacement) to ensure proper cleanup.
"""

import pytest


@pytest.fixture(autouse=True)
def detect_torch_pollution():
    """Detect if torch module was replaced with a Mock.

    This is a safety check to catch test pollution bugs early.
    If torch is replaced in sys.modules during import (not via @patch),
    it won't be cleaned up and will corrupt all subsequent tests.
    """
    yield  # Run the test

    # After test: verify torch is not a Mock object
    import torch
    if hasattr(torch, '_mock_name'):
        pytest.fail(
            "torch module was replaced with Mock - test pollution detected. "
            "Use @patch decorators instead of sys.modules replacement."
        )
