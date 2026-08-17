"""
pytest configuration — fixtures and settings for the full test suite.
"""

import pytest


def pytest_configure(config):
    """Suppress deprecation warnings that don't affect test correctness."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore::pytest.PytestRemovedIn10Warning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore::UserWarning:huggingface_hub",
    )
