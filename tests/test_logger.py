import pytest
from coffea.logger import setup_logger


def test_invalid_level():
    null_level = "NOT_ALLOWED"
    with pytest.raises(ValueError):
        setup_logger(level=null_level)
