import pytest

from bot.config import load_config
from bot.data.split import dev_range, holdout_range, load_holdout


def test_dev_and_holdout_do_not_overlap():
    cfg = load_config()
    d0, d1 = dev_range(cfg)
    h0, h1 = holdout_range(cfg)
    assert d0 < d1 == h0 < h1


def test_holdout_requires_explicit_confirmation():
    with pytest.raises(PermissionError):
        load_holdout(load_config(), confirm="yes", reason="test")
