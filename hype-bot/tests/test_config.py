import pytest
from pydantic import ValidationError

from bot.config import EXAMPLE_CONFIG, Config, load_config


def test_example_config_matches_user_parameters():
    cfg = load_config(EXAMPLE_CONFIG)
    r = cfg.risk
    assert cfg.symbol == "HYPEUSDT" and cfg.category == "linear"
    assert r.starting_equity_usdt == 10
    assert r.max_leverage is None  # лимит плеча снят по решению пользователя
    assert r.risk_per_trade == 0.05
    assert r.daily_loss_limit == 0.25
    assert r.max_drawdown == 0.80
    assert r.loss_streak_pause_trades == 4 and r.loss_streak_pause_hours == 24
    assert r.margin_mode == "ISOLATED_MARGIN"
    assert cfg.research.holdout_months == 6
    assert cfg.live.enabled is False
    assert cfg.costs.maker_fee == 0.0002 and cfg.costs.taker_fee == 0.00055


def _risk(**over):
    base = dict(starting_equity_usdt=10, risk_per_trade=0.05,
                daily_loss_limit=0.25, max_drawdown=0.8, loss_streak_pause_trades=4,
                loss_streak_pause_hours=24)
    base.update(over)
    return {"risk": base, "costs": {"maker_fee": 0.0002, "taker_fee": 0.00055}}


@pytest.mark.parametrize("over", [
    {"risk_per_trade": 0.5},           # выше потолка 20%
    {"max_leverage": 0.5},
    {"daily_loss_limit": 0.01},        # меньше риска одной сделки
    {"max_drawdown": 0.1},             # меньше дневного лимита
    {"max_drawdown": 1.0},
    {"unknown_key": 1},                # опечатки в конфиге не проходят молча
])
def test_invalid_risk_rejected(over):
    with pytest.raises(ValidationError):
        Config.model_validate(_risk(**over))


def test_live_disabled_by_default():
    assert Config.model_validate(_risk()).live.enabled is False
