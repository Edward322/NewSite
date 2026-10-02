import numpy as np
import pandas as pd
import pytest

from bot.backtest.engine import Backtester, EngineConfig
from bot.backtest.portfolio import PortfolioBacktester, PortfolioConfig, PortfolioStrategy, SingleSymbol
from bot.data.panel import BASE_MS, Panel, SymbolData, aggregate_panel
from bot.strategy.base import LONG, SHORT, Enter, Exit
from bot.strategy.candidates import make
from tests.helpers import COSTS, RISK, T0
from tests.instruments import hype
from tests.strategies import SmaCross

HOUR = 3_600_000


def make_panel(n_syms=1, n=4000, seed=0, funding=True, vol=0.004) -> Panel:
    rng = np.random.default_rng(seed)
    shape = (n_syms, n)
    c = 40 * np.exp(np.cumsum(rng.normal(0, vol, shape), axis=1))
    o = np.concatenate([np.full((n_syms, 1), 40.0), c[:, :-1]], axis=1)
    sp = np.abs(rng.normal(0, vol / 2, shape))
    h, lo = np.maximum(o, c) * (1 + sp), np.minimum(o, c) * (1 - sp)
    ts = T0 + np.arange(n, dtype="int64") * BASE_MS
    names = [f"S{i}USDT" for i in range(n_syms)]
    sym = {}
    for i, s in enumerate(names):
        f_ts = np.arange(T0, ts[-1], 8 * HOUR, dtype="int64") if funding else np.array([], dtype="int64")
        f_r = rng.normal(0.0001, 0.0003, len(f_ts))
        sym[s] = SymbolData(s, hype(), f_ts, f_r)
    vol_ = rng.uniform(10, 1000, shape)
    return Panel(symbols=names, signal_only=[], ts=ts, open=o, high=h, low=lo, close=c, volume=vol_,
                 turnover=vol_ * c, sym=sym, fear_greed=None, end_ms=int(ts[-1]) + BASE_MS)


def run_old(panel, strategy, equity=10.0, **kw):
    md = panel.minute_data(panel.symbols[0])
    sd = panel.sym[panel.symbols[0]]
    cfg = EngineConfig(risk=RISK, costs=COSTS, initial_equity=equity, **kw)
    return Backtester(md, sd.funding_ts, sd.funding_rate, sd.inst, strategy, cfg).run()


def run_new(panel, strategy, equity=10.0, **kw):
    cfg = PortfolioConfig(risk=RISK, costs=COSTS, initial_equity=equity, **kw)
    return PortfolioBacktester(panel, strategy, cfg).run()


CASES = [
    ("donchian", "1h", {"n": 20, "k_stop": 1.5, "k_trail": 2.0, "er_min": 0.0}),
    ("meanrev", "1h", {"n": 20, "z_in": 2.0, "k_stop": 2.5, "er_max": 1.0}),
    ("momentum", "4h", {"L": 12, "z_in": 1.0, "k_stop": 2.0, "er_min": 0.0}),
    ("fade", "30m", {"n": 20, "k_stop": 1.0, "tp_k": 1.0, "hold": 8, "er_max": 1.0}),
]


@pytest.mark.parametrize("family,tf,params", CASES)
@pytest.mark.parametrize("equity", [10.0, 1000.0])
def test_single_symbol_matches_old_engine_trade_for_trade(family, tf, params, equity):
    panel = make_panel(seed=3)
    old = run_old(panel, make(family, tf, params), equity=equity)
    new = run_new(panel, SingleSymbol(make(family, tf, params)), equity=equity)
    assert len(old.trades) >= 5
    cols = list(old.trades.columns)
    pd.testing.assert_frame_equal(old.trades[cols], new.trades[cols], check_exact=False, rtol=1e-12)
    assert new.final_equity == pytest.approx(old.final_equity, rel=1e-12)
    np.testing.assert_allclose(new.equity["equity"].to_numpy(), old.equity["equity"].to_numpy(), rtol=1e-12)
    assert [e.kind for e in new.events] == [e.kind for e in old.events]


def test_single_symbol_match_with_trade_window_and_sma():
    panel = make_panel(seed=5, n=3000)
    start, end = T0 + 5 * 86_400_000, T0 + 25 * 86_400_000
    old = run_old(panel, SmaCross(timeframe="15m"), trade_start_ms=start, trade_end_ms=end)
    new = run_new(panel, SingleSymbol(SmaCross(timeframe="15m")), trade_start_ms=start, trade_end_ms=end)
    assert len(old.trades) >= 5
    pd.testing.assert_frame_equal(old.trades, new.trades.drop(columns="symbol"), check_exact=False, rtol=1e-12)


class Script(PortfolioStrategy):
    """Решения по номеру свечи: {k: [(строка, решение), ...]}."""
    name = "script"

    def __init__(self, script, timeframe="1h"):
        self.script, self.timeframe = script, timeframe

    def prepare(self, bars):
        self.bars = bars

    def on_bar(self, k, positions):
        return self.script.get(k, [])


def flat_panel(n_syms, n=400, price=40.0):
    p = make_panel(n_syms=n_syms, n=n, funding=False)
    for a in (p.open, p.high, p.low, p.close):
        a[:] = price
    return p


def test_max_positions_and_priority_order():
    panel = flat_panel(5)
    d = [(r, Enter(LONG, 39.0)) for r in (4, 2, 0, 1, 3)]
    res = run_new(panel, Script({1: d}), equity=1000.0, max_positions=3, max_open_risk=1.0)
    assert sorted(res.trades["symbol"]) == ["S0USDT", "S2USDT", "S4USDT"]
    assert sum(1 for s in res.skips if s[2] == "нет свободных мест") == 2


def test_max_open_risk_limits_concurrent_risk():
    panel = flat_panel(5)
    d = [(r, Enter(LONG, 39.0)) for r in range(5)]
    res = run_new(panel, Script({1: d}), equity=1000.0, max_positions=10, max_open_risk=0.12)
    # каждая позиция рискует до 5 % — при лимите 12 % помещаются только две
    assert len(res.trades) == 2
    # риск считается от цены решения; фактический вход хуже на проскальзывание
    assert (res.trades["planned_loss"] <= 0.05 * 1000 * 1.01).all()


def test_open_risk_uses_current_stop_distance():
    panel = flat_panel(3)
    # первая позиция с подтянутым к цене стопом почти не несёт риска → место для двух новых
    script = {1: [(0, Enter(LONG, 39.0))]}
    res0 = run_new(panel, Script(script), equity=1000.0, max_open_risk=0.12)
    assert len(res0.trades) == 1
    from bot.strategy.base import MoveStop
    script = {1: [(0, Enter(LONG, 39.0))], 3: [(0, MoveStop(39.99))],
              5: [(1, Enter(LONG, 39.0)), (2, Enter(LONG, 39.0))]}
    res = run_new(panel, Script(script), equity=1000.0, max_open_risk=0.12)
    assert sorted(res.trades["symbol"]) == ["S0USDT", "S1USDT", "S2USDT"]


def test_shared_cash_and_account_level_streak_pause():
    panel = flat_panel(5, n=600)
    # четыре монеты подряд проваливаются под стоп → пауза 24 ч на весь счёт
    for r, k in zip(range(4), (10, 20, 30, 40)):
        panel.low[r, k * 4 + 1] = 38.0
    script = {}
    for r, k in zip(range(4), (1, 11, 21, 31)):
        script[k] = [(r, Enter(LONG, 39.0))]
    script[45] = [(4, Enter(LONG, 39.0))]
    res = run_new(panel, Script(script), equity=1000.0)
    assert list(res.trades["exit_reason"][:4]) == ["stop"] * 4
    assert "streak_pause" in [e.kind for e in res.events]
    assert any(s[1] == "S4USDT" and "пауза" in s[2] for s in res.skips)


def test_funding_charged_per_symbol_on_its_own_schedule():
    panel = flat_panel(2, n=200)
    panel.sym["S0USDT"].funding_ts = np.array([T0 + 8 * HOUR], dtype="int64")
    panel.sym["S0USDT"].funding_rate = np.array([0.001])
    panel.sym["S1USDT"].funding_ts = np.array([T0 + 4 * HOUR, T0 + 6 * HOUR], dtype="int64")
    panel.sym["S1USDT"].funding_rate = np.array([0.001, -0.002])
    res = run_new(panel, Script({1: [(0, Enter(LONG, 39.0)), (1, Enter(SHORT, 41.0))],
                                 20: [(0, Exit()), (1, Exit())]}), equity=1000.0)
    t = res.trades.set_index("symbol")
    assert t.loc["S0USDT", "funding"] == pytest.approx(t.loc["S0USDT", "qty"] * 40 * 0.001)
    assert t.loc["S1USDT", "funding"] == pytest.approx(-t.loc["S1USDT", "qty"] * 40 * (0.001 - 0.002))


def test_margin_of_open_positions_reduces_available_cash():
    panel = flat_panel(3)
    # стоп 0,5 % → большой объём; маржа первых двух съедает почти весь кошелёк
    d = [(r, Enter(LONG, 39.8)) for r in range(3)]
    res = run_new(panel, Script({1: d}), equity=10.0, max_open_risk=1.0)
    margins = res.trades["margin"].sum()
    assert margins <= 10.0
    first = res.trades.iloc[0]
    assert first["qty"] * first["entry_price"] / first["leverage"] == pytest.approx(first["margin"])


def test_drawdown_kill_with_two_positions_closes_and_halts():
    panel = flat_panel(2, n=300)
    script = {1: [(0, Enter(LONG, 30.0)), (1, Enter(LONG, 30.0))]}
    from bot.config import RiskCfg
    risk = RiskCfg(starting_equity_usdt=1000, risk_per_trade=0.05, daily_loss_limit=0.08, max_drawdown=0.08,
                   loss_streak_pause_trades=4, loss_streak_pause_hours=24)
    # обе монеты одновременно падают до 31,5: стопы (30) не задеты, но вместе позиции
    # теряют больше 8 % счёта → аварийное закрытие по порогу просадки и остановка
    for r in range(2):
        panel.low[r, 20] = 31.5
        panel.open[r, 21:] = panel.high[r, 21:] = panel.low[r, 21:] = panel.close[r, 21:] = 31.5
    cfg = PortfolioConfig(risk=risk, costs=COSTS, initial_equity=1000.0, max_open_risk=1.0)
    res = PortfolioBacktester(panel, Script(script), cfg).run()
    assert res.halted
    assert list(res.trades["exit_reason"]) == ["max_drawdown", "max_drawdown"]
    assert (res.trades["exit_ts"] == panel.ts[20]).all()
    # монеты одинаковые → порог считается с учётом худшей точки другой позиции, цены равны
    assert res.trades["exit_price"].iloc[0] == pytest.approx(res.trades["exit_price"].iloc[1])
    assert res.trades["exit_price"].iloc[0] > 31.5
    # порог просадки 920; закрытие с запасом — другая позиция считается в худшей точке свечи
    assert 1000 * 0.92 * 0.99 <= res.final_equity <= 1000 * 0.93


def test_aggregate_panel_marks_incomplete_bars_nan():
    panel = make_panel(n_syms=2, n=200)
    panel.close[1, :50] = np.nan
    panel.open[1, :50] = panel.high[1, :50] = panel.low[1, :50] = np.nan
    b = aggregate_panel(panel, "1h")
    assert np.isnan(b.close[1, :12]).all() and not np.isnan(b.close[1, 13:]).any()
    assert not np.isnan(b.close[0]).any()
    k = 20
    sl = slice(b.m_start[k], b.m_end[k])
    assert b.high[0, k] == panel.high[0, sl].max() and b.close[0, k] == panel.close[0, sl][-1]


def test_panel_dev_loader_never_returns_holdout(tmp_path):
    from bot.config import load_config
    from bot.data import store
    from bot.data.panel import load_basket_dev, load_basket_holdout
    from bot.data.split import HOLDOUT_CONFIRM  # noqa: F401
    from tests.instruments import HYPE_INSTRUMENT, HYPE_TIERS
    cfg = load_config()
    hs = int(pd.Timestamp(cfg.research.holdout_start).value // 1_000_000)
    ts = np.arange(hs - 40 * HOUR, hs + 40 * HOUR, BASE_MS, dtype="int64")
    d = tmp_path / "SXUSDT"
    d.mkdir()
    k = pd.DataFrame({"ts": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "turnover": 1.0})
    k.to_parquet(d / "kline_last_15m.parquet")
    hts = np.arange(hs - 40 * HOUR, hs + 40 * HOUR, HOUR, dtype="int64")
    pd.DataFrame({"ts": hts, "oi": 1.0}).to_parquet(d / "oi_1h.parquet")
    pd.DataFrame({"ts": hts[::8], "rate": 0.0}).to_parquet(d / "funding.parquet")
    store.write_json(d / "instrument.json", {"instrument": HYPE_INSTRUMENT})
    store.write_json(d / "risk_limit.json", {"tiers": HYPE_TIERS})
    store.write_json(tmp_path / "universe.json", {"tradable": ["SXUSDT"], "signal_only": []})
    p = load_basket_dev(cfg, tmp_path)
    assert p.ts[-1] + BASE_MS <= hs
    oi = p.sym["SXUSDT"].aux["oi_1h"]
    assert oi.known_ts[-1] <= hs and p.sym["SXUSDT"].funding_ts[-1] < hs
    with pytest.raises(PermissionError):
        load_basket_holdout(cfg, "yes", "test", tmp_path)
