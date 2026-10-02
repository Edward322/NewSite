import copy

import numpy as np
import pytest

from bot.backtest.portfolio import PortfolioBacktester, PortfolioConfig
from bot.data.panel import BASE_MS, DAY_MS, HOUR_MS, Panel, Series, SymbolData, aggregate_panel
from bot.strategy.base import LONG, SHORT, Enter, Exit, MoveStop, PositionView
from bot.strategy.basket import COMBOS, FAMILIES, MAX_STOP_FRAC, grid_configs, make
from tests.helpers import COSTS, RISK, T0
from tests.instruments import hype

N_DAYS = 160


def synth_panel(seed=0, n_trade=6, days=N_DAYS) -> Panel:
    rng = np.random.default_rng(seed)
    n = days * 96
    names = [f"C{i}USDT" for i in range(n_trade)] + ["BTCUSDT"]
    m = len(names)
    # общий рыночный фактор + собственный шум, с трендовыми участками
    mkt = np.cumsum(rng.normal(0, 0.002, n) + 0.0004 * np.sin(np.arange(n) / 3000))
    c = 10 * np.exp(mkt[None, :] + np.cumsum(rng.normal(0, 0.002, (m, n)), axis=1))
    o = np.concatenate([c[:, :1], c[:, :-1]], axis=1)
    sp = np.abs(rng.normal(0, 0.001, (m, n)))
    h, lo = np.maximum(o, c) * (1 + sp), np.minimum(o, c) * (1 - sp)
    # монета 5 появляется позже (листинг)
    for a in (o, h, lo, c):
        a[5, : 40 * 96] = np.nan
    ts = T0 + np.arange(n, dtype="int64") * BASE_MS
    hts = T0 + np.arange(days * 24, dtype="int64") * HOUR_MS
    sym = {}
    for i, s in enumerate(names):
        f_ts = T0 + np.arange(days * 3, dtype="int64") * 8 * HOUR_MS
        aux = {"premium_1h": Series(hts + HOUR_MS, rng.normal(0, 3e-4, len(hts))),
               "oi_1h": Series(hts + HOUR_MS, 1e6 * np.exp(np.cumsum(rng.normal(0, 0.01, len(hts))))),
               "lsr_1h": Series(hts + HOUR_MS, np.clip(0.6 + np.cumsum(rng.normal(0, 0.005, len(hts))), 0.2, 0.9))}
        sym[s] = SymbolData(s, hype(), f_ts, rng.normal(1e-4, 2e-4, len(f_ts)), aux)
    vol = rng.uniform(100, 1000, (m, n))
    fg = Series(T0 + np.arange(days, dtype="int64") * DAY_MS + DAY_MS, rng.uniform(5, 95, days))
    return Panel(symbols=names[:-1], signal_only=["BTCUSDT"], ts=ts, open=o, high=h, low=lo, close=c,
                 volume=vol, turnover=vol * c, sym=sym, fear_greed=fg, end_ms=int(ts[-1]) + BASE_MS)


def perturb_after(panel: Panel, t_cut: int, seed=1) -> Panel:
    """Подменяет всё, что становится известно после t_cut: цены, ряды, индекс."""
    rng = np.random.default_rng(seed)
    p = copy.deepcopy(panel)
    j = int(np.searchsorted(p.ts, t_cut, "left"))   # свечи с открытием ≥ t_cut закрываются позже
    f = np.exp(rng.normal(0, 0.3, (p.close.shape[0], p.close.shape[1] - j)))
    for a in (p.open, p.high, p.low, p.close):
        a[:, j:] *= f
    for sd in p.sym.values():
        later = sd.funding_ts > t_cut
        sd.funding_rate[later] = rng.normal(0, 1e-3, later.sum())
        for ser in sd.aux.values():
            later = ser.known_ts > t_cut
            ser.value[later] = ser.value[later] * rng.uniform(0.5, 1.5, later.sum())
    later = p.fear_greed.known_ts > t_cut
    p.fear_greed.value[later] = rng.uniform(0, 100, later.sum())
    return p


def sample_configs(family, k=4, seed=0):
    cfgs = grid_configs(family)
    rng = np.random.default_rng(seed)
    idx = sorted({0, len(cfgs) - 1, *rng.choice(len(cfgs), size=min(k, len(cfgs)), replace=False).tolist()})
    return [cfgs[i] for i in idx]


PANEL = synth_panel()


CUT_DAYS = (95, 104, 113, 122, 131, 140, 149)


def causal_diffs(family_or_cls, tf, params, last=20):
    """Решения на свечах до момента среза не должны зависеть от данных после него."""
    fake = {0: PositionView(LONG, 10.0, 1.0, 9.0, None, int(T0), 3),
            1: PositionView(SHORT, 10.0, 1.0, 11.0, None, int(T0), 3)}
    mk = (lambda: make(family_or_cls, tf, params)) if isinstance(family_or_cls, str) \
        else (lambda: family_or_cls(timeframe=tf, **params))
    b0 = aggregate_panel(PANEL, tf)
    s0 = mk()
    s0.prepare(b0)
    diffs = []
    for i, day in enumerate(CUT_DAYS):
        t_cut = T0 + day * DAY_MS
        b1 = aggregate_panel(perturb_after(PANEL, t_cut, seed=i), tf)
        s1 = mk()
        s1.prepare(b1)
        ks = np.flatnonzero(b0.close_ts <= t_cut)[-last:]
        for k in ks:
            for pos in ({}, fake):
                if s0.on_bar(int(k), pos) != s1.on_bar(int(k), pos):
                    diffs.append((day, int(k)))
    return diffs


@pytest.mark.parametrize("family,tf", COMBOS)
def test_basket_strategies_are_causal(family, tf):
    for params in sample_configs(family):
        assert causal_diffs(family, tf, params) == [], (family, tf, params)


def test_causality_check_catches_one_bar_leaks():
    from bot.strategy.basket import Crowd, XsMom

    class PeekPrice(XsMom):
        def prepare(self, bars):
            super().prepare(bars)
            self.score = np.concatenate([self.score[:, 1:], self.score[:, -1:]], axis=1)

    class PeekAux(Crowd):
        def prepare(self, bars):
            super().prepare(bars)
            for r, s in enumerate(bars.symbols[:self.n]):
                ser = bars.panel.sym[s].aux["lsr_1h"]
                idx = np.searchsorted(ser.known_ts - HOUR_MS, bars.close_ts, "right") - 1
                self.score[r] = -ser.value[np.clip(idx, 0, None)]

    assert causal_diffs(PeekPrice, "1d", {"L": 3, "dir": "mom", "k_stop": 3.0, "mode": "ls"})
    assert causal_diffs(PeekAux, "4h", {"W": 3, "k_stop": 3.0, "mode": "ls"})


@pytest.mark.parametrize("family,tf", COMBOS)
def test_basket_strategies_trade_and_respect_stop_cap(family, tf):
    cfg = PortfolioConfig(risk=RISK, costs=COSTS, initial_equity=1000.0)
    total = 0
    for params in sample_configs(family, k=2):
        s = make(family, tf, params)
        res = PortfolioBacktester(PANEL, s, cfg).run()
        total += len(res.trades)
        if len(res.trades):
            dist = (res.trades["entry_price"] - res.trades["stop_initial"]).abs() / res.trades["entry_price"]
            assert (dist <= MAX_STOP_FRAC * 1.01).all()
            assert (res.trades["symbol"] != "BTCUSDT").all()
            # монета 5 до своего листинга и прогрева не торгуется
            late = res.trades[res.trades["symbol"] == "C5USDT"]
            assert (late["decision_ts"] >= T0 + 40 * DAY_MS).all()
    assert total > 0, (family, tf)


def test_cross_section_ranks_and_exits():
    b = aggregate_panel(PANEL, "1d")
    s = make("xsmom", "1d", {"L": 3, "dir": "mom", "k_stop": 3.0, "mode": "ls"})
    s.prepare(b)
    k = len(b) - 2
    ds = s.on_bar(k, {})
    enters = [(r, d) for r, d in ds if isinstance(d, Enter)]
    avail = np.flatnonzero(s.ok[:, k])
    order = avail[np.argsort(-s.score[avail, k], kind="stable")]
    longs = {r for r, d in enters if d.side == LONG}
    shorts = {r for r, d in enters if d.side == SHORT}
    assert longs <= set(order[:2].tolist()) and shorts <= set(order[-2:].tolist())
    # позиция в лонге на худшей монете закрывается
    worst = int(order[-1])
    ds = s.on_bar(k, {worst: PositionView(LONG, 10.0, 1.0, 9.0, None, int(T0), 3)})
    assert (worst, Exit("rank")) in ds


def test_breakout_trails_stop():
    b = aggregate_panel(PANEL, "4h")
    s = make("breakout", "4h", {"n": 20, "k_stop": 2.0, "k_trail": 3.0, "sides": "both", "regime": "none"})
    s.prepare(b)
    k = len(b) - 5
    r = 0
    pos = PositionView(LONG, float(s.c[r, k - 10]), 1.0, 0.01, None, int(T0), 10)
    ds = dict(s.on_bar(k, {r: pos}))
    assert isinstance(ds.get(r), (MoveStop, Exit))


def test_regime_btc_blocks_longs_below_ema():
    b = aggregate_panel(PANEL, "1d")
    s = make("tsmom", "1d", {"L": 3, "z_in": 1.0, "k_stop": 3.0, "sides": "both", "regime": "btc"})
    s.prepare(b)
    assert not (s.long_ok & s.short_ok).any()
    for k in range(len(b)):
        for _, d in s.on_bar(k, {}):
            if isinstance(d, Enter):
                assert (s.long_ok[k] if d.side == LONG else s.short_ok[k])


def test_family_grid_sizes_match_protocol():
    assert {f: len(grid_configs(f)) for f in FAMILIES} == {
        "tsmom": 72, "breakout": 48, "xsmom": 40, "carry": 16, "oi_breakout": 24, "crowd": 12}
    assert len(COMBOS) == 11
