"""Портфельные правила риска (docs/RISK_PROTOCOL.md): лимиты, корреляция, волатильность,
ликвидность, ступени просадки, теневые варианты стратегии."""
import copy
import numpy as np
import pytest

from bot.backtest.portfolio import PortfolioBacktester, PortfolioConfig
from bot.config import RiskCfg
from bot.data.panel import DAY_MS, aggregate_panel
from bot.risk.guards import RiskGuard
from bot.risk.portfolio import Exposure, LiquidityRule, PortfolioRisk, PortfolioRules, liquidity_mask
from bot.risk.sizing import LONG, SHORT, SizingParams, Skip
from bot.strategy.base import Enter, Exit
from bot.strategy.basket import BreakoutShadow, make_breakout
from tests.helpers import COSTS, RISK, T0
from tests.instruments import hype
from tests.test_basket_strategies import PANEL, causal_diffs, perturb_after
from tests.test_portfolio import Script, flat_panel

MAIN = {"n": 38, "k_stop": 2.5, "k_trail": 4.0, "sides": "both", "regime": "none"}
SP = SizingParams(risk_per_trade=0.02, taker_fee=0.00055, slippage=0.0002, liq_buffer=2.0)


# ---------------------------------------------------------------- стратегия
@pytest.mark.parametrize("params", [MAIN, {**MAIN, "adx_min": 20.0}, {**MAIN, "bull_long_only": True},
                                    {**MAIN, "adx_min": 20.0, "bull_long_only": True}])
def test_fixed_and_shadow_breakout_are_causal(params):
    cls = BreakoutShadow if set(params) - set(MAIN) else None
    fam = cls or "breakout"
    assert causal_diffs(fam, "4h", params) == []


def test_make_breakout_picks_shadow_class_only_with_filters():
    assert type(make_breakout("4h", MAIN)).__name__ == "Breakout"
    assert isinstance(make_breakout("4h", {**MAIN, "adx_min": 20.0}), BreakoutShadow)


def test_bull_long_only_blocks_shorts_in_bull_regime_only():
    b = aggregate_panel(PANEL, "4h")
    s = make_breakout("4h", {**MAIN, "n": 20, "bull_long_only": True})
    s.prepare(b)
    base = make_breakout("4h", {**MAIN, "n": 20})
    base.prepare(b)
    shorts_bull = shorts_bear = 0
    for k in range(len(b)):
        got = [(r, d.side, d.stop) for r, d in s.on_bar(k, {}) if isinstance(d, Enter)]
        ref = [(r, d.side, d.stop) for r, d in base.on_bar(k, {}) if isinstance(d, Enter)]
        if s.bull[k]:
            assert all(x[1] == LONG for x in got)
            shorts_bull += sum(x[1] == SHORT for x in ref)
        else:
            assert got == ref
            shorts_bear += sum(x[1] == SHORT for x in got)
    assert shorts_bull > 0 and shorts_bear > 0     # фильтр реально что-то отсекал


def test_adx_filter_only_removes_entries():
    b = aggregate_panel(PANEL, "4h")
    s = make_breakout("4h", {**MAIN, "n": 20, "adx_min": 20.0})
    s.prepare(b)
    base = make_breakout("4h", {**MAIN, "n": 20})
    base.prepare(b)
    removed = 0
    for k in range(len(b)):
        got = {(r, d.side) for r, d in s.on_bar(k, {}) if isinstance(d, Enter)}
        ref = {(r, d.side) for r, d in base.on_bar(k, {}) if isinstance(d, Enter)}
        assert got <= ref
        removed += len(ref - got)
        for r, _ in got:
            assert s.adx[r, k] >= 20.0
    assert removed > 0


# ------------------------------------------------------------------- guards
def test_drawdown_steps_halve_risk_and_recover():
    cfg = RiskCfg(starting_equity_usdt=25, risk_per_trade=0.02, daily_loss_limit=0.06, max_drawdown=0.40,
                  drawdown_steps=[(0.20, 0.5)])
    g = RiskGuard(cfg, 25.0, T0)
    g.update_equity(T0 + 1, 30.0)
    assert g.risk_multiplier(30.0) == 1.0
    assert g.risk_multiplier(24.01) == 1.0          # −19,97 %
    assert g.risk_multiplier(24.0) == 0.5           # ровно −20 %
    assert g.risk_multiplier(26.0) == 1.0           # вернулся выше порога
    g.update_equity(T0 + DAY_MS, 18.0)              # −40 % → остановка
    assert g.state.halted


def test_streak_rule_disabled_when_none():
    cfg = RiskCfg(starting_equity_usdt=25, risk_per_trade=0.02, daily_loss_limit=0.06, max_drawdown=0.40)
    g = RiskGuard(cfg, 25.0, T0)
    for i in range(10):
        assert g.on_trade_closed(T0 + i, -0.1) == []
    assert g.can_open(T0 + 20)[0]


def test_invalid_drawdown_step_rejected():
    with pytest.raises(ValueError):
        RiskCfg(starting_equity_usdt=25, risk_per_trade=0.02, daily_loss_limit=0.06, max_drawdown=0.40,
                drawdown_steps=[(0.5, 0.5)])


# ------------------------------------------------------------- корреляция
def _prisk(rules, panel=PANEL, tf="4h"):
    return PortfolioRisk(aggregate_panel(panel, tf), rules)


def test_corr_room_solves_heat_cap_exactly():
    pr = _prisk(PortfolioRules(corr_cap=0.04))
    k = len(pr.bars) - 1
    book = [Exposure(0, LONG, 10.0), Exposure(1, SHORT, 5.0), Exposure(2, LONG, 8.0)]
    cap = 40.0
    x = pr.corr_room(k, book, 3, LONG, cap)
    assert x > 0
    assert pr.corr_heat(k, book + [Exposure(3, LONG, x)]) == pytest.approx(cap, rel=1e-9)
    # без корреляции (ρ = 1 для одинаковых монет) H равен сумме рисков одной стороны
    c = pr.corr(k)
    assert np.allclose(np.diag(c), 1.0) and np.all(np.abs(c) <= 1 + 1e-12)


def test_corr_cap_never_allows_more_than_sum_cap():
    pr = _prisk(PortfolioRules(max_open_risk=0.06, downsize_to_fit=True, corr_cap=0.04))
    k = len(pr.bars) - 1
    inst = hype()
    book = [Exposure(0, LONG, 20.0), Exposure(1, LONG, 20.0)]
    s = pr.plan_entry(k=k, row=2, side=LONG, px=40.0, stop=38.0, equity=1000.0, available=1000.0, inst=inst,
                      sp=SP, risk_mult=1.0, book=book)
    assert not isinstance(s, Skip)
    assert s.planned_loss <= 20.0 + 1e-9                                   # остаток суммы 60 − 40
    assert pr.corr_heat(k, book + [Exposure(2, LONG, s.planned_loss)]) <= 40.0 + 1e-6


def test_downsize_and_skip_below_minimum():
    pr = _prisk(PortfolioRules(max_open_risk=0.06, downsize_to_fit=True))
    k = len(pr.bars) - 1
    inst = hype()
    # капитал 25: база 0,5 USDT; занято 1,3 из 1,5 → остаток 0,2 USDT → объём ниже 5 USDT
    book = [Exposure(0, LONG, 0.8), Exposure(1, SHORT, 0.5)]
    s = pr.plan_entry(k=k, row=2, side=LONG, px=40.0, stop=37.6, equity=25.0, available=25.0, inst=inst,
                      sp=SP, risk_mult=1.0, book=book)
    assert isinstance(s, Skip) and "после уменьшения" in s.reason
    s2 = pr.plan_entry(k=k, row=2, side=LONG, px=40.0, stop=37.6, equity=25.0, available=25.0, inst=inst,
                       sp=SP, risk_mult=1.0, book=[])
    assert not isinstance(s2, Skip) and s2.planned_loss <= 0.5


def test_risk_multiplier_scales_budget():
    pr = _prisk(PortfolioRules(max_open_risk=0.06, downsize_to_fit=True))
    k = len(pr.bars) - 1
    full = pr.plan_entry(k=k, row=0, side=LONG, px=40.0, stop=38.0, equity=1000.0, available=1000.0,
                         inst=hype(), sp=SP, risk_mult=1.0, book=[])
    half = pr.plan_entry(k=k, row=0, side=LONG, px=40.0, stop=38.0, equity=1000.0, available=1000.0,
                         inst=hype(), sp=SP, risk_mult=0.5, book=[])
    assert half.qty == pytest.approx(full.qty / 2, rel=0.01)


# --------------------------------------------------------- волатильность
def test_vol_multiplier_at_most_one_and_causal():
    rules = PortfolioRules(vol_scaling=True, vol_ref_days=60, vol_ref_min_days=20)
    pr = _prisk(rules)
    assert (pr.vol_mult <= 1.0).all() and (pr.vol_mult > 0).all()
    assert (pr.vol_mult < 1.0).any()
    t_cut = T0 + 120 * DAY_MS
    pr2 = PortfolioRisk(aggregate_panel(perturb_after(PANEL, t_cut), "4h"), rules)
    ks = np.flatnonzero(pr.bars.close_ts <= t_cut)
    np.testing.assert_array_equal(pr.vol_mult[ks], pr2.vol_mult[ks])


# ------------------------------------------------------------- ликвидность
def _liq_panel():
    """Корзина с мелким шагом цены (0,0001 при цене ~10 → 0,001 %)."""
    from bot.market import Instrument
    from tests.instruments import HYPE_INSTRUMENT, HYPE_TIERS
    p = copy.deepcopy(PANEL)
    inst = Instrument.from_api({**HYPE_INSTRUMENT, "priceFilter": {"tickSize": "0.0001"}}, HYPE_TIERS)
    for sd in p.sym.values():
        sd.inst = inst
    return p


def test_liquidity_rule_tick_turnover_zero_volume_and_listing():
    p = _liq_panel()
    # цены около 10, шаг HYPE 0.001 → 0,01 % — проходит
    p.turnover[:] = 20_000_000 / 96          # 20 млн в сутки
    p.volume[:] = 1.0
    p.turnover[1] = 5_000_000 / 96           # монета 1 — мало оборота
    p.volume[2, ::50] = 0.0                  # монета 2 — 2 % свечей без сделок
    for a in (p.open, p.high, p.low, p.close):
        a[3] = a[3] / 1000                   # монета 3 — цена ~0,01 → шаг 10 %
    b = aggregate_panel(p, "4h")
    m = liquidity_mask(b, LiquidityRule())
    late = b.close_ts > T0 + 31 * DAY_MS
    assert m[0, late].all()
    assert not m[1].any() and not m[2].any() and not m[3].any()
    # монета 5 листингуется на 40-й день: первые 30 дней после листинга входов нет
    k5 = np.flatnonzero(m[5])
    assert b.close_ts[k5[0]] >= T0 + 70 * DAY_MS
    assert not m[0, b.close_ts < T0 + 30 * DAY_MS].any()


def test_liquidity_mask_is_causal():
    p = _liq_panel()
    p.turnover[:] = 20_000_000 / 96
    p.volume[:] = 1.0
    t_cut = T0 + 100 * DAY_MS
    q = perturb_after(p, t_cut)
    j = int(np.searchsorted(q.ts, t_cut))
    q.turnover[:, j:] = 1.0                   # после среза оборот пропадает
    m1 = liquidity_mask(aggregate_panel(p, "4h"), LiquidityRule())
    b2 = aggregate_panel(q, "4h")
    m2 = liquidity_mask(b2, LiquidityRule())
    ks = np.flatnonzero(b2.close_ts <= t_cut)
    np.testing.assert_array_equal(m1[:, ks], m2[:, ks])
    assert not m2[:, b2.close_ts > t_cut + 31 * DAY_MS].any()


# --------------------------------------------------------------- бэктест
def test_backtester_uses_liquidity_rule():
    p = _liq_panel()
    p.turnover[:] = 20_000_000 / 96
    p.volume[:] = 1.0
    p.turnover[0] = 1.0
    s = make_breakout("4h", {**MAIN, "n": 20})
    cfg = PortfolioConfig(risk=RISK, costs=COSTS, initial_equity=1000.0,
                          rules=PortfolioRules(max_open_risk=0.2, liquidity=LiquidityRule()))
    res = PortfolioBacktester(p, s, cfg).run()
    assert len(res.trades) > 0
    assert "C0USDT" not in set(res.trades["symbol"])
    assert any("ликвидности" in x[2] for x in res.skips)
    assert (res.trades["decision_ts"] > T0 + 30 * DAY_MS).all()


def test_drawdown_step_halves_planned_loss_in_backtest():
    """Вторая сделка после просадки ≥ 20 % получает половину риска."""
    panel = flat_panel(3, n=800)
    j = 4 * 31          # открытие свечи 31 (1h): здесь исполняются выходы, решённые на свече 30
    for a in (panel.open, panel.high, panel.low, panel.close):
        a[0, j:] = a[2, j:] = 26.0
    risk = RiskCfg(starting_equity_usdt=1000, risk_per_trade=0.2, daily_loss_limit=0.50, max_drawdown=0.60,
                   drawdown_steps=[(0.20, 0.5)])
    script = {1: [(0, Enter(LONG, 25.0)), (2, Enter(LONG, 25.0))], 30: [(0, Exit()), (2, Exit())],
              40: [(1, Enter(LONG, 39.0))]}
    cfg = PortfolioConfig(risk=risk, costs=COSTS, initial_equity=1000.0,
                          rules=PortfolioRules(max_open_risk=1.0, downsize_to_fit=True))
    res = PortfolioBacktester(panel, Script(script), cfg).run()
    t = res.trades.set_index("symbol")
    eq_before = 1000.0 + t.loc["S0USDT", "net_pnl"] + t.loc["S2USDT", "net_pnl"]
    assert eq_before <= 800.0                                   # просадка ≥ 20 %
    assert t.loc["S1USDT", "planned_loss"] == pytest.approx(eq_before * 0.2 * 0.5, rel=0.02)


def test_margin_fraction_sizing_uses_max_leverage_and_liquidates_before_stop():
    from bot.risk.sizing import size_by_margin
    inst = hype()                                         # плечо до 75
    s = size_by_margin(LONG, 40.0, 37.5, margin_budget=6.25, available=25.0, inst=inst, p=SP)
    assert s.leverage == 75
    assert s.qty == pytest.approx(inst.floor_qty(6.25 * 75 / 40.0))
    assert s.liq_price > s.stop                           # ликвидация ближе стопа: стоп не спасает
    assert s.planned_loss == pytest.approx(s.margin + 2 * s.notional * SP.taker_fee)
    # режим по умолчанию не изменился: объём от риска до стопа
    pr = _prisk(PortfolioRules(sizing="risk"))
    r = pr.plan_entry(k=len(pr.bars) - 1, row=0, side=LONG, px=40.0, stop=37.5, equity=25.0, available=25.0,
                      inst=inst, sp=SP, risk_mult=1.0, book=[])
    assert r.planned_loss <= 0.5 + 1e-9 and r.liq_price < r.stop


def test_margin_stop_near_liquidation_keeps_max_leverage():
    """Правило «стоп у ликвидации»: плечо максимальное, стоп за 3,5 % расстояния до ликвидации до неё."""
    from bot.risk.sizing import size_by_margin
    inst = hype()
    for side, stop in ((LONG, 37.5), (SHORT, 42.5)):
        s = size_by_margin(side, 40.0, stop, margin_budget=6.25, available=25.0, inst=inst, p=SP,
                           stop_rule="near_liq", liq_gap=0.035)
        assert s.leverage == 75
        liq_dist = abs(40.0 - s.liq_price)
        assert abs(40.0 - s.stop) == pytest.approx(0.965 * liq_dist, abs=float(inst.tick_size))
        assert side * (s.stop - s.liq_price) > 0 and side * (40.0 - s.stop) > 0   # между входом и ликвидацией
        assert s.planned_loss < s.margin                    # стоп срабатывает раньше потери всей маржи


def test_margin_stop_gap_lowers_leverage_behind_strategy_stop():
    """Правило «ликвидация на 3,5 % цены дальше стопа»: стоп стратегии, плечо снижено."""
    from bot.risk.sizing import size_by_margin
    inst = hype()
    for side, stop in ((LONG, 37.5), (SHORT, 42.5)):
        s = size_by_margin(side, 40.0, stop, margin_budget=6.25, available=25.0, inst=inst, p=SP,
                           stop_rule="gap", liq_gap=0.035)
        assert s.stop == stop and 1 <= s.leverage < 75
        assert side * (s.stop - s.liq_price) >= 0.035 * 40.0 - 1e-9      # ликвидация ≥ 3,5 % цены за стопом
        assert s.margin == pytest.approx(6.25, rel=0.02)                 # маржа — по-прежнему 25 % баланса
        assert s.planned_loss == pytest.approx(s.qty * (2.5 + (40.0 + stop) * (SP.taker_fee + SP.slippage)))
    with pytest.raises(ValueError):
        size_by_margin(LONG, 40.0, 37.5, 6.25, 25.0, inst, SP, stop_rule="??")


@pytest.mark.parametrize("rule,reason", [("none", "liquidation"), ("near_liq", "stop"), ("gap", "stop")])
def test_margin_stop_rules_in_backtest(rule, reason):
    """Цена плавно падает на 6,5 %: без стопа у ликвидации — ликвидация, с правилами — выход по стопу."""
    panel = flat_panel(1, n=800)
    j = 4 * 31
    path = np.r_[40.0 - 0.01 * np.arange(1, 261), np.full(800 - j - 260, 37.4)]
    panel.close[0, j:] = panel.low[0, j:] = path
    panel.open[0, j:] = panel.high[0, j:] = np.r_[40.0, path[:-1]]
    risk = RiskCfg(starting_equity_usdt=100, risk_per_trade=0.02, daily_loss_limit=0.98, max_drawdown=0.99,
                   drawdown_steps=[])
    rules = PortfolioRules(max_open_risk=10.0, sizing="margin", margin_fraction=0.25, margin_stop=rule,
                           liq_gap=0.035)
    cfg = PortfolioConfig(risk=risk, costs=COSTS, initial_equity=100.0, max_open_risk=10.0, rules=rules)
    t = PortfolioBacktester(panel, Script({1: [(0, Enter(LONG, 37.5))]}), cfg).run().trades
    assert len(t) == 1 and t["exit_reason"].iloc[0] == reason
    p = t.iloc[0]
    if rule == "near_liq":
        assert p["leverage"] == 75 and p["liq_price"] < p["stop_initial"] < p["entry_price"]
    if rule == "gap":
        assert p["stop_initial"] == 37.5 and p["liq_price"] <= 37.5 - 0.035 * p["entry_price"] + 1e-9
    if rule != "none":
        assert -p["net_pnl"] < p["margin"]                  # потеря меньше маржи


def test_margin_stop_near_liquidation_does_not_survive_a_gap():
    """Скачок цены сразу за ликвидацию: стоп у ликвидации не спасает — ликвидация, потеря всей маржи."""
    panel = flat_panel(1, n=800)
    for a in (panel.open, panel.high, panel.low, panel.close):
        a[0, 4 * 31:] = 39.3
    risk = RiskCfg(starting_equity_usdt=100, risk_per_trade=0.02, daily_loss_limit=0.98, max_drawdown=0.99,
                   drawdown_steps=[])
    rules = PortfolioRules(max_open_risk=10.0, sizing="margin", margin_stop="near_liq", liq_gap=0.035)
    cfg = PortfolioConfig(risk=risk, costs=COSTS, initial_equity=100.0, max_open_risk=10.0, rules=rules)
    t = PortfolioBacktester(panel, Script({1: [(0, Enter(LONG, 37.5))]}), cfg).run().trades
    assert t["exit_reason"].iloc[0] == "liquidation"


@pytest.mark.parametrize("optimistic,reason", [(False, "liquidation"), (True, "stop")])
def test_stop_beats_liquidation_option(optimistic, reason):
    """Свеча открылась до стопа и ушла за ликвидацию: по умолчанию — ликвидация (исполнение стопа за ценой
    ликвидации), в оптимистичной модели — выход по стопу с убытком не больше маржи."""
    panel = flat_panel(1, n=800)
    j = 4 * 31
    panel.low[0, j] = panel.close[0, j] = 39.0
    for a in (panel.open, panel.high, panel.low, panel.close):
        a[0, j + 1:] = 39.0
    risk = RiskCfg(starting_equity_usdt=100, risk_per_trade=0.02, daily_loss_limit=0.98, max_drawdown=0.99,
                   drawdown_steps=[])
    rules = PortfolioRules(max_open_risk=10.0, sizing="margin", margin_stop="near_liq", liq_gap=0.035)
    cfg = PortfolioConfig(risk=risk, costs=COSTS, initial_equity=100.0, max_open_risk=10.0, rules=rules,
                          stop_beats_liq=optimistic)
    t = PortfolioBacktester(panel, Script({1: [(0, Enter(LONG, 37.5))]}), cfg).run().trades
    p = t.iloc[0]
    assert p["exit_reason"] == reason
    assert -p["gross_pnl"] <= p["margin"] + 1e-9                     # убыток по цене — не больше маржи
    if optimistic:
        assert p["liq_price"] > p["exit_price"] and -p["gross_pnl"] < p["margin"]
