import pandas as pd

from research import wf


def test_trade_stats_keys_do_not_collide_with_strategy_params():
    from bot.strategy.candidates import FAMILIES
    stat_keys = set(wf.trade_stats(pd.DataFrame()).keys())
    for cls in FAMILIES.values():
        assert not (stat_keys & set(cls.grid)), cls.family


def test_folds_cover_dev_without_overlap():
    d0, d1 = 0, 456 * wf.DAY_MS
    f = wf.folds(d0, d1)
    assert f[0][0] == d0 and f[-1][2] == d1
    for (a, b, c), (a2, b2, c2) in zip(f, f[1:]):
        assert c == b2 and b - a == wf.TRAIN_DAYS * wf.DAY_MS


def test_select_prefers_plateau_over_isolated_peak():
    from bot.strategy.candidates import grid_configs
    rows = []
    for p in grid_configs("squeeze"):
        sqn = 1.0 if p["k_trail"] in (3.0, 4.0) and p["n"] == 40 else 0.0
        if p == {"n": 20, "p": 0.15, "k_stop": 1.5, "k_trail": 2.0}:
            sqn = 5.0                      # одиночный пик среди нулей
        rows.append({**p, "trades": 100, "sqn": sqn})
    best, score = wf.select(pd.DataFrame(rows), "squeeze")
    assert best["n"] == 40 and best["k_trail"] in (3.0, 4.0)


def test_select_ignores_configs_with_too_few_trades():
    from bot.strategy.candidates import grid_configs
    rows = [{**p, "trades": 10, "sqn": 3.0} for p in grid_configs("squeeze")]
    assert wf.select(pd.DataFrame(rows), "squeeze") == (None, 0.0)
