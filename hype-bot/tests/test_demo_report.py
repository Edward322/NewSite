"""Скрипт сравнения с бэктестом и отчёт по демо (docs/DEMO_PROTOCOL.md) на имитаторе биржи."""
import json

import pytest

from bot.data.panel import DAY_MS
from bot.report.compare import compare
from bot.report.demo import bootstrap_interval, build
from tests.engine_env import make_env
from tests.helpers import T0

START, END = 108, 128


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("demo")
    env = make_env(tmp, start_day=START, equity=25.0)
    assert env.engine.start()
    env.run_until(T0 + END * DAY_MS + 3 * 3_600_000)
    return env


def baseline(tmp_path, rs):
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps({"variant": "B", "r_multiples": rs}), encoding="utf-8")
    return p


def test_compare_matches_everything_on_simulator(run):
    C = compare(run.cfg, run.db, run.data.dir, run.clock.now_ms())
    assert len(C.rows) >= 5
    assert C.passed, C.text()
    assert {r.status for r in C.rows} <= {"совпадает", "ожидает"}
    assert C.count("совпадает") >= 5


def test_demo_report_passes_engine_execution_and_match(run, tmp_path):
    text, v = build(run.cfg, run.db, run.data.dir, "sim", run.clock.now_ms(),
                    baseline(tmp_path, [-1.0, -1.0, 0.5, 3.0, -0.9, 6.0, -1.0, 1.2]))
    assert v.criteria["1. Движок"] is True
    assert v.criteria["2. Исполнение"] is True
    assert v.criteria["3. Совпадение с бэктестом"] is True
    assert v.criteria["4. Результат и просадка"] in (True, False)
    assert "проверка движка и отсутствия явного провала" in text
    assert v.status in ("ПРОЙДЕНО", "НЕ ПРОЙДЕНО")


def test_result_outside_bootstrap_band_fails(run, tmp_path):
    # история, где каждая сделка −1 R: любой иной итог демо — вне интервала
    _, v = build(run.cfg, run.db, run.data.dir, "sim", run.clock.now_ms(), baseline(tmp_path, [-1.0] * 50))
    total = sum(t["r_multiple"] for t in run.db.trades())
    assert v.criteria["4. Результат и просадка"] is (abs(total + len(run.db.trades())) < 1e-9)
    assert v.status == "НЕ ПРОЙДЕНО" or v.criteria["4. Результат и просадка"]


def test_too_early_is_reported_as_early(tmp_path):
    env = make_env(tmp_path, start_day=START, equity=25.0)
    assert env.engine.start()
    env.run_until(T0 + (START + 3) * DAY_MS)
    _, v = build(env.cfg, env.db, env.data.dir, "sim", env.clock.now_ms(), baseline(tmp_path, [1.0, -1.0]))
    assert v.status in ("РАНО", "НЕ ПРОЙДЕНО")
    assert v.status == "РАНО" or any(x is False for x in v.criteria.values())


def test_mismatch_event_fails_engine_criterion(run, tmp_path):
    run.db.event(run.clock.now_ms() - 1000, "error", "mismatch", "тест: расхождение")
    try:
        _, v = build(run.cfg, run.db, run.data.dir, "sim", run.clock.now_ms(), baseline(tmp_path, [1.0, -1.0]))
        assert v.criteria["1. Движок"] is False and v.status == "НЕ ПРОЙДЕНО"
    finally:
        run.db.c.execute("DELETE FROM events WHERE message='тест: расхождение'")


def test_compare_flags_tampered_volume(run):
    t = run.db.trades()[0]
    run.db.c.execute("UPDATE trades SET qty=qty*1.5 WHERE id=?", (t["id"],))
    try:
        C = compare(run.cfg, run.db, run.data.dir, run.clock.now_ms())
        assert not C.passed and any(r.status == "объём" for r in C.rows)
    finally:
        run.db.c.execute("UPDATE trades SET qty=? WHERE id=?", (t["qty"], t["id"]))


def test_divergence_after_downtime_is_explained(tmp_path):
    env = make_env(tmp_path, start_day=START, equity=25.0)
    assert env.engine.start()
    env.run_until(T0 + (START + 4) * DAY_MS)
    env.clock.advance_to(T0 + (START + 7) * DAY_MS)     # компьютер выключен трое суток
    env.run_until(T0 + END * DAY_MS)
    assert env.db.events(kinds=("missed",))
    C = compare(env.cfg, env.db, env.data.dir, env.clock.now_ms())
    assert C.passed, C.text()
    assert C.count("объяснено") >= 1


def test_bootstrap_interval_is_sensible():
    lo, med, hi = bootstrap_interval([-1.0, 2.0], 10)
    assert -10 <= lo < med < hi <= 20
