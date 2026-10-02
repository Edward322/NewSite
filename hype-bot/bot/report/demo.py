"""Отчёт по демо-проверке одним скриптом (критерии — docs/DEMO_PROTOCOL.md, записаны до старта).

    python -m bot.report.demo [--mode demo]   → reports/DEMO_REPORT.md
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bot.config import PROJECT_ROOT, Config
from bot.engine.control import utc
from bot.report.compare import compare
from bot.report.execution import per_trade, summary, summary_text

DAY_MS = 86_400_000
BASELINE = PROJECT_ROOT / "reports" / "stage1" / "baseline.json"
MIN_DAYS, MIN_MARKET = 7, 5
N_BOOT = 10_000


@dataclass
class Verdict:
    status: str                      # ПРОЙДЕНО | РАНО | НЕ ПРОЙДЕНО
    criteria: dict                   # имя → True / False / None (недостаточно данных)


def ok_text(v) -> str:
    return "выполнен" if v is True else "НЕ выполнен" if v is False else "недостаточно данных"


def bootstrap_interval(base_r: list[float], n: int, seed: int = 7) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    sums = rng.choice(np.asarray(base_r), size=(N_BOOT, n), replace=True).sum(axis=1)
    return float(np.percentile(sums, 2.5)), float(np.median(sums)), float(np.percentile(sums, 97.5))


def build(cfg: Config, db, data_dir: Path, mode: str, now_ms: int, baseline: Path = BASELINE) -> tuple[str, Verdict]:
    start = int(db.get("started_ms"))
    eq0 = float(db.get("equity_start") or cfg.risk.starting_equity_usdt)
    days = (now_ms - start) / DAY_MS
    trades = db.trades()
    ev = db.events(since_ms=start)
    L = [f"# Отчёт по проверке на демо-счёте ({mode})", "",
         f"Период: {utc(start)} — {utc(now_ms)} ({days:.1f} сут.). Критерии записаны до старта: "
         "`docs/DEMO_PROTOCOL.md`.", "",
         "> **8 недель на 4h — это проверка движка и отсутствия явного провала, а не доказательство "
         "прибыльности.** На таком числе сделок перевес стратегии нельзя ни доказать, ни опровергнуть.", ""]
    crit = {}

    # ---------------------------------------------------------------- 1. движок
    mism = [e for e in ev if e["kind"] == "mismatch"]
    stop_ev = [e for e in ev if e["kind"] == "stop" and e["level"] in ("warn", "error")]
    bad_exit = [t for t in trades if t["exit_reason"] in ("no_stop", "liquidation")]
    loop_err = [e for e in ev if e["kind"] in ("loop", "hook") and e["level"] == "error"]
    hanging = db.orders(status=("pending", "sent", "unknown"))
    missed = [e for e in ev if e["kind"] == "missed"]
    late = [e for e in ev if e["kind"] == "late"]
    blk = [e for e in ev if e["kind"] == "blocker" and e["level"] == "warn"]
    crit["1. Движок"] = not (mism or stop_ev or bad_exit or loop_err or hanging)
    L += ["## 1. Движок", "",
          "| Проверка | Число | |", "|---|---|---|",
          f"| Расхождения с биржей | {len(mism)} | {'✅' if not mism else '❌'} |",
          f"| Позиция без биржевого стопа | {len(stop_ev)} | {'✅' if not stop_ev else '❌'} |",
          f"| Закрытия «нет стопа» / ликвидации | {len(bad_exit)} | {'✅' if not bad_exit else '❌'} |",
          f"| Необработанные ошибки | {len(loop_err)} | {'✅' if not loop_err else '❌'} |",
          f"| Незавершённые ордера | {len(hanging)} | {'✅' if not hanging else '❌'} |",
          f"| Для информации: пропуски свечей / поздние свечи / блокировки входов | {len(missed)} / {len(late)} / "
          f"{len(blk)} | |", ""]
    for e in (mism + stop_ev + loop_err)[:10]:
        L.append(f"- {utc(e['ts'])}: {e['message'][:200]}")
    L.append(f"Итог: **{ok_text(crit['1. Движок'])}**")
    L.append("")

    # ---------------------------------------------------------------- 2. исполнение
    df = per_trade(trades, data_dir, cfg.costs)
    s = summary(df, cfg.costs)
    if s.get("market_n", 0) >= MIN_MARKET:
        crit["2. Исполнение"] = bool(s["slip_ok"] and s["stop_ok"] and s["fee_ok"])
    else:
        crit["2. Исполнение"] = None
    L += ["## 2. Исполнение", "", summary_text(s), "",
          f"Рыночных исполнений: {s.get('market_n', 0)} (нужно ≥ {MIN_MARKET}), выходов по стопу: {s.get('stop_n', 0)}.",
          f"Итог: **{ok_text(crit['2. Исполнение'])}**", ""]

    # ---------------------------------------------------------------- 3. совпадение
    C = compare(cfg, db, data_dir, now_ms)
    crit["3. Совпадение с бэктестом"] = C.passed
    L += ["## 3. Совпадение с бэктестом", "", "```", C.text(), "```",
          f"Итог: **{ok_text(crit['3. Совпадение с бэктестом'])}**", ""]

    # ---------------------------------------------------------------- 4. результат
    R = [t["r_multiple"] for t in trades if t["r_multiple"] is not None]
    base = json.loads(Path(baseline).read_text(encoding="utf-8"))
    rows = db.equity_rows(start)
    eqs = pd.Series([eq0] + [r["equity"] for r in rows])
    dd = float((1 - eqs / eqs.cummax()).max())
    halted = bool(db.get("halted"))
    half = bool((1 - eqs / eqs.cummax() >= 0.20 - 1e-9).any())
    L += ["## 4. Результат и просадка", ""]
    if R:
        lo, med, hi = bootstrap_interval(base["r_multiples"], len(R))
        in_band = lo <= sum(R) <= hi
        L.append(f"Закрытых сделок {len(R)}, сумма R {sum(R):+.2f}. Бутстрэп из истории варианта "
                 f"{base['variant']} ({len(base['r_multiples'])} сделок, данные не чистые): 95 % интервал суммы R "
                 f"для {len(R)} сделок — [{lo:+.2f}; {hi:+.2f}], медиана {med:+.2f} → "
                 f"{'внутри' if in_band else 'ВНЕ интервала'}.")
    else:
        in_band = None
        L.append("Закрытых сделок нет — сравнивать с историей нечего.")
    eq_now = float(rows[-1]["equity"]) if rows else eq0
    net = sum(t["net_pnl"] for t in trades)
    L.append(f"Капитал бота: {eq0:.2f} → {eq_now:.2f} USDT ({(eq_now / eq0 - 1) * 100:+.1f}%), закрытые сделки "
             f"{net:+.2f} USDT; максимальная просадка {dd * 100:.1f}% (лимит {cfg.risk.max_drawdown:.0%}); "
             f"остановка: {'ДА' if halted else 'нет'}; режим половинного риска был: {'да' if half else 'нет'}.")
    dd_ok = (not halted) and dd < cfg.risk.max_drawdown
    crit["4. Результат и просадка"] = (None if in_band is None else bool(in_band)) if dd_ok else False
    L.append(f"Итог: **{ok_text(crit['4. Результат и просадка'])}**")
    L.append("")

    # ---------------------------------------------------------------- сделки
    if trades:
        L += ["## Сделки", "", "| Закрыта | Монета | Сторона | Объём | Вход | Выход | Причина | USDT | R |",
              "|---|---|---|---|---|---|---|---|---|"]
        for t in trades:
            L.append(f"| {utc(t['exit_ts'])} | {t['symbol']} | {'лонг' if t['side'] > 0 else 'шорт'} | {t['qty']:g} | "
                     f"{t['entry_price']:.6g} | {t['exit_price']:.6g} | {t['exit_reason']} | {t['net_pnl']:+.3f} | "
                     f"{(t['r_multiple'] or 0):+.2f} |")
        L.append("")

    # ---------------------------------------------------------------- итог
    if any(v is False for v in crit.values()):
        status = "НЕ ПРОЙДЕНО"
    elif days < MIN_DAYS or not trades or any(v is None for v in crit.values()):
        status = "РАНО"
    else:
        status = "ПРОЙДЕНО"
    head = [f"## Итог: **{status}**", ""]
    head += [f"- {k}: {ok_text(v)}" for k, v in crit.items()]
    if status == "РАНО":
        head.append(f"- Нужно ≥ {MIN_DAYS} суток, хотя бы одна закрытая сделка и ≥ {MIN_MARKET} рыночных исполнений "
                    "— демо продолжается.")
    head.append("- Реальные деньги — только после вашего явного «да».")
    head.append("")
    text = "\n".join(L[:7] + head + L[7:]) + "\n"
    return text, Verdict(status, crit)


def main(argv=None) -> int:
    from bot.config import load_bot_config
    from bot.engine.control import paths
    from bot.engine.state import StateDB
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "live"], default="demo")
    a = ap.parse_args(argv)
    cfg = load_bot_config()
    db_path, data_dir = paths(cfg, a.mode)
    if not db_path.exists():
        print("Бот ещё не запускался в этом режиме.")
        return 1
    db = StateDB(db_path)
    text, v = build(cfg, db, data_dir, a.mode, int(time.time() * 1000))
    out = PROJECT_ROOT / "reports" / ("DEMO_REPORT.md" if a.mode == "demo" else "LIVE_REPORT.md")
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Отчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
