"""Недельный отчёт: капитал, сделки, ограничители, ошибки, исполнение, теневые варианты."""
from __future__ import annotations

from datetime import datetime, timezone

from bot.config import Config
from bot.report.execution import per_trade, summary, summary_text
from bot.report.shadow import run_variants

WEEK_MS = 7 * 86_400_000


def _d(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def weekly_text(cfg: Config, db, data_dir, mode: str, now_ms: int, equity_now: float | None) -> str:
    start = int(db.get("started_ms") or now_ms)
    eq0 = float(db.get("equity_start") or cfg.risk.starting_equity_usdt)
    week0 = max(start, now_ms - WEEK_MS)
    rows = db.equity_rows()
    L = [f"Недельный отчёт ({'демо' if mode == 'demo' else 'РЕАЛЬНЫЙ счёт' if mode == 'live' else mode}), "
         f"{_d(week0)} — {_d(now_ms)}"]
    eq_week0 = next((r["equity"] for r in rows if r["ts"] >= week0), eq0)
    if equity_now is not None:
        import pandas as pd
        e = pd.Series([eq0] + [r["equity"] for r in rows] + [equity_now])
        peak, dd = float(e.max()), float((1 - e / e.cummax()).max())
        L.append(f"Капитал: за неделю {eq_week0:.2f} → {equity_now:.2f} USDT ({(equity_now / eq_week0 - 1) * 100:+.1f}%), "
                 f"с начала {eq0:.2f} → {equity_now:.2f} ({(equity_now / eq0 - 1) * 100:+.1f}%); пик {peak:.2f}, "
                 f"макс. просадка с начала {dd * 100:.1f}%")
    tr_all = db.trades()
    tr = [t for t in tr_all if t["exit_ts"] >= week0]
    if tr:
        wins = sum(1 for t in tr if t["net_pnl"] > 0)
        L.append(f"Сделок закрыто за неделю: {len(tr)} (в плюсе {wins}), итог {sum(t['net_pnl'] for t in tr):+.2f} USDT, "
                 f"средний R {sum((t['r_multiple'] or 0) for t in tr) / len(tr):+.2f}")
        for t in tr:
            L.append(f"  {_d(t['exit_ts'])} {t['symbol']} {'лонг' if t['side'] > 0 else 'шорт'}: {t['net_pnl']:+.3f} USDT "
                     f"({(t['r_multiple'] or 0):+.2f} R, {t['exit_reason']})")
    else:
        L.append("Сделок за неделю не закрыто.")
    L.append(f"Открытых позиций: {len(db.positions())}")
    ev = db.events(since_ms=week0)
    guards = [e for e in ev if e["kind"] in ("guard", "halt", "kill", "blocker")]
    errors = [e for e in ev if e["level"] == "error"]
    mism = [e for e in ev if e["kind"] == "mismatch"]
    L.append(f"Ограничители и блокировки: {len(guards)} событий; расхождений с биржей: {len(mism)}; ошибок: {len(errors)}")
    for e in (guards + mism)[-5:]:
        L.append(f"  {_d(e['ts'])} {e['message'][:160]}")
    s = summary(per_trade(tr_all, data_dir, cfg.costs), cfg.costs)
    L.append(summary_text(s) + " (с начала работы)")
    try:
        anchor = int(db.get("anchor_ms"))
        res = run_variants(cfg, data_dir, anchor, start, now_ms, eq0)
        L.append("Теневые варианты (виртуально, без ордеров; тот же риск; с начала работы):")
        for v in res:
            L.append("  " + v.line())
        L.append("Перевеса варианта над основной стратегией за несколько недель это не доказывает.")
    except Exception as e:      # отчёт не должен ломаться из-за данных
        L.append(f"Теневые варианты не посчитаны: {e}")
    return "\n".join(L)
