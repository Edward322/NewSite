"""Сравнение сделок движка с бэктестом того же периода на тех же данных (критерий 3, docs/DEMO_PROTOCOL.md).

    python -m bot.report.compare [--mode demo|live]

Бэктест: тот же код (bot/backtest/portfolio.py), конфиг, начальный капитал, первая свеча и 15m-свечи,
которые скачал бот. Сделки сопоставляются по ключу «монета + сторона + свеча решения»:
  - сигналы — пара есть у обеих сторон;
  - объёмы — разница не больше шага объёма или 2 %;
  - выходы — та же причина на той же 15-минутной свече (или обе позиции ещё открыты).
Исключения: решение или выход на последней незакрытой свече 4h («ожидает»); расхождения после
операционного события движка (бот выключен, свеча обработана поздно, блокировка входов).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from bot.config import Config
from bot.data.bars import TF_MINUTES
from bot.engine.control import utc
from bot.market import Instrument
from bot.report.shadow import run_variants

M15 = 15 * 60_000
OPERATIONAL = ("missed", "late", "data", "blocker", "kill", "halt")


@dataclass
class Row:
    symbol: str
    side: int
    decision_ts: int
    status: str
    note: str = ""
    engine: dict | None = None
    bt: dict | None = None


@dataclass
class Comparison:
    rows: list[Row] = field(default_factory=list)
    first_operational: int | None = None
    operational: list[dict] = field(default_factory=list)
    end_ms: int = 0

    def count(self, status: str) -> int:
        return sum(1 for r in self.rows if r.status == status)

    @property
    def unexplained(self) -> list[Row]:
        return [r for r in self.rows if r.status in ("только движок", "только бэктест", "объём", "выход")]

    @property
    def passed(self) -> bool:
        return not self.unexplained

    def text(self) -> str:
        L = [f"Сравнение движка с бэктестом (до {utc(self.end_ms)}): сделок/позиций {len(self.rows)}; "
             f"совпадает {self.count('совпадает')}, ожидает закрытия свечи {self.count('ожидает')}, "
             f"объяснено операционными событиями {self.count('объяснено')}, НЕ совпадает {len(self.unexplained)}"]
        if self.first_operational:
            L.append(f"Первое операционное событие: {utc(self.first_operational)}")
        for r in self.rows:
            e, b = r.engine or {}, r.bt or {}
            L.append(f"  {utc(r.decision_ts)} {r.symbol:13s} {'лонг' if r.side > 0 else 'шорт'}  {r.status:16s} "
                     f"объём {e.get('qty', '—')} / {b.get('qty', '—')}; выход {e.get('exit_reason', 'открыта')} / "
                     f"{b.get('exit_reason', 'открыта')}{'; ' + r.note if r.note else ''}")
        return "\n".join(L)


def _records(trades, open_pos) -> dict:
    out = {}
    for t in trades:
        out[(t["symbol"], int(t["side"]), int(t["decision_ts"]))] = dict(t)
    for p in open_pos:
        out[(p["symbol"], int(p["side"]), int(p["decision_ts"]))] = {**p, "exit_reason": None, "exit_ts": None}
    return out


def compare(cfg: Config, db, data_dir: Path, now_ms: int) -> Comparison:
    tf = TF_MINUTES[cfg.strategy.timeframe] * 60_000
    start = int(db.get("started_ms"))
    eq0 = float(db.get("equity_start") or cfg.risk.starting_equity_usdt)
    last_bar = (now_ms // tf) * tf                     # закрытие последней полной свечи
    res = run_variants(cfg, data_dir, int(db.get("anchor_ms")), start, now_ms, eq0, shadows=False)[0].res
    bt_trades = res.trades.to_dict("records") if len(res.trades) else []
    eng = _records(db.trades(), [{"symbol": p.symbol, "side": p.side, "qty": p.qty, "decision_ts": p.decision_ts,
                                  "entry_price": p.entry_price} for p in db.positions().values()])
    bt = _records(bt_trades, res.open_positions)
    ops = [e for e in db.events(since_ms=start, kinds=OPERATIONAL) if e["level"] in ("warn", "error")]
    ops += [e for e in db.events(since_ms=start, kinds=("skip",))
            if "блокировка" in e["message"] or "запрещены" in e["message"]]
    first_op = min((e["ts"] for e in ops), default=None)
    steps = {}
    C = Comparison(first_operational=first_op, operational=ops, end_ms=now_ms)
    for key in sorted(set(eng) | set(bt), key=lambda k: (k[2], k[0])):
        sym, side, dts = key
        e, b = eng.get(key), bt.get(key)
        row = Row(sym, side, dts, "", engine=e, bt=b)
        if sym not in steps:
            try:
                steps[sym] = float(Instrument.from_files(Path(data_dir), sym).qty_step)
            except FileNotFoundError:
                steps[sym] = 0.0
        late_exit = any(x and x.get("exit_ts") and x["exit_ts"] >= last_bar for x in (e, b))
        if dts >= last_bar:
            row.status = "ожидает"
        elif e and b:
            qty_ok = abs(e["qty"] - b["qty"]) <= max(steps[sym], 0.02 * b["qty"]) + 1e-12
            if e["exit_reason"] is None and b["exit_reason"] is None:
                exit_ok = True
            elif e["exit_reason"] is None or b["exit_reason"] is None:
                exit_ok = None if late_exit else False
            else:
                exit_ok = (e["exit_reason"] == b["exit_reason"] and e["exit_ts"] // M15 == b["exit_ts"] // M15)
            if exit_ok is None:
                row.status = "ожидает"
            elif not qty_ok:
                row.status = "объём"
            elif not exit_ok:
                row.status = "выход"
            else:
                row.status = "совпадает"
        else:
            row.status = "только движок" if e else "только бэктест"
        C.rows.append(row)
    # Каскад: первое расхождение объяснено, только если на его свече у движка была операционная причина;
    # тогда и все следующие расхождения считаются его следствием.
    bad = [r for r in C.rows if r.status in ("только движок", "только бэктест", "объём", "выход")]
    if bad:
        d0 = bad[0].decision_ts
        op_bars = {int(t): why for t, why in (db.get("op_bars") or [])}
        if d0 in op_bars:
            for r in bad:
                r.note = f"следствие свечи {utc(d0)}: {op_bars[d0]}"
                r.status = "объяснено"
    return C


def main(argv=None) -> int:
    from bot.config import PROJECT_ROOT, load_bot_config
    from bot.engine.control import paths
    from bot.engine.state import StateDB
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "live"], default="demo")
    a = ap.parse_args(argv)
    cfg = load_bot_config()
    db_path, data_dir = paths(cfg, a.mode)
    db = StateDB(db_path)
    C = compare(cfg, db, data_dir, int(time.time() * 1000))
    text = C.text()
    out = PROJECT_ROOT / "reports" / "live"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"compare_{a.mode}.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if C.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
