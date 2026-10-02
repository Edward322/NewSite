"""Качество исполнения на бирже против допущений бэктеста.

Для каждой закрытой сделки из базы движка:
  - вход (рыночный): проскальзывание = сторона × (цена входа / открытие 15m-свечи исполнения − 1);
    в бэктесте вход = открытие следующей 15m-свечи × (1 + проскальзывание 0,02 %);
  - рыночный выход: то же с обратным знаком (положительное — хуже);
  - выход по стопу: факт против модели бэктеста на той же 15m-свече
    (стоп хуже на stop_penetration пути до экстремума свечи и на проскальзывание);
    положительное — на бирже хуже модели;
  - комиссии: фактическая ставка (комиссия / оборот) против taker_fee.
Цены свечей — из данных, которые бот скачал сам (data/live/<режим>).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bot.config import CostsCfg
from bot.data import store

M15 = 15 * 60_000


class Candles:
    def __init__(self, data_dir: Path):
        self.dir, self.cache = Path(data_dir), {}

    def at(self, symbol: str, t_ms: int) -> dict | None:
        if symbol not in self.cache:
            p = store.kline_path(self.dir, symbol, store.KIND_LAST, "15")
            self.cache[symbol] = pd.read_parquet(p).set_index("ts") if p.exists() else None
        df = self.cache[symbol]
        ts = (int(t_ms) // M15) * M15
        if df is None or ts not in df.index:
            return None
        return df.loc[ts].to_dict()


def per_trade(trades: list[dict], data_dir: Path, costs: CostsCfg) -> pd.DataFrame:
    c = Candles(data_dir)
    rows = []
    for t in trades:
        side = int(t["side"])
        ce = c.at(t["symbol"], t["entry_ts"])
        cx = c.at(t["symbol"], t["exit_ts"])
        r = {"symbol": t["symbol"], "side": side, "entry_ts": t["entry_ts"], "exit_ts": t["exit_ts"],
             "exit_reason": t["exit_reason"], "entry_slip": np.nan, "exit_slip": np.nan, "stop_excess": np.nan,
             "entry_fee_rate": t["entry_fee"] / (t["qty"] * t["entry_price"]) if t["qty"] else np.nan,
             "exit_fee_rate": t["exit_fee"] / (t["qty"] * t["exit_price"]) if t["qty"] and t["exit_price"] else np.nan,
             "delay_s": (t["entry_ts"] - t["decision_ts"]) / 1000}
        if ce:
            r["entry_slip"] = side * (t["entry_price"] / ce["open"] - 1)
        if cx:
            if t["exit_reason"] == "stop" and t["stop_final"]:
                stop = float(t["stop_final"])
                if side > 0:
                    base = min(cx["open"], stop)
                    model = (base - costs.stop_penetration * max(0.0, base - cx["low"])) * (1 - costs.slippage)
                else:
                    base = max(cx["open"], stop)
                    model = (base + costs.stop_penetration * max(0.0, cx["high"] - base)) * (1 + costs.slippage)
                r["stop_excess"] = side * (model - t["exit_price"]) / stop
            elif t["exit_reason"] not in ("liquidation",):
                r["exit_slip"] = -side * (t["exit_price"] / cx["open"] - 1)
        rows.append(r)
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame, costs: CostsCfg) -> dict:
    """Средние значения и сравнение с допущениями бэктеста (True — не хуже заложенного)."""
    if df.empty:
        return {"n": 0}
    market = pd.concat([df["entry_slip"], df["exit_slip"]]).dropna()
    fees = pd.concat([df["entry_fee_rate"], df["exit_fee_rate"]]).dropna()
    stops = df["stop_excess"].dropna()
    out = {
        "n": len(df), "market_n": len(market), "stop_n": len(stops),
        "slip_mean": float(market.mean()) if len(market) else np.nan,
        "slip_assumed": costs.slippage,
        "stop_excess_mean": float(stops.mean()) if len(stops) else np.nan,
        "fee_mean": float(fees.mean()) if len(fees) else np.nan,
        "fee_assumed": costs.taker_fee,
        "delay_median_s": float(df["delay_s"].median()),
    }
    out["slip_ok"] = bool(len(market) == 0 or out["slip_mean"] <= costs.slippage + 1e-12)
    out["stop_ok"] = bool(len(stops) == 0 or out["stop_excess_mean"] <= 1e-12)
    out["fee_ok"] = bool(len(fees) == 0 or out["fee_mean"] <= costs.taker_fee * 1.0001)
    return out


def summary_text(s: dict) -> str:
    if not s.get("n"):
        return "Исполнение: закрытых сделок ещё нет."
    pct = lambda x: "—" if x != x else f"{x * 100:+.3f}%"  # noqa: E731
    return (f"Исполнение ({s['n']} сделок): проскальзывание рыночных ордеров в среднем {pct(s['slip_mean'])} "
            f"(заложено {s['slip_assumed'] * 100:.3f}%) — {'не хуже' if s['slip_ok'] else 'ХУЖЕ'}; "
            f"стопы относительно модели бэктеста {pct(s['stop_excess_mean'])} — "
            f"{'не хуже' if s['stop_ok'] else 'ХУЖЕ'}; комиссия {pct(s['fee_mean'])} "
            f"(заложено {s['fee_assumed'] * 100:.3f}%) — {'не хуже' if s['fee_ok'] else 'ХУЖЕ'}; "
            f"задержка входа после закрытия свечи: медиана {s['delay_median_s']:.0f} с.")


def baseline(trades: list[dict], data_dir: Path, costs: CostsCfg) -> dict:
    """База исполнения (по демо): средние проскальзывание и разница стопов; без сделок — допущения бэктеста."""
    s = summary(per_trade(trades, data_dir, costs), costs) if trades else {"n": 0}
    slip = s.get("slip_mean") if s.get("market_n") else None
    stop = s.get("stop_excess_mean") if s.get("stop_n") else None
    return {"slip": float(slip) if slip is not None and slip == slip else costs.slippage,
            "stop": float(stop) if stop is not None and stop == stop else 0.0,
            "n": int(s.get("n", 0)), "source": "демо" if trades else "допущения бэктеста"}


def execution_alarm(trades: list[dict], data_dir: Path, costs: CostsCfg, base: dict, window: int, min_n: int,
                    max_slip_excess: float, max_stop_excess: float) -> str | None:
    """Текст причины, если исполнение заметно хуже базы (docs/LIVE_PROTOCOL.md), иначе None."""
    recent = trades[-window:]
    df = per_trade(recent, data_dir, costs)
    if df.empty:
        return None
    market = pd.concat([df["entry_slip"], df["exit_slip"]]).dropna()
    stops = df["stop_excess"].dropna()
    if len(market) < min_n:
        return None
    if market.mean() > base["slip"] + max_slip_excess:
        return (f"проскальзывание {market.mean() * 100:.3f}% против {base['slip'] * 100:.3f}% на демо "
                f"(допуск +{max_slip_excess * 100:.2f} п.п.)")
    if len(stops) and stops.mean() > base["stop"] + max_stop_excess:
        return (f"стопы хуже модели на {stops.mean() * 100:.3f}% против {base['stop'] * 100:.3f}% на демо "
                f"(допуск +{max_stop_excess * 100:.2f} п.п.)")
    return None
