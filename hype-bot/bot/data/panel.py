"""Данные корзины монет в общей сетке времени (база — свечи 15m).

Все монеты приводятся к одной сетке 15-минутных меток UTC; где у монеты свечи нет
(до листинга, пропуск), стоят NaN — бэктестер такие свечи пропускает.

Когда значение становится известно (это и есть защита от заглядывания в будущее):
  - свеча 15m с открытием T — известна в T + 15 мин;
  - финансирование с меткой T — в момент выплаты T;
  - премиальный индекс, открытый интерес, лонг/шорт (1h) с меткой T — в T + 1 ч
    (метка ряда — начало интервала; берём консервативно его конец);
  - индекс страха и жадности за день D — в D + 1 день.
Поле known_ts у каждого вспомогательного ряда — момент, начиная с которого значение
можно использовать.

Отложенные данные (holdout) открываются только load_basket_holdout с подтверждением
и записью в reports/holdout_access.log — так же, как для одной монеты (split.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bot.config import PROJECT_ROOT, Config
from bot.data import store
from bot.data.bars import MinuteData, to_ms
from bot.data.split import ACCESS_LOG, HOLDOUT_CONFIRM
from bot.market import Instrument

BASE_MS = 15 * 60_000
HOUR_MS = 3_600_000
DAY_MS = 86_400_000
BASKET_DIR = PROJECT_ROOT / "data" / "basket"
AUX = {"premium_1h": ("close", HOUR_MS), "oi_1h": ("oi", HOUR_MS), "lsr_1h": ("buy_ratio", HOUR_MS)}


@dataclass
class Series:
    """Вспомогательный ряд: значение value[i] можно использовать с момента known_ts[i]."""
    known_ts: np.ndarray
    value: np.ndarray

    def asof(self, t: np.ndarray) -> np.ndarray:
        """Последнее известное к моменту t значение (NaN, если ещё ничего не известно)."""
        idx = np.searchsorted(self.known_ts, t, "right") - 1
        out = np.full(len(t), np.nan)
        ok = idx >= 0
        out[ok] = self.value[idx[ok]]
        return out


@dataclass
class SymbolData:
    symbol: str
    inst: Instrument
    funding_ts: np.ndarray
    funding_rate: np.ndarray
    aux: dict[str, Series] = field(default_factory=dict)


@dataclass
class Panel:
    """Свечи 15m всех монет на общей сетке: массивы формы (n_symbols, n_base)."""
    symbols: list[str]           # торгуемые монеты (порядок = строки массивов)
    signal_only: list[str]       # только для сигналов (их строки идут после торгуемых)
    ts: np.ndarray               # время открытия 15m-свечи, общая сетка
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    turnover: np.ndarray
    sym: dict[str, SymbolData]
    fear_greed: Series | None
    end_ms: int                  # данные доступны строго до этого момента

    @property
    def all_symbols(self) -> list[str]:
        return self.symbols + self.signal_only

    def row(self, symbol: str) -> int:
        return self.all_symbols.index(symbol)

    def slice(self, start_ms: int | None, end_ms: int | None) -> "Panel":
        """Окно сетки [start_ms, end_ms) — ряды-массивы режутся без копирования."""
        lo = 0 if start_ms is None else int(np.searchsorted(self.ts, start_ms, "left"))
        hi = len(self.ts) if end_ms is None else int(np.searchsorted(self.ts, end_ms, "left"))
        cut = {c: getattr(self, c)[:, lo:hi] for c in ("open", "high", "low", "close", "volume", "turnover")}
        return Panel(symbols=self.symbols, signal_only=self.signal_only, ts=self.ts[lo:hi], sym=self.sym,
                     fear_greed=self.fear_greed, end_ms=self.end_ms if end_ms is None else min(end_ms, self.end_ms),
                     **cut)

    def minute_data(self, symbol: str) -> MinuteData:
        """Ряд одной монеты как MinuteData с шагом 15m (mark-цена = последняя): для старого движка."""
        r = self.row(symbol)
        ok = ~np.isnan(self.close[r])
        return MinuteData(ts=self.ts[ok], open=self.open[r][ok], high=self.high[r][ok], low=self.low[r][ok],
                          close=self.close[r][ok], volume=self.volume[r][ok], mark_open=self.open[r][ok],
                          mark_high=self.high[r][ok], mark_low=self.low[r][ok], mark_close=self.close[r][ok],
                          base_ms=BASE_MS)


def _read(data_dir: Path, symbol: str, name: str) -> pd.DataFrame | None:
    p = store.symbol_dir(data_dir, symbol) / f"{name}.parquet"
    return pd.read_parquet(p).sort_values("ts").drop_duplicates("ts") if p.exists() else None


def load_panel(data_dir: Path, symbols: list[str], signal_only: list[str], start_ms: int | None,
               end_ms: int) -> Panel:
    """Всё, что известно строго до end_ms. start_ms — начало сетки (None — с первой свечи)."""
    frames, sym = {}, {}
    for s in symbols + signal_only:
        k = _read(data_dir, s, "kline_last_15m")
        if k is None:
            raise FileNotFoundError(f"Нет свечей {s} в {data_dir}")
        # свеча используется, только если она закрылась до end_ms
        k = k[(k["ts"] + BASE_MS <= end_ms) & ((k["ts"] >= start_ms) if start_ms is not None else True)]
        frames[s] = k
        f = _read(data_dir, s, "funding")
        f = f[f["ts"] < end_ms] if f is not None else pd.DataFrame({"ts": [], "rate": []})
        aux = {}
        for name, (col, lag) in AUX.items():
            a = _read(data_dir, s, name)
            if a is not None:
                a = a[a["ts"] + lag <= end_ms]
                aux[name] = Series(a["ts"].to_numpy("int64") + lag, a[col].to_numpy("float64"))
        sym[s] = SymbolData(s, Instrument.from_files(data_dir, s), f["ts"].to_numpy("int64"),
                            f["rate"].to_numpy("float64"), aux)
    lo = min(int(f["ts"].min()) for f in frames.values() if len(f))
    hi = max(int(f["ts"].max()) for f in frames.values() if len(f))
    grid = np.arange(lo, hi + BASE_MS, BASE_MS, dtype="int64")
    n_all = len(symbols) + len(signal_only)
    arrs = {c: np.full((n_all, len(grid)), np.nan) for c in ("open", "high", "low", "close", "volume", "turnover")}
    for r, s in enumerate(symbols + signal_only):
        k = frames[s]
        idx = (k["ts"].to_numpy("int64") - lo) // BASE_MS
        for c in arrs:
            arrs[c][r, idx] = k[c].to_numpy("float64")
    fg = None
    fgp = data_dir / "fear_greed.parquet"
    if fgp.exists():
        g = pd.read_parquet(fgp).sort_values("ts")
        g = g[g["ts"] + DAY_MS <= end_ms]
        fg = Series(g["ts"].to_numpy("int64") + DAY_MS, g["value"].to_numpy("float64"))
    return Panel(symbols=list(symbols), signal_only=list(signal_only), ts=grid, sym=sym, fear_greed=fg,
                 end_ms=end_ms, **arrs)


def _universe(data_dir: Path) -> tuple[list[str], list[str]]:
    u = store.read_json(data_dir / "universe.json")
    if u is None:
        raise FileNotFoundError(f"Нет {data_dir / 'universe.json'}")
    return list(u["tradable"]), list(u["signal_only"])


def load_basket_dev(cfg: Config, data_dir: Path = BASKET_DIR) -> Panel:
    """Период разработки: всё строго до начала отложенных данных."""
    tradable, sig = _universe(data_dir)
    return load_panel(data_dir, tradable, sig, None, to_ms(cfg.research.holdout_start))


def load_basket_holdout(cfg: Config, confirm: str, reason: str, data_dir: Path = BASKET_DIR) -> Panel:
    """Вся история до конца отложенного периода — только для финальной проверки."""
    if confirm != HOLDOUT_CONFIRM:
        raise PermissionError("Отложенные данные открываются только для финальной проверки")
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ACCESS_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\tBASKET\t{reason}\n")
    tradable, sig = _universe(data_dir)
    return load_panel(data_dir, tradable, sig, None, to_ms(cfg.research.holdout_end))


@dataclass
class PanelBars:
    """Свечи таймфрейма tf для всех монет на общей сетке: массивы (n_symbols, n_bars).

    Свеча монеты — NaN, если у неё нет хотя бы одной 15m-свечи внутри (до листинга).
    m_start/m_end — строки 15m-сетки панели, из которых собрана свеча.
    """
    tf: str
    panel: Panel
    ts: np.ndarray
    close_ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    turnover: np.ndarray
    m_start: np.ndarray
    m_end: np.ndarray
    _aux: dict = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.ts)

    @property
    def symbols(self) -> list[str]:
        return self.panel.all_symbols

    @property
    def n_trade(self) -> int:
        return len(self.panel.symbols)

    def aux(self, name: str) -> np.ndarray:
        """Вспомогательный ряд, известный на закрытии каждой свечи: (n_symbols, n_bars).

        name: premium_1h | oi_1h | lsr_1h | funding (последняя выплаченная ставка).
        """
        if name not in self._aux:
            out = np.full((len(self.symbols), len(self)), np.nan)
            for r, s in enumerate(self.symbols):
                sd = self.panel.sym[s]
                ser = Series(sd.funding_ts, sd.funding_rate) if name == "funding" else sd.aux.get(name)
                if ser is not None and len(ser.known_ts):
                    out[r] = ser.asof(self.close_ts)
            self._aux[name] = out
        return self._aux[name]

    def fear_greed(self) -> np.ndarray:
        fg = self.panel.fear_greed
        return fg.asof(self.close_ts) if fg is not None else np.full(len(self), np.nan)

    def row_bars(self, r: int):
        """Свечи одной монеты в формате Bars (для однотипных стратегий)."""
        from bot.data.bars import Bars
        return Bars(self.tf, self.ts, self.close_ts, self.open[r], self.high[r], self.low[r], self.close[r],
                    self.volume[r], self.m_start, self.m_end)


def aggregate_panel(panel: Panel, tf: str) -> PanelBars:
    from bot.data.bars import MINUTE_MS, TF_MINUTES
    step = TF_MINUTES[tf] * MINUTE_MS
    if step % BASE_MS:
        raise ValueError(f"Таймфрейм {tf} не кратен 15 минутам")
    n = step // BASE_MS
    ts = panel.ts
    first = int(np.searchsorted(ts, -(-int(ts[0]) // step) * step)) if len(ts) else 0
    nb = (len(ts) - first) // n
    sl = slice(first, first + nb * n)
    shape = (panel.close.shape[0], nb, n)
    cube = {c: getattr(panel, c)[:, sl].reshape(shape) for c in ("open", "high", "low", "close", "volume", "turnover")}
    m_start = first + np.arange(nb, dtype="int64") * n
    bts = ts[m_start] if nb else np.array([], dtype="int64")
    with np.errstate(invalid="ignore"):
        # max/min/sum распространяют NaN: свеча с пропуском внутри становится NaN целиком
        return PanelBars(tf=tf, panel=panel, ts=bts, close_ts=bts + step,
                         open=np.where(np.isnan(cube["close"]).any(axis=2), np.nan, cube["open"][:, :, 0]),
                         high=cube["high"].max(axis=2), low=cube["low"].min(axis=2),
                         close=np.where(np.isnan(cube["open"]).any(axis=2), np.nan, cube["close"][:, :, -1]),
                         volume=cube["volume"].sum(axis=2), turnover=cube["turnover"].sum(axis=2),
                         m_start=m_start, m_end=m_start + n)
