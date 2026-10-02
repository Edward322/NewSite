"""Ограничители риска: дневной лимит, максимальная просадка, серия убытков, стоп-флаги.

Один и тот же класс используется в бэктесте и в боевом режиме; состояние
сериализуется (to_dict/from_dict) и в бою хранится в SQLite, чтобы переживать
перезапуск.

- Дневной лимит: капитал упал на daily_loss_limit от капитала на начало дня
  (UTC) → новые позиции не открываются до следующего дня UTC.
- Максимальная просадка: капитал ≤ пик × (1 − max_drawdown) → остановка
  навсегда (флаг halted); закрыть позиции и отменить ордера должен вызывающий.
- Серия: loss_streak_pause_trades убыточных сделок подряд → пауза на
  loss_streak_pause_hours, после паузы счётчик с нуля (None — правило выключено).
- Ступени по просадке (drawdown_steps): при просадке от пика ≥ порога риск новой сделки
  умножается на множитель ступени (risk_multiplier); ступень снимается, когда просадка
  снова меньше порога.
- Блокировки (обрыв связи, устаревшие данные, расхождение с биржей) ставит
  боевой движок; пока хоть одна активна, новые позиции не открываются.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

from bot.config import RiskCfg

HOUR_MS = 3_600_000
EPS = 1e-9  # допуск сравнения долей: просадка ровно на пороге тоже срабатывает


def utc_day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def next_utc_midnight(ms: int) -> int:
    d = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date() + timedelta(days=1)
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


@dataclass
class RiskState:
    peak_equity: float
    day: str
    day_start_equity: float
    loss_streak: int = 0
    pause_until_ms: int = 0
    pause_reason: str = ""
    halted: bool = False
    halt_reason: str = ""
    blockers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RiskState":
        return cls(**d)


@dataclass(frozen=True)
class GuardEvent:
    ts: int
    kind: str      # day_start | daily_limit | streak_pause | max_drawdown | halt | blocker_on | blocker_off
    detail: str


class RiskGuard:
    def __init__(self, cfg: RiskCfg, equity: float, now_ms: int, state: RiskState | None = None):
        self.cfg = cfg
        self.state = state or RiskState(peak_equity=equity, day=utc_day(now_ms), day_start_equity=equity)

    # --- обновления ---
    def update_equity(self, now_ms: int, equity: float) -> list[GuardEvent]:
        s, ev = self.state, []
        day = utc_day(now_ms)
        if day != s.day:
            s.day, s.day_start_equity = day, equity
            ev.append(GuardEvent(now_ms, "day_start", f"капитал на начало дня {equity:.4f}"))
        if equity > s.peak_equity:
            s.peak_equity = equity
        if not s.halted and self._loss_frac(equity, s.peak_equity) >= self.cfg.max_drawdown - EPS:
            ev.append(self.halt(now_ms, f"просадка от пика ≥ {self.cfg.max_drawdown:.0%} "
                                        f"(капитал {equity:.4f}, пик {s.peak_equity:.4f})", kind="max_drawdown"))
        if (not s.halted and s.pause_until_ms <= now_ms
                and self._loss_frac(equity, s.day_start_equity) >= self.cfg.daily_loss_limit - EPS):
            s.pause_until_ms = next_utc_midnight(now_ms)
            s.pause_reason = "дневной лимит убытка"
            ev.append(GuardEvent(now_ms, "daily_limit",
                                 f"убыток за день ≥ {self.cfg.daily_loss_limit:.0%}, пауза до конца дня UTC"))
        return ev

    def on_trade_closed(self, now_ms: int, net_pnl: float) -> list[GuardEvent]:
        s = self.state
        if net_pnl >= 0:
            s.loss_streak = 0
            return []
        s.loss_streak += 1
        if self.cfg.loss_streak_pause_trades is not None and s.loss_streak >= self.cfg.loss_streak_pause_trades:
            s.loss_streak = 0
            until = now_ms + int(self.cfg.loss_streak_pause_hours * HOUR_MS)
            if until > s.pause_until_ms:
                s.pause_until_ms, s.pause_reason = until, "серия убыточных сделок"
            return [GuardEvent(now_ms, "streak_pause",
                               f"{self.cfg.loss_streak_pause_trades} убыточных сделок подряд, "
                               f"пауза {self.cfg.loss_streak_pause_hours:g} ч")]
        return []

    def halt(self, now_ms: int, reason: str, kind: str = "halt") -> GuardEvent:
        self.state.halted, self.state.halt_reason = True, reason
        return GuardEvent(now_ms, kind, reason)

    def set_blocker(self, now_ms: int, key: str, reason: str | None) -> GuardEvent | None:
        b = self.state.blockers
        if reason and b.get(key) != reason:
            b[key] = reason
            return GuardEvent(now_ms, "blocker_on", f"{key}: {reason}")
        if not reason and key in b:
            del b[key]
            return GuardEvent(now_ms, "blocker_off", key)
        return None

    # --- запросы ---
    @staticmethod
    def _loss_frac(equity: float, base: float) -> float:
        return 1 - equity / base if base > 0 else 1.0

    def drawdown(self, equity: float) -> float:
        return max(0.0, self._loss_frac(equity, self.state.peak_equity))

    def risk_multiplier(self, equity: float) -> float:
        """Множитель риска новой сделки по текущей просадке от пика (1 — без снижения)."""
        dd, mult = self.drawdown(equity), 1.0
        for thr, factor in sorted(self.cfg.drawdown_steps):
            if dd >= thr - EPS:
                mult = factor
        return mult

    def drawdown_floor(self) -> float:
        return self.state.peak_equity * (1 - self.cfg.max_drawdown)

    def can_open(self, now_ms: int) -> tuple[bool, str]:
        s = self.state
        if s.halted:
            return False, f"остановлен: {s.halt_reason}"
        if s.blockers:
            return False, "блокировка: " + "; ".join(f"{k}: {v}" for k, v in s.blockers.items())
        if now_ms < s.pause_until_ms:
            return False, f"пауза ({s.pause_reason})"
        return True, ""
