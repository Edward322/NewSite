"""Самопроверка на компьютере пользователя: всё ли готово к работе бота с настоящей биржей.

    python -m bot.selfcheck --mode demo --quick   быстро, без ордеров (запускает установщик)
    python -m bot.selfcheck --mode demo           полностью: + пробный минимальный ордер со стопом на ДЕМО
                                                  и «сухой» прогон движка на реальных данных (без ордеров)

Вывод — на экран и в файл reports/selfcheck/selfcheck_<режим>_<время>.txt (его нужно прислать
разработчику). Секретов в выводе нет: ключ показывается только первыми 4 символами.
Пробный ордер выставляется только в режиме demo: ~5 USDT виртуальных денег, сразу закрывается.
"""
from __future__ import annotations

import argparse
import math
import platform
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from bot.config import PROJECT_ROOT, load_bot_config
from bot.envfile import get_keys, load_env

TEST_SYMBOL = "XRPUSDT"


class Report:
    def __init__(self):
        self.lines: list[str] = []
        self.results: list[tuple[str, str]] = []

    def p(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def check(self, name: str, fn, *a, **kw):
        self.p(f"\n== {name}")
        t0 = time.time()
        try:
            res = fn(*a, **kw)
            status = "WARN" if res == "warn" else "OK"
        except Exception as e:
            status = "FAIL"
            self.p(f"   ОШИБКА: {type(e).__name__}: {e}")
            self.p("   " + traceback.format_exc().strip().splitlines()[-1][:300])
            res = None
        self.results.append((name, status))
        self.p(f"   → {status} ({time.time() - t0:.1f} с)")
        return res


def mask(s: str) -> str:
    return (s[:4] + "…") if s else "—"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "live"], default="demo")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    mode = a.mode
    R = Report()
    now = datetime.now(timezone.utc)
    R.p(f"САМОПРОВЕРКА бота ({mode}{', быстрая' if a.quick else ', полная'}), {now:%Y-%m-%d %H:%M} UTC")
    load_env()
    cfg = load_bot_config()
    st: dict = {}

    # ------------------------------------------------------------ окружение
    def env():
        import pybit
        R.p(f"   Python {sys.version.split()[0]}, {platform.platform()}, pybit {getattr(pybit, 'VERSION', '?')}")
        R.p(f"   Папка бота: {PROJECT_ROOT}")
        if not str(PROJECT_ROOT).isascii() or " " in str(PROJECT_ROOT):
            R.p("   Внимание: в пути есть русские буквы или пробелы — лучше перенести папку, например в C:\\hype")
            return "warn"
        import shutil
        free = shutil.disk_usage(PROJECT_ROOT).free / 2**30
        R.p(f"   Свободно на диске: {free:.1f} ГБ")
    R.check("Окружение", env)

    def config():
        r, pc, s = cfg.risk, cfg.portfolio, cfg.strategy
        R.p(f"   Риск {r.risk_per_trade:.2%} на сделку, капитал {r.starting_equity_usdt:g} USDT, дневной лимит "
            f"{r.daily_loss_limit:.0%}, просадка {r.drawdown_steps} → остановка {r.max_drawdown:.0%}")
        R.p(f"   Сумма риска ≤ {pc.max_open_risk:.0%}, с учётом корреляции ≤ {pc.corr_cap}, позиций ≤ {pc.max_positions}")
        R.p(f"   Стратегия {s.timeframe} {s.params}; монет {len(s.tradable)}; live.enabled={cfg.live.enabled}")
        if mode == "live" and not cfg.live.enabled:
            raise RuntimeError("live.enabled = false — реальная торговля выключена")
    R.check("Конфиг", config)

    def keys():
        k, sec = get_keys(mode)
        st["keys"] = (k, sec)
        R.p(f"   Ключ: {mask(k)} (секрет не показывается)")
    R.check("Ключи API в .env", keys)
    if "keys" not in st:
        return finish(R, mode)

    from bot.engine.control import make_session
    from bot.exchange.client import BybitClient, ExchangeError
    client = BybitClient(make_session(cfg, mode, *st["keys"]))
    syms = cfg.strategy.tradable + cfg.strategy.signal_only

    def clock():
        srv = client.server_time_ms()
        off = (srv - time.time() * 1000) / 1000
        R.p(f"   Часы компьютера отличаются от биржи на {off:+.2f} с")
        if abs(off) > 2:
            R.p("   Включите синхронизацию времени: Параметры → Время и язык → «Синхронизировать сейчас»")
            return "warn"
    R.check("Связь с биржей и часы", clock)

    def instruments():
        bad = []
        for s in syms:
            i = client.instrument(s)
            lot, pf = i["lotSizeFilter"], i["priceFilter"]
            ok = i.get("status") == "Trading"
            if not ok:
                bad.append(s)
            R.p(f"   {s:13s} {str(i.get('status')):8s} шаг цены {pf['tickSize']:>10s} шаг объёма {lot['qtyStep']:>6s} "
                f"мин. ордер {lot.get('minNotionalValue', '?')} USDT, плечо до {i['leverageFilter']['maxLeverage']}")
        if bad:
            raise RuntimeError(f"не торгуются: {bad}")
    R.check("Инструменты корзины", instruments)

    def spreads():
        worse = []
        for s in cfg.strategy.tradable:
            t = client.ticker(s)
            half = (t.ask - t.bid) / 2 / ((t.ask + t.bid) / 2) if t.ask and t.bid else float("nan")
            flag = "" if half <= cfg.costs.slippage else "  > заложенного"
            if flag:
                worse.append(s)
            R.p(f"   {s:13s} бид {t.bid:<12g} аск {t.ask:<12g} половина спреда {half * 100:.4f}%{flag}")
        R.p(f"   Заложено в бэктест: {cfg.costs.slippage * 100:.3f}% на рыночную сделку")
        return "warn" if worse else None
    R.check("Спреды сейчас", spreads)

    def account():
        info = client.account_info()
        R.p(f"   Режим маржи: {info.get('marginMode')}, статус единого аккаунта: {info.get('unifiedMarginStatus')}")
        w = client.wallet()
        R.p(f"   Баланс USDT: капитал {w.equity:.2f}, кошелёк {w.wallet_balance:.2f}, свободно {w.available:.2f}")
        pos = client.positions()
        R.p(f"   Открытых позиций: {len(pos)}" + "".join(f"\n     {p.symbol} {p.side:+d} {p.size:g} стоп {p.stop_loss}"
                                                          for p in pos))
        oo = client.open_orders()
        R.p(f"   Открытых ордеров (включая стопы): {len(oo)}")
        st["wallet"] = w
        if mode == "live" and w.equity < cfg.risk.starting_equity_usdt * 0.9:
            R.p(f"   На счёте меньше {cfg.risk.starting_equity_usdt} USDT")
            return "warn"
    R.check("Счёт", account)

    def key_perms():
        res = client._get("get_api_key_information")
        perms = res.get("permissions") or {}
        ro = res.get("readOnly")
        ips = res.get("ips") or []
        R.p(f"   Только чтение: {'да' if ro else 'нет'}; IP: {', '.join(ips) if ips else '—'}; истекает: "
            f"{res.get('expiredAt') or '—'}")
        for k, v in perms.items():
            if v:
                R.p(f"   {k}: {', '.join(v)}")
        flat = [x for v in perms.values() for x in (v or [])]
        if "Withdraw" in flat:
            raise RuntimeError("у ключа есть право ВЫВОДА средств — создайте ключ без него")
        if ro:
            raise RuntimeError("ключ только для чтения — боту нужны права «Ордера» и «Позиции»")
        if mode == "live" and (not ips or ips == ["*"]):
            raise RuntimeError("реальный ключ должен быть привязан к вашему IP")
    R.check("Права ключа", key_perms)

    def fees():
        worse = []
        for s in cfg.strategy.tradable[:3]:
            mk, tk = client.fee_rate(s)
            R.p(f"   {s}: мейкер {mk * 100:.4f}%, тейкер {tk * 100:.4f}%")
            if tk > cfg.costs.taker_fee + 1e-9:
                worse.append(s)
        R.p(f"   Заложено: тейкер {cfg.costs.taker_fee * 100:.4f}%")
        return "warn" if worse else None
    R.check("Комиссии", fees)

    def data():
        from bot.engine.data import LiveData
        with tempfile.TemporaryDirectory() as d:
            ld = LiveData(client, Path(d), cfg.strategy.tradable[:2], [])
            end = client.server_time_ms()
            n = ld.update(end - 2 * 86_400_000, end)
            last = max(ld.last_ts(s) or 0 for s in ld.symbols)
            R.p(f"   Скачано свечей 15m: {n}; последняя закрытая: "
                f"{datetime.fromtimestamp(last / 1000, tz=timezone.utc):%Y-%m-%d %H:%M} UTC")
            if n < 180 or end - last > 31 * 60_000:
                raise RuntimeError("данные неполные")
    R.check("Загрузка свечей", data)

    def ws():
        from bot.engine.stream import BybitStream
        s = BybitStream(syms, mode, *st["keys"], domain=cfg.exchange.domain, tld=cfg.exchange.tld)
        s.start()
        time.sleep(20)
        age = s.public_age_s()
        R.p(f"   Публичный поток: последнее сообщение {age:.1f} с назад" if age is not None else "   нет сообщений")
        R.p(f"   Приватный поток: {'подключён' if s.priv is not None else 'НЕ подключён'}")
        ok = s._last_public is not None and s.priv is not None
        s.stop()
        if not ok:
            raise RuntimeError("WebSocket не работает: бот не будет открывать позиции (ws_required)")
    R.check("WebSocket", ws)

    if not a.quick and mode == "demo":
        R.check("Пробный ордер со стопом (демо)", order_test, R, client, cfg, ExchangeError)
    if not a.quick:
        R.check("Сухой прогон движка на реальных данных (без ордеров)", dry_run, R, client, cfg, mode)
    return finish(R, mode)


def order_test(R: Report, client, cfg, ExchangeError):
    from bot.market import Instrument
    s = TEST_SYMBOL
    inst = Instrument.from_api(client.instrument(s), client.risk_limits(s))
    changed = client.ensure_isolated_one_way()
    R.p(f"   Изолированная маржа / одна позиция: {'изменено: ' + ', '.join(changed) if changed else 'уже так'}")
    if any(p.symbol == s for p in client.positions()):
        raise RuntimeError(f"по {s} уже есть позиция — тест пропущен")
    px = client.ticker(s).last
    step = float(inst.qty_step)
    qty = math.ceil(max(inst.min_notional * 1.1 / px, inst.min_qty) / step) * step
    sl = inst.round_price(px * 0.95, "down")
    client.set_leverage(s, 2)
    link = f"hbtest{int(time.time()):x}"
    t0 = int(time.time() * 1000) - 5000
    opened = False
    try:
        oid = client.place_market(s, 1, inst.fmt_qty(qty), link, stop_loss=inst.fmt_price(sl))
        R.p(f"   1. Вход {s} {inst.fmt_qty(qty)} по рынку со стопом {sl} одним запросом: orderId {oid[:8]}…")
        o = None
        for _ in range(10):
            o = client.order_by_link_id(link, s)
            if o and o.done:
                break
            time.sleep(1)
        R.p(f"   2. Ордер по orderLinkId: статус {o.status if o else 'НЕ НАЙДЕН'}, исполнено {o.filled_qty if o else 0} "
            f"по {o.avg_price if o else 0}, комиссия {o.fee if o else 0}")
        opened = bool(o and o.filled_qty > 0)
        pos = {p.symbol: p for p in client.positions()}.get(s)
        R.p(f"   3. Позиция: {pos.size if pos else 0}, стоп на бирже {pos.stop_loss if pos else None}, "
            f"плечо {pos.leverage if pos else None}, ликвидация {pos.liq_price if pos else None}")
        if not pos or pos.stop_loss is None or abs(pos.stop_loss - sl) > float(inst.tick_size) / 2:
            raise RuntimeError("стоп не выставлен вместе со входом")
        try:
            client.place_market(s, 1, inst.fmt_qty(qty), link, stop_loss=inst.fmt_price(sl))
            R.p("   4. ПОВТОР того же orderLinkId ПРИНЯТ — это опасно (возможен двойной вход)")
            raise RuntimeError("биржа приняла повторный orderLinkId")
        except ExchangeError as e:
            R.p(f"   4. Повтор того же orderLinkId отклонён: код {e.code} «{e.message}» (ожидается 110072)")
        sl2 = inst.round_price(px * 0.96, "down")
        client.set_stop(s, inst.fmt_price(sl2))
        pos = {p.symbol: p for p in client.positions()}.get(s)
        R.p(f"   5. Стоп подтянут до {sl2}: на бирже {pos.stop_loss if pos else None}")
        try:
            client.set_stop(s, inst.fmt_price(inst.round_price(px * 1.05)))
            R.p("   6. Стоп выше цены для лонга ПРИНЯТ — неожиданно")
        except ExchangeError as e:
            R.p(f"   6. Неверный стоп отклонён: код {e.code} «{e.message[:120]}»")
    finally:
        if opened:
            client.place_market(s, -1, inst.fmt_qty(qty), link + "c", reduce_only=True)
            time.sleep(2)
            left = {p.symbol: p for p in client.positions()}.get(s)
            R.p(f"   7. Закрытие reduce-only: позиция {'ЗАКРЫТА' if left is None else f'ОСТАЛАСЬ {left.size}'}")
    try:
        client.place_market(s, -1, inst.fmt_qty(qty), link + "r", reduce_only=True)
        R.p("   8. reduce-only без позиции принят — неожиданно")
    except ExchangeError as e:
        R.p(f"   8. reduce-only без позиции отклонён: код {e.code} «{e.message[:100]}» (ожидается 110017)")
    ex = client.executions(t0, int(time.time() * 1000) + 5000, s)
    for e in ex:
        R.p(f"   исполнение: {e.side:+d} {e.qty:g} по {e.price:g}, комиссия {e.fee:g} "
            f"({e.fee / (e.qty * e.price) * 100:.4f}%), тип {e.exec_type}, orderLinkId {e.link_id or '—'}")
    f = client.funding_paid(s, t0, int(time.time() * 1000))
    R.p(f"   Журнал финансирования: {'доступен' if f is not None else 'НЕДОСТУПЕН (финансирование не будет учтено по сделкам)'}")
    if {p.symbol for p in client.positions()} & {s}:
        raise RuntimeError("тестовая позиция не закрыта — закройте её вручную в приложении")


def dry_run(R: Report, client, cfg, mode):
    from bot.engine.data import LiveData
    from bot.engine.engine import Engine
    from bot.engine.state import StateDB
    from bot.engine.control import RealClock, utc
    c2 = cfg.model_copy(update={"engine": cfg.engine.model_copy(update={"ws_required": False})})
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        db = StateDB(d / "dry.sqlite")
        clock = RealClock()
        clock.sync(client.server_time_ms())
        data = LiveData(client, d / "live", c2.strategy.tradable, c2.strategy.signal_only)
        eng = Engine(c2, client, db, clock, data, mode=mode, base_dir=d)
        R.p("   Загрузка истории для индикаторов (около минуты)...")
        if not eng.start():
            raise RuntimeError("движок не запустился")
        t_close = int(db.get("last_bar_close"))
        out = eng.plan_bar(t_close, update_guard=False)
        if out is None:
            raise RuntimeError("нет плана: данные не готовы")
        plan, syms, _, _ = out
        R.p(f"   Свеча {utc(t_close)}: решений {len(plan.decisions)}, входов было бы {len(plan.entries)}, "
            f"пропусков {len(plan.skips)}")
        for r, _, dd, s in plan.entries:
            R.p(f"     вход {'лонг' if dd.side > 0 else 'шорт'} {syms[r]}: объём {s.qty:g}, стоп {s.stop:g}, "
                f"риск {s.planned_loss:.3f} USDT, плечо {s.leverage:g}")
        for t, sym, why in plan.skips[:10]:
            R.p(f"     пропуск {sym}: {why}")
        R.p(f"   Монет в корзине: {len(syms)}; капитал бота {eng.equity()[0]:.2f} USDT")
        db.close()


def finish(R: Report, mode: str) -> int:
    R.p("\nИТОГ:")
    for name, status in R.results:
        R.p(f"   {status:4s}  {name}")
    fails = [n for n, s in R.results if s == "FAIL"]
    R.p("Всё готово." if not fails else f"Есть ошибки ({len(fails)}): бот запускать нельзя, пока они не исправлены.")
    out = PROJECT_ROOT / "reports" / "selfcheck"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"selfcheck_{mode}_{datetime.now(timezone.utc):%Y%m%d_%H%M}.txt"
    path.write_text("\n".join(R.lines) + "\n", encoding="utf-8")
    print(f"\nОтчёт сохранён: {path}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
