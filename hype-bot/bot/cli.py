"""Командная строка бота.

    python -m bot.cli data download          # загрузить/дозагрузить историю
    python -m bot.cli data check             # отчёт о качестве → reports/data_quality.md
    python -m bot.cli data fees              # реальные комиссии (нужен ключ в .env)
    python -m bot.cli data pack              # упаковать данные в upload/hype-data.zip
    python -m bot.cli data unpack ФАЙЛ.zip   # распаковать архив с проверкой sha256
"""
from __future__ import annotations

import argparse
import logging
import sys

from bot.config import PROJECT_ROOT, load_config
from bot.data import downloader, store, transfer
from bot.data.report import build_report

import pandas as pd

log = logging.getLogger("bot")


def _symbols(cfg, args) -> list[str]:
    if args.symbols:
        return [s.upper() for s in args.symbols]
    syms = [cfg.symbol]
    if not args.no_extra:
        syms += cfg.data.extra_symbols
    return syms


def cmd_download(cfg, args) -> int:
    session = downloader.make_public_session(cfg.exchange.domain, cfg.exchange.tld,
                                             cfg.exchange.http_timeout_s)
    call = downloader.Caller(pause_s=cfg.data.request_pause_s)
    data_dir = cfg.data_dir()
    now_ms = downloader.server_time_ms(session, call)
    for sym in _symbols(cfg, args):
        history = None if sym == cfg.symbol else cfg.data.extra_history_days
        kinds = (store.KIND_LAST,) if (sym != cfg.symbol or args.no_mark) else (store.KIND_LAST, store.KIND_MARK)
        summary = downloader.download_symbol(session, call, data_dir, sym, history_days=history,
                                             kinds=kinds, now_ms=now_ms)
        log.info("Готово: %s", summary)
    manifest = downloader.write_manifest(data_dir, PROJECT_ROOT / "data" / "manifest.json",
                                         [cfg.symbol] + cfg.data.extra_symbols)
    log.info("Манифест: %d файлов; запросов к API: %d", len(manifest["files"]), call.requests_made)
    return 0


def cmd_check(cfg, args) -> int:
    text = build_report(cfg.data_dir(), _symbols(cfg, args))
    out = PROJECT_ROOT / "reports" / "data_quality.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Отчёт записан: {out}")
    return 0


def cmd_fees(cfg, args) -> int:
    from pybit.unified_trading import HTTP

    from bot.envfile import get_keys, load_env

    load_env()
    key, secret = get_keys("live")
    session = HTTP(testnet=False, domain=cfg.exchange.domain, tld=cfg.exchange.tld,
                   api_key=key, api_secret=secret, max_retries=1)
    call = downloader.Caller(pause_s=0.3)
    rates = downloader.fetch_fee_rate(session, call, cfg.symbol)
    store.write_json(store.json_path(cfg.data_dir(), cfg.symbol, "fee_rate"), {"list": rates})
    for r in rates:
        print(f"{r.get('symbol')}: maker {r.get('makerFeeRate')}, taker {r.get('takerFeeRate')}")
    return 0


def cmd_basket(cfg, args) -> int:
    """Отбор корзины, загрузка всех рядов по каждой монете и упаковка в upload/basket-data.zip."""
    import json

    from bot.data import basket

    b = cfg.basket
    session = downloader.make_public_session(cfg.exchange.domain, cfg.exchange.tld, cfg.exchange.http_timeout_s)
    call = downloader.Caller(pause_s=cfg.data.request_pause_s)
    data_dir = PROJECT_ROOT / "data" / "basket"
    data_dir.mkdir(parents=True, exist_ok=True)
    now_ms = downloader.server_time_ms(session, call)
    instruments = basket.all_linear_instruments(session, call)
    tickers = call(session.get_tickers, category="linear")["result"]["list"]
    rule = {"listed_before": b.listed_before, "min_turnover_24h": b.min_turnover_24h,
            "max_min_order_usdt": b.max_min_order_usdt, "n_max": b.n_max, "include": b.include}
    chosen, table = basket.select_universe(instruments, tickers, rule)
    symbols = chosen + [s for s in b.signal_only if s not in chosen]
    table.to_csv(data_dir / "universe_table.csv", index=False)
    store.write_json(data_dir / "universe.json", {"generated_at": basket.utc_now_iso(), "rule": rule,
                                                  "tradable": chosen, "signal_only": b.signal_only})
    log.info("Корзина (%d): %s; только сигналы: %s", len(chosen), chosen, b.signal_only)
    start_ms = int(pd.Timestamp(b.history_start, tz="UTC").value // 1_000_000)

    def one(sym: str) -> dict:
        # у каждого потока своё соединение: requests.Session не рассчитан на общий доступ
        s = downloader.make_public_session(cfg.exchange.domain, cfg.exchange.tld, cfg.exchange.http_timeout_s)
        c = downloader.Caller(pause_s=cfg.data.request_pause_s)
        return basket.download_symbol_basket(s, c, data_dir, sym, start_ms, now_ms, b.kline_interval)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    failed = []
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {ex.submit(one, sym): sym for sym in symbols}
        for k, fut in enumerate(as_completed(futures), 1):
            sym = futures[fut]
            try:
                log.info("[%d/%d] %s готово: %s", k, len(symbols), sym, json.dumps(fut.result(), ensure_ascii=False))
            except Exception as e:
                failed.append(sym)
                log.error("[%d/%d] %s: ошибка %s", k, len(symbols), sym, e)
    if failed:
        log.error("Не загружены: %s. Запустите ещё раз — загрузка продолжится с места остановки.", failed)
        return 2
    try:
        n = basket.download_fear_greed(data_dir / "fear_greed.parquet")
        log.info("Индекс страха и жадности: %d дней", n)
    except Exception as e:  # внешний источник необязателен
        log.warning("Индекс страха и жадности не скачан: %s", e)
    out = PROJECT_ROOT / "upload" / "basket-data.zip"
    manifest = transfer.pack(data_dir, symbols, out, extra_files=["universe.json", "universe_table.csv",
                                                                  "fear_greed.parquet"])
    print(f"Архив готов: {out} ({out.stat().st_size / 1e6:.1f} МБ, файлов: {len(manifest['files'])})")
    return 0


def cmd_pack(cfg, args) -> int:
    out = PROJECT_ROOT / "upload" / "hype-data.zip"
    manifest = transfer.pack(cfg.data_dir(), [cfg.symbol] + cfg.data.extra_symbols, out)
    size_mb = out.stat().st_size / 1e6
    print(f"Архив готов: {out} ({size_mb:.1f} МБ, файлов: {len(manifest['files'])})")
    return 0


def cmd_unpack(cfg, args) -> int:
    manifest = transfer.unpack(args.zip, cfg.data_dir(), PROJECT_ROOT / "data" / "manifest.json")
    print(f"Распаковано и проверено файлов: {len(manifest['files'])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "bot.log", encoding="utf-8")],
    )
    p = argparse.ArgumentParser(prog="bot")
    p.add_argument("--config", help="путь к YAML-конфигу")
    sub = p.add_subparsers(dest="group", required=True)
    data = sub.add_parser("data", help="исторические данные")
    dsub = data.add_subparsers(dest="cmd", required=True)
    for name in ("download", "check"):
        sp = dsub.add_parser(name)
        sp.add_argument("--symbols", nargs="*")
        sp.add_argument("--no-extra", action="store_true", help="только основной инструмент")
        sp.add_argument("--no-mark", action="store_true", help="без свечей mark-цены")
    dsub.add_parser("fees")
    dsub.add_parser("pack")
    bp = dsub.add_parser("basket")
    bp.add_argument("--threads", type=int, default=3)
    up = dsub.add_parser("unpack")
    up.add_argument("zip")
    args = p.parse_args(argv)
    if sys.version_info < (3, 11):
        log.error("Нужен Python 3.11 или новее, у вас %s", sys.version.split()[0])
        return 2
    cfg = load_config(args.config)
    handlers = {"download": cmd_download, "check": cmd_check, "fees": cmd_fees,
                "pack": cmd_pack, "unpack": cmd_unpack, "basket": cmd_basket}
    try:
        return handlers[args.cmd](cfg, args)
    except (downloader.DownloadError, transfer.TransferError) as e:
        log.error("%s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
