"""Перенос скачанных данных одним архивом (компьютер пользователя → среда исследования).

Архив содержит manifest.json с sha256 каждого файла; при распаковке все суммы
сверяются, так что исследование идёт ровно на тех данных, что скачаны у вас.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

from bot.data import store
from bot.data.downloader import write_manifest

_MEMBER = re.compile(r"^raw/([A-Z0-9]{2,30})/([a-z0-9_]{1,40}\.(?:parquet|json))$")
_ROOT_MEMBER = re.compile(r"^raw/([a-z0-9_]{1,40}\.(?:parquet|json|csv))$")


class TransferError(RuntimeError):
    pass


def pack(data_dir: Path, symbols: list[str], out_zip: Path, extra_files: list[str] = ()) -> dict:
    """extra_files — общие файлы в корне data_dir (состав корзины, внешние ряды)."""
    data_dir = Path(data_dir)
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_zip.parent / "manifest.json"
    manifest = write_manifest(data_dir, manifest_path, symbols)
    for name in extra_files:
        f = data_dir / name
        if f.exists():
            manifest["files"][name] = {"sha256": store.file_sha256(f), "bytes": f.stat().st_size}
    store.write_json(manifest_path, manifest)
    if not manifest["files"]:
        raise TransferError(f"В {data_dir} нет данных для упаковки")
    tmp = Path(str(out_zip) + ".tmp")
    with zipfile.ZipFile(tmp, "w") as z:
        z.write(manifest_path, "manifest.json", compress_type=zipfile.ZIP_DEFLATED)
        for rel in manifest["files"]:
            src = data_dir / rel
            # parquet уже сжат (zstd), повторное сжатие не даёт выигрыша
            ctype = zipfile.ZIP_STORED if src.suffix == ".parquet" else zipfile.ZIP_DEFLATED
            z.write(src, f"raw/{rel}", compress_type=ctype)
    tmp.replace(out_zip)
    manifest_path.unlink()
    return manifest


def unpack(zip_path: Path, data_dir: Path, manifest_out: Path) -> dict:
    """Распаковывает архив в data_dir, проверяя имена и sha256. Возвращает манифест."""
    import json

    data_dir = Path(data_dir)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if "manifest.json" not in names:
            raise TransferError("В архиве нет manifest.json")
        manifest = json.loads(z.read("manifest.json").decode("utf-8"))
        expected = manifest.get("files", {})
        for name in names:
            if name == "manifest.json":
                continue
            m, mr = _MEMBER.match(name), _ROOT_MEMBER.match(name)
            if not (m or mr):
                raise TransferError(f"Неожиданный файл в архиве: {name!r}")
            rel = f"{m.group(1)}/{m.group(2)}" if m else mr.group(1)
            if rel not in expected:
                raise TransferError(f"Файла {name} нет в манифесте")
        missing = [rel for rel in expected if f"raw/{rel}" not in names]
        if missing:
            raise TransferError(f"В архиве не хватает файлов: {missing}")
        for rel, info in expected.items():
            dst = data_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(dst) + ".tmp")
            tmp.write_bytes(z.read(f"raw/{rel}"))
            digest = store.file_sha256(tmp)
            if digest != info["sha256"]:
                tmp.unlink()
                raise TransferError(f"Контрольная сумма не совпала: {rel}")
            tmp.replace(dst)
    store.write_json(Path(manifest_out), manifest)
    return manifest
