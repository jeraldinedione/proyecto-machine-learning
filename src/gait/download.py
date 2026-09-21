from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from .config import GAITNDD, GAITPDB, RAW_DIR, DatasetSpec

DATASETS = {spec.slug: spec for spec in (GAITNDD, GAITPDB)}


def _report(block_index: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = min(block_index * block_size, total_size)
    percent = 100 * downloaded / total_size
    print(f"\r  {percent:5.1f}%  {downloaded / 1e6:7.1f} / {total_size / 1e6:.1f} MB", end="")


def fetch(spec: DatasetSpec, force: bool = False) -> Path:
    if spec.path.exists() and not force:
        return spec.path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / f"{spec.folder}.zip"

    print(f"Descargando {spec.slug} desde PhysioNet")
    urllib.request.urlretrieve(spec.url, archive, reporthook=_report)
    print()

    if spec.path.exists():
        shutil.rmtree(spec.path)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(RAW_DIR)
    archive.unlink()

    return spec.path


def fetch_all(force: bool = False) -> dict[str, Path]:
    return {slug: fetch(spec, force=force) for slug, spec in DATASETS.items()}


def require(spec: DatasetSpec) -> Path:
    if not spec.path.exists():
        raise FileNotFoundError(
            f"No se encontro {spec.path}. Ejecuta gait.download.fetch(gait.config.{spec.slug.upper()})"
        )
    return spec.path
