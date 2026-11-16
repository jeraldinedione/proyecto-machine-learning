from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gait import download

if __name__ == "__main__":
    for slug, ruta in download.fetch_all().items():
        print(f"{slug}: {ruta}")
