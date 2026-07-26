"""Decompress account_registry.db.gz for Render build (keeps repo under GitHub 100 MB limit)."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DB_PATH = MODELS_DIR / "account_registry.db"
GZ_PATH = MODELS_DIR / "account_registry.db.gz"


def main() -> None:
  if DB_PATH.exists():
    print(f"Registry DB already present: {DB_PATH}")
    return
  if not GZ_PATH.exists():
    raise FileNotFoundError(
      f"Missing {GZ_PATH.name}. Run: python scripts/compress_registry_db.py"
    )
  MODELS_DIR.mkdir(parents=True, exist_ok=True)
  with gzip.open(GZ_PATH, "rb") as src, DB_PATH.open("wb") as dst:
    shutil.copyfileobj(src, dst)
  print(f"Decompressed {GZ_PATH.name} -> {DB_PATH.name} ({DB_PATH.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
  main()
