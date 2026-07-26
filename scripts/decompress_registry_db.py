"""Decompress account_registry.db.gz for Render build (keeps repo under GitHub 100 MB limit)."""

from __future__ import annotations

import gzip
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_utils import is_valid_registry_db

MODELS_DIR = PROJECT_ROOT / "models"
DB_PATH = MODELS_DIR / "account_registry.db"
GZ_PATH = MODELS_DIR / "account_registry.db.gz"


def main() -> None:
  if DB_PATH.exists() and is_valid_registry_db(DB_PATH):
    print(f"Registry DB already present: {DB_PATH}")
    return

  if DB_PATH.exists():
    print(f"Removing invalid registry DB at {DB_PATH}")
    DB_PATH.unlink()

  if not GZ_PATH.exists():
    raise FileNotFoundError(
      f"Missing {GZ_PATH.name}. Run: python scripts/compress_registry_db.py"
    )

  MODELS_DIR.mkdir(parents=True, exist_ok=True)
  with gzip.open(GZ_PATH, "rb") as src, DB_PATH.open("wb") as dst:
    shutil.copyfileobj(src, dst)

  if not is_valid_registry_db(DB_PATH):
    DB_PATH.unlink(missing_ok=True)
    raise RuntimeError(
      f"Decompressed registry at {DB_PATH} is invalid. "
      f"Re-create {GZ_PATH.name} with scripts/compress_registry_db.py."
    )

  print(
    f"Decompressed {GZ_PATH.name} -> {DB_PATH.name} "
    f"({DB_PATH.stat().st_size / 1024 / 1024:.1f} MB)"
  )


if __name__ == "__main__":
  main()
