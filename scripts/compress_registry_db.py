"""Compress account_registry.db for Git (GitHub file limit is 100 MB)."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DB_PATH = MODELS_DIR / "account_registry.db"
GZ_PATH = MODELS_DIR / "account_registry.db.gz"


def main() -> None:
  if not DB_PATH.exists():
    raise FileNotFoundError(f"Missing {DB_PATH}. Run: python scripts/export_deploy_artifacts.py")
  with DB_PATH.open("rb") as src, gzip.open(GZ_PATH, "wb", compresslevel=9) as dst:
    shutil.copyfileobj(src, dst)
  raw_mb = DB_PATH.stat().st_size / (1024 * 1024)
  gz_mb = GZ_PATH.stat().st_size / (1024 * 1024)
  print(f"Compressed {DB_PATH.name} ({raw_mb:.1f} MB) -> {GZ_PATH.name} ({gz_mb:.1f} MB)")
  if gz_mb > 100:
    print("WARNING: compressed file still exceeds 100 MB; use Git LFS.")


if __name__ == "__main__":
  main()
