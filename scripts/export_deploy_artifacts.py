"""Export SQLite-backed deploy artefacts for Render / low-memory hosting."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_utils import export_deploy_artifacts


def main() -> None:
  deploy_path, db_path = export_deploy_artifacts()
  deploy_mb = deploy_path.stat().st_size / (1024 * 1024)
  db_mb = db_path.stat().st_size / (1024 * 1024)
  print(f"Deploy pipeline: {deploy_path} ({deploy_mb:.2f} MB)")
  print(f"Registry DB:     {db_path} ({db_mb:.2f} MB)")
  gz_path = db_path.with_suffix(db_path.suffix + ".gz")
  if gz_path.exists():
    print(f"Registry archive: {gz_path} ({gz_path.stat().st_size / (1024 * 1024):.1f} MB)")
  print("Commit fraud_detection_pipeline_deploy.joblib and account_registry.db.gz before deploying.")


if __name__ == "__main__":
  main()
