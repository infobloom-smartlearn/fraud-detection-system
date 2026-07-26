"""Shared dataset loading, feature engineering, and 50/50 balancing."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "Cifer-Fraud-Detection-Dataset-AF"
PROCESSED_TUNED_DIR = PROJECT_ROOT / "data" / "processed" / "tuned"
TARGET_COLUMN = "isFraud"
RANDOM_STATE = 42
MIN_PART_BYTES = 120 * 1024 * 1024


class ThresholdClassifier:
  """Apply a tuned probability threshold for classification."""

  def __init__(self, estimator, threshold: float = 0.5):
    self.estimator = estimator
    self.threshold = threshold

  def fit(self, X, y):
    self.estimator.fit(X, y)
    return self

  def predict_proba(self, X):
    return self.estimator.predict_proba(X)

  def predict(self, X):
    proba = self.predict_proba(X)[:, 1]
    return (proba >= self.threshold).astype(int)


MIN_REGISTRY_DB_BYTES = 10 * 1024 * 1024


def is_valid_registry_db(db_path: str | Path) -> bool:
  """Return True when the SQLite registry exists and has the expected schema."""
  path = Path(db_path)
  if not path.is_file() or path.stat().st_size < MIN_REGISTRY_DB_BYTES:
    return False

  import sqlite3

  conn = sqlite3.connect(path)
  try:
    row = conn.execute(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='legitimate_origins'"
    ).fetchone()
    return row is not None
  finally:
    conn.close()


class AccountLegitimacyRegistry:
  """Accounts observed in legitimate transactions across the training corpus."""

  def __init__(self, db_path: str | Path | None = None):
    self.legitimate_origins: set[str] = set()
    self.legitimate_destinations: set[str] = set()
    self._db_path = str(db_path) if db_path else None
    self._conn = None

  def fit_from_dataframe(self, df: pd.DataFrame) -> AccountLegitimacyRegistry:
    legit = df.loc[df[TARGET_COLUMN] == 0]
    self.legitimate_origins = set(legit["nameOrig"].astype(str))
    self.legitimate_destinations = set(legit["nameDest"].astype(str))
    return self

  def uses_sqlite(self) -> bool:
    return bool(self._db_path)

  def ensure_ready(self) -> None:
    """Verify the SQLite registry is present and usable (for deploy startup checks)."""
    if not self._db_path:
      return
    if not is_valid_registry_db(self._db_path):
      raise RuntimeError(
        f"Account registry database missing or invalid at {self._db_path}. "
        "Run `python scripts/decompress_registry_db.py` during build."
      )

  def _connection(self):
    if not self._db_path:
      return None
    self.ensure_ready()
    if self._conn is None:
      import sqlite3

      self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
    return self._conn

  def _is_legit_origin(self, name: str) -> int:
    conn = self._connection()
    if conn is None:
      return int(str(name) in self.legitimate_origins)
    row = conn.execute(
      "SELECT 1 FROM legitimate_origins WHERE name = ? LIMIT 1",
      (str(name),),
    ).fetchone()
    return int(row is not None)

  def _is_legit_dest(self, name: str) -> int:
    conn = self._connection()
    if conn is None:
      return int(str(name) in self.legitimate_destinations)
    row = conn.execute(
      "SELECT 1 FROM legitimate_destinations WHERE name = ? LIMIT 1",
      (str(name),),
    ).fetchone()
    return int(row is not None)

  def save_sqlite(self, path: str | Path) -> Path:
    import sqlite3

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
      path.unlink()

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
      "CREATE TABLE legitimate_origins (name TEXT PRIMARY KEY NOT NULL)"
    )
    conn.execute(
      "CREATE TABLE legitimate_destinations (name TEXT PRIMARY KEY NOT NULL)"
    )

    def _insert_many(table: str, values: set[str]) -> None:
      batch: list[tuple[str]] = []
      for value in values:
        batch.append((str(value),))
        if len(batch) >= 5000:
          conn.executemany(f"INSERT OR IGNORE INTO {table} VALUES (?)", batch)
          batch.clear()
      if batch:
        conn.executemany(f"INSERT OR IGNORE INTO {table} VALUES (?)", batch)

    _insert_many("legitimate_origins", self.legitimate_origins)
    _insert_many("legitimate_destinations", self.legitimate_destinations)
    conn.commit()
    conn.close()
    return path

  @classmethod
  def from_sqlite(cls, db_path: str | Path) -> AccountLegitimacyRegistry:
    return cls(db_path=db_path)

  def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if self._db_path:
      out["orig_in_global_legit"] = out["nameOrig"].astype(str).map(self._is_legit_origin)
      out["dest_in_global_legit"] = out["nameDest"].astype(str).map(self._is_legit_dest)
    else:
      out["orig_in_global_legit"] = out["nameOrig"].astype(str).isin(self.legitimate_origins).astype(int)
      out["dest_in_global_legit"] = out["nameDest"].astype(str).isin(self.legitimate_destinations).astype(int)
    return out

  def lookup(self, name_orig: str, name_dest: str) -> dict[str, int]:
    return {
      "orig_in_global_legit": self._is_legit_origin(name_orig),
      "dest_in_global_legit": self._is_legit_dest(name_dest),
    }


class FraudDetectionPipeline:
  """End-to-end inference pipeline for the Flask app."""

  def __init__(
    self,
    model,
    scaler: StandardScaler,
    account_registry: AccountLegitimacyRegistry,
    feature_columns: list[str],
    model_name: str,
    threshold: float = 0.5,
    use_scaled_inputs: bool = False,
  ):
    self.model = model
    self.scaler = scaler
    self.account_registry = account_registry
    self.feature_columns = feature_columns
    self.model_name = model_name
    self.threshold = threshold
    self.use_scaled_inputs = use_scaled_inputs
    self.input_columns = [
      "step",
      "type",
      "amount",
      "oldbalanceOrg",
      "newbalanceOrig",
      "nameOrig",
      "nameDest",
      "oldbalanceDest",
      "newbalanceDest",
      "isFlaggedFraud",
    ]

  def prepare_frame(self, records: list[dict] | dict) -> pd.DataFrame:
    if isinstance(records, dict):
      records = [records]
    frame = pd.DataFrame(records)
    for col in self.input_columns:
      if col not in frame.columns:
        frame[col] = 0 if col != "type" else "PAYMENT"
    if TARGET_COLUMN not in frame.columns:
      frame[TARGET_COLUMN] = 0
    frame = clean_dataframe(frame)
    frame = engineer_features(frame)
    frame = self.account_registry.transform(frame)
    return frame

  def _feature_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
    features = frame[self.feature_columns].copy()
    if self.use_scaled_inputs:
      features[SCALER_COLUMNS] = self.scaler.transform(features[SCALER_COLUMNS])
    return features

  def predict_proba(self, records: list[dict] | dict) -> np.ndarray:
    frame = self.prepare_frame(records)
    features = self._feature_matrix(frame)
    estimator = self.model.estimator if hasattr(self.model, "estimator") else self.model
    return estimator.predict_proba(features)[:, 1]

  def predict(self, records: list[dict] | dict) -> list[dict]:
    frame = self.prepare_frame(records)
    if frame.empty:
      raise ValueError("No valid transaction rows after validation (check step, amount, and balances).")
    features = self._feature_matrix(frame)
    if hasattr(self.model, "predict"):
      if hasattr(self.model, "estimator"):
        proba = self.model.estimator.predict_proba(features)[:, 1]
        labels = (proba >= self.threshold).astype(int)
      else:
        labels = self.model.predict(features)
        proba = self.predict_proba(records)
    else:
      proba = self.predict_proba(records)
      labels = (proba >= self.threshold).astype(int)

    if isinstance(records, dict):
      records = [records]
    results = []
    for record, label, score in zip(records, labels, proba):
      risk = "High" if score >= 0.75 else "Medium" if score >= 0.4 else "Low"
      results.append(
        {
          "transaction_type": record.get("type", "PAYMENT"),
          "amount": record.get("amount", 0),
          "is_fraud": int(label),
          "fraud_probability": round(float(score), 4),
          "risk_level": risk,
          "model": self.model_name,
        }
      )
    return results


RAW_COLUMNS = [
  "step",
  "type",
  "amount",
  "nameOrig",
  "oldbalanceOrg",
  "newbalanceOrig",
  "nameDest",
  "oldbalanceDest",
  "newbalanceDest",
  "isFraud",
  "isFlaggedFraud",
]

BASE_FEATURE_COLUMNS = [
  "step",
  "amount",
  "oldbalanceOrg",
  "newbalanceOrig",
  "oldbalanceDest",
  "newbalanceDest",
  "balance_change_orig",
  "balance_change_dest",
  "orig_balance_error",
  "dest_balance_error",
  "abs_orig_balance_error",
  "abs_dest_balance_error",
  "has_orig_balance_error",
  "has_dest_balance_error",
  "amount_log",
  "amount_ratio_orig",
  "amount_ratio_dest",
  "orig_account_emptied",
  "dest_account_emptied",
  "hour_of_day",
  "day_of_simulation",
  "type_CASH_IN",
  "type_CASH_OUT",
  "type_DEBIT",
  "type_PAYMENT",
  "type_TRANSFER",
  "isFlaggedFraud",
  "transfer_with_orig_error",
  "cashout_with_orig_error",
]

ACCOUNT_FEATURE_COLUMNS = [
  "orig_in_global_legit",
  "dest_in_global_legit",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ACCOUNT_FEATURE_COLUMNS

SCALER_COLUMNS = [
  "step",
  "amount",
  "oldbalanceOrg",
  "newbalanceOrig",
  "oldbalanceDest",
  "newbalanceDest",
  "balance_change_orig",
  "balance_change_dest",
  "orig_balance_error",
  "dest_balance_error",
  "abs_orig_balance_error",
  "abs_dest_balance_error",
  "amount_log",
  "amount_ratio_orig",
  "amount_ratio_dest",
  "hour_of_day",
  "day_of_simulation",
]


def list_complete_parts(data_dir: Path = RAW_DATA_DIR) -> list[Path]:
  parts = sorted(data_dir.glob("Cifer-Fraud-Detection-Dataset-AF-part-*-14.csv"))
  return [p for p in parts if p.stat().st_size >= MIN_PART_BYTES]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df.columns = df.columns.str.strip()

  numeric_cols = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
  ]
  for col in numeric_cols:
    if col in df.columns:
      df[col] = pd.to_numeric(df[col], errors="coerce")

  if "type" in df.columns:
    df["type"] = df["type"].astype(str).str.strip().str.upper()
  if "nameOrig" in df.columns:
    df["nameOrig"] = df["nameOrig"].astype(str).str.strip()
  if "nameDest" in df.columns:
    df["nameDest"] = df["nameDest"].astype(str).str.strip()

  df = df.drop_duplicates()
  if "amount" in df.columns:
    df = df.loc[df["amount"] >= 0]
  if "step" in df.columns:
    df = df.loc[(df["step"] >= 0) & (df["step"] <= 744)]
  if TARGET_COLUMN in df.columns:
    df = df.loc[df[TARGET_COLUMN].isin([0, 1])]
  return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()

  df["balance_change_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
  df["balance_change_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
  df["orig_balance_error"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
  df["dest_balance_error"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
  df["abs_orig_balance_error"] = df["orig_balance_error"].abs()
  df["abs_dest_balance_error"] = df["dest_balance_error"].abs()
  df["has_orig_balance_error"] = (df["abs_orig_balance_error"] > 1.0).astype(int)
  df["has_dest_balance_error"] = (df["abs_dest_balance_error"] > 1.0).astype(int)
  df["amount_log"] = np.log1p(df["amount"])
  df["amount_ratio_orig"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
  df["amount_ratio_dest"] = df["amount"] / (df["oldbalanceDest"] + 1.0)
  df["orig_account_emptied"] = (
    (df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)
  ).astype(int)
  df["dest_account_emptied"] = (
    (df["oldbalanceDest"] > 0) & (df["newbalanceDest"] == 0)
  ).astype(int)
  df["hour_of_day"] = df["step"] % 24
  df["day_of_simulation"] = df["step"] // 24

  type_dummies = pd.get_dummies(df["type"], prefix="type", dtype=int)
  expected_types = [
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
  ]
  for col in expected_types:
    if col not in type_dummies.columns:
      type_dummies[col] = 0
  df = pd.concat([df, type_dummies[expected_types]], axis=1)

  df["transfer_with_orig_error"] = (
    (df["type"] == "TRANSFER") & (df["has_orig_balance_error"] == 1)
  ).astype(int)
  df["cashout_with_orig_error"] = (
    (df["type"] == "CASH_OUT") & (df["has_orig_balance_error"] == 1)
  ).astype(int)

  return df


def build_account_registry(parts: list[Path] | None = None) -> AccountLegitimacyRegistry:
  parts = parts or list_complete_parts()
  legit_origins: set[str] = set()
  legit_destinations: set[str] = set()

  for path in parts:
    chunk = pd.read_csv(path, usecols=["nameOrig", "nameDest", TARGET_COLUMN])
    chunk.columns = chunk.columns.str.strip()
    legit = chunk.loc[chunk[TARGET_COLUMN] == 0]
    legit_origins.update(legit["nameOrig"].astype(str))
    legit_destinations.update(legit["nameDest"].astype(str))

  registry = AccountLegitimacyRegistry()
  registry.legitimate_origins = legit_origins
  registry.legitimate_destinations = legit_destinations
  return registry


def build_balanced_dataset_from_parts(
  parts: list[Path] | None = None,
  legitimate_ratio: int = 1,
  account_registry: AccountLegitimacyRegistry | None = None,
) -> tuple[pd.DataFrame, dict, AccountLegitimacyRegistry]:
  parts = parts or list_complete_parts()
  account_registry = account_registry or AccountLegitimacyRegistry()
  legit_origins: set[str] = set(account_registry.legitimate_origins)
  legit_destinations: set[str] = set(account_registry.legitimate_destinations)

  part_frames: list[pd.DataFrame] = []
  total_legit_available = 0
  n_fraud_total = 0

  for path in parts:
    chunk = pd.read_csv(path, usecols=RAW_COLUMNS)
    chunk = clean_dataframe(chunk)
    part_frames.append(chunk)
    total_legit_available += int((chunk[TARGET_COLUMN] == 0).sum())
    n_fraud_total += int((chunk[TARGET_COLUMN] == 1).sum())

  n_legit_target = min(total_legit_available, n_fraud_total * legitimate_ratio)

  fraud_frames: list[pd.DataFrame] = []
  legit_sample_frames: list[pd.DataFrame] = []

  for chunk in part_frames:
    legit_chunk = chunk.loc[chunk[TARGET_COLUMN] == 0]
    legit_n = len(legit_chunk)
    legit_origins.update(legit_chunk["nameOrig"].astype(str))
    legit_destinations.update(legit_chunk["nameDest"].astype(str))
    fraud_frames.append(chunk.loc[chunk[TARGET_COLUMN] == 1])

    if legit_n == 0 or n_legit_target == 0:
      continue

    n_sample = int(round(n_legit_target * legit_n / total_legit_available))
    n_sample = max(0, min(n_sample, legit_n))
    if n_sample > 0:
      legit_sample_frames.append(
        legit_chunk.sample(
          n=n_sample,
          random_state=RANDOM_STATE,
        )
      )

  account_registry.legitimate_origins = legit_origins
  account_registry.legitimate_destinations = legit_destinations

  fraud = pd.concat(fraud_frames, ignore_index=True)
  legitimate_sample = pd.concat(legit_sample_frames, ignore_index=True)

  if len(legitimate_sample) > n_legit_target:
    legitimate_sample = legitimate_sample.sample(
      n=n_legit_target,
      random_state=RANDOM_STATE,
    )

  combined = pd.concat([legitimate_sample, fraud], ignore_index=True)
  combined = combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
  combined = engineer_features(combined)
  combined = account_registry.transform(combined)

  report = {
    "parts_used": [p.name for p in parts],
    "fraud_rows": len(fraud),
    "legitimate_rows_selected": len(legitimate_sample),
    "total_rows": len(combined),
    "balance": "50/50" if legitimate_ratio == 1 else f"{legitimate_ratio}:1",
    "legitimate_origin_accounts": len(account_registry.legitimate_origins),
    "legitimate_destination_accounts": len(account_registry.legitimate_destinations),
  }
  return combined, report, account_registry


def prepare_train_test_splits(
  df: pd.DataFrame,
  test_size: float = 0.2,
) -> tuple[
  pd.DataFrame,
  pd.Series,
  pd.DataFrame,
  pd.Series,
  pd.DataFrame,
  pd.Series,
  StandardScaler,
]:
  X = df[FEATURE_COLUMNS]
  y = df[TARGET_COLUMN]

  X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=RANDOM_STATE,
    stratify=y,
  )

  scaler = StandardScaler()
  X_train_scaled = X_train.copy()
  X_test_scaled = X_test.copy()
  X_train_scaled[SCALER_COLUMNS] = scaler.fit_transform(X_train[SCALER_COLUMNS])
  X_test_scaled[SCALER_COLUMNS] = scaler.transform(X_test[SCALER_COLUMNS])

  return X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, scaler


def save_tuned_dataset(
  df: pd.DataFrame,
  X_train_scaled: pd.DataFrame,
  y_train: pd.Series,
  X_test_scaled: pd.DataFrame,
  y_test: pd.Series,
  scaler: StandardScaler,
  account_registry: AccountLegitimacyRegistry,
  report: dict,
) -> Path:
  PROCESSED_TUNED_DIR.mkdir(parents=True, exist_ok=True)
  df.to_csv(PROCESSED_TUNED_DIR / "combined_50_50_enhanced.csv", index=False)
  X_train_scaled.assign(**{TARGET_COLUMN: y_train}).to_csv(
    PROCESSED_TUNED_DIR / "train_scaled.csv",
    index=False,
  )
  X_test_scaled.assign(**{TARGET_COLUMN: y_test}).to_csv(
    PROCESSED_TUNED_DIR / "test_scaled.csv",
    index=False,
  )

  joblib.dump(scaler, PROCESSED_TUNED_DIR / "scaler.joblib")
  joblib.dump(account_registry, PROCESSED_TUNED_DIR / "account_registry.joblib")

  meta_path = PROCESSED_TUNED_DIR / "dataset_report.json"
  with meta_path.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
  return PROCESSED_TUNED_DIR


MODELS_DIR = PROJECT_ROOT / "models"
DEPLOY_PIPELINE_NAME = "fraud_detection_pipeline_deploy.joblib"
DEPLOY_REGISTRY_NAME = "account_registry.db"
DEPLOY_REGISTRY_REL = DEPLOY_REGISTRY_NAME


def export_deploy_artifacts(
  pipeline_path: Path | None = None,
  models_dir: Path | None = None,
) -> tuple[Path, Path]:
  """Build SQLite-backed deploy artefacts for low-memory hosting (e.g. Render free tier)."""
  models_dir = models_dir or MODELS_DIR
  db_path = models_dir / DEPLOY_REGISTRY_NAME
  deploy_path = models_dir / DEPLOY_PIPELINE_NAME
  results_dir = PROJECT_ROOT / "outputs" / "model_results"

  if not db_path.exists():
    registry_source = models_dir / "account_registry.joblib"
    if not registry_source.exists():
      pipeline_path = pipeline_path or models_dir / "fraud_detection_pipeline.joblib"
      if not pipeline_path.exists():
        raise FileNotFoundError(
          "No registry source found. Train models first or provide account_registry.joblib."
        )
      pipeline = joblib.load(pipeline_path)
      pipeline.account_registry.save_sqlite(db_path)
    else:
      registry = joblib.load(registry_source)
      registry.save_sqlite(db_path)
      del registry

  report_path = results_dir / "training_report_tuned.json"
  if not report_path.exists():
    report_path = results_dir / "training_report.json"
  if not report_path.exists():
    pipeline_path = pipeline_path or models_dir / "fraud_detection_pipeline.joblib"
    if pipeline_path.exists():
      pipeline = joblib.load(pipeline_path)
      deploy_registry = AccountLegitimacyRegistry(db_path=DEPLOY_REGISTRY_REL)
      pipeline.account_registry = deploy_registry
      joblib.dump(pipeline, deploy_path)
      return deploy_path, db_path
    raise FileNotFoundError("training_report_tuned.json not found.")

  report = json.loads(report_path.read_text(encoding="utf-8"))
  best_name = report["best_model"]
  best_metrics = next(m for m in report["metrics"] if m["model"] == best_name)
  threshold = float(best_metrics.get("threshold", 0.5))

  model_path = models_dir / f"{best_name}_tuned.joblib"
  if not model_path.exists():
    model_path = models_dir / "best_model.joblib"
  best_model = joblib.load(model_path)

  scaler_path = PROCESSED_TUNED_DIR / "scaler.joblib"
  if scaler_path.exists():
    scaler = joblib.load(scaler_path)
  else:
    pipeline_path = pipeline_path or models_dir / "fraud_detection_pipeline.joblib"
    if pipeline_path.exists():
      scaler = joblib.load(pipeline_path).scaler
    else:
      scaler = StandardScaler()

  deploy_registry = AccountLegitimacyRegistry(db_path=DEPLOY_REGISTRY_REL)
  pipeline = FraudDetectionPipeline(
    model=best_model,
    scaler=scaler,
    account_registry=deploy_registry,
    feature_columns=FEATURE_COLUMNS,
    model_name=best_name,
    threshold=threshold,
    use_scaled_inputs=False,
  )
  joblib.dump(pipeline, deploy_path)

  gz_path = db_path.with_suffix(db_path.suffix + ".gz")
  try:
    import gzip
    import shutil

    with db_path.open("rb") as src, gzip.open(gz_path, "wb", compresslevel=9) as dst:
      shutil.copyfileobj(src, dst)
  except OSError as exc:
    print(f"Note: could not compress registry DB: {exc}", flush=True)
  else:
    print(
      f"Registry archive: {gz_path.name} ({gz_path.stat().st_size / (1024 * 1024):.1f} MB)",
      flush=True,
    )

  return deploy_path, db_path
