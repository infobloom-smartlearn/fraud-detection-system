"""
Preprocessing and exploratory data analysis for Cifer Fraud Detection Dataset (part 1).

Stages:
  1. Data loading and cleaning
  2. Missing value handling
  3. Feature engineering
  4. Feature scaling
  5. Class imbalance treatment (undersampling + SMOTE)
  6. Exploratory data analysis with visualisations
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "Cifer-Fraud-Detection-Dataset-AF"
    / "Cifer-Fraud-Detection-Dataset-AF-part-1-14.csv"
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "part-1"
EDA_DIR = PROJECT_ROOT / "outputs" / "eda" / "part-1"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
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
    "amount_log",
    "hour_of_day",
    "day_of_simulation",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "isFlaggedFraud",
]

TARGET_COLUMN = "isFraud"
LEGITIMATE_TO_FRAUD_RATIO = 1  # 1:1 => 50/50 legitimate vs fraud


def combine_legitimate_and_fraud(
  df: pd.DataFrame,
  legitimate_ratio: int = LEGITIMATE_TO_FRAUD_RATIO,
) -> tuple[pd.DataFrame, dict]:
  """
  Explicitly combine legitimate (isFraud=0) and fraud (isFraud=1) rows.

  Part 1 already contains both classes, but fraud is <0.2% of rows so it can
  look like a legitimate-only dataset. This step keeps every fraud transaction
  and samples the same number of legitimate rows for a 50/50 split.
  """
  legitimate = df.loc[df[TARGET_COLUMN] == 0].copy()
  fraud = df.loc[df[TARGET_COLUMN] == 1].copy()

  n_fraud = len(fraud)
  n_legit_target = min(len(legitimate), n_fraud * legitimate_ratio)

  legitimate_sample = legitimate.sample(
    n=n_legit_target,
    random_state=RANDOM_STATE,
  )

  combined = pd.concat([legitimate_sample, fraud], ignore_index=True)
  combined["class_label"] = np.where(
    combined[TARGET_COLUMN] == 1,
    "Fraud",
    "Legitimate",
  )
  combined = combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

  report = {
    "legitimate_rows_available": len(legitimate),
    "fraud_rows_available": n_fraud,
    "legitimate_rows_selected": n_legit_target,
    "fraud_rows_selected": n_fraud,
    "combined_rows": len(combined),
    "combined_class_distribution": combined[TARGET_COLUMN].value_counts().to_dict(),
    "combined_class_labels": combined["class_label"].value_counts().to_dict(),
    "legitimate_to_fraud_ratio": legitimate_ratio,
    "balance": "50/50" if legitimate_ratio == 1 else f"{legitimate_ratio}:1",
  }
  return combined, report


def load_raw_data(path: Path) -> pd.DataFrame:
  print(f"Loading data from {path.name} ...")
  df = pd.read_csv(path)
  print(f"  Loaded {len(df):,} rows x {df.shape[1]} columns")
  return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
  """Remove duplicates, invalid rows, and enforce sensible dtypes."""
  report: dict = {"initial_rows": len(df)}

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
    df[col] = pd.to_numeric(df[col], errors="coerce")

  df["type"] = df["type"].astype(str).str.strip().str.upper()

  report["missing_before"] = df.isnull().sum().to_dict()
  report["duplicates_removed"] = int(df.duplicated().sum())
  df = df.drop_duplicates()

  invalid_amount = df["amount"] < 0
  report["invalid_amount_removed"] = int(invalid_amount.sum())
  df = df.loc[~invalid_amount]

  invalid_step = (df["step"] < 0) | (df["step"] > 744)
  report["invalid_step_removed"] = int(invalid_step.sum())
  df = df.loc[~invalid_step]

  invalid_target = ~df["isFraud"].isin([0, 1])
  report["invalid_target_removed"] = int(invalid_target.sum())
  df = df.loc[~invalid_target]

  report["rows_after_cleaning"] = len(df)
  report["missing_after"] = df.isnull().sum().to_dict()
  return df.reset_index(drop=True), report


def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
  """Impute or drop remaining missing values."""
  report: dict = {}
  df = df.copy()

  missing = df.isnull().sum()
  report["missing_counts"] = missing[missing > 0].to_dict()

  if report["missing_counts"]:
    balance_cols = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    for col in balance_cols:
      if col in df.columns and df[col].isnull().any():
        df[col] = df[col].fillna(0)

    if df["type"].isnull().any():
      df = df.dropna(subset=["type"])

    if df["isFraud"].isnull().any():
      df = df.dropna(subset=["isFraud"])

  report["rows_after_imputation"] = len(df)
  report["remaining_missing"] = int(df.isnull().sum().sum())
  return df.reset_index(drop=True), report


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
  """Create derived features inspired by PaySim fraud-detection literature."""
  df = df.copy()

  df["balance_change_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
  df["balance_change_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
  df["orig_balance_error"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
  df["dest_balance_error"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
  df["amount_log"] = np.log1p(df["amount"])
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

  return df


def scale_features(
  X_train: pd.DataFrame,
  X_test: pd.DataFrame,
  feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
  scaler = StandardScaler()
  X_train_scaled = X_train.copy()
  X_test_scaled = X_test.copy()

  X_train_scaled[feature_columns] = scaler.fit_transform(X_train[feature_columns])
  X_test_scaled[feature_columns] = scaler.transform(X_test[feature_columns])
  return X_train_scaled, X_test_scaled, scaler


def address_class_imbalance(
  X_train: pd.DataFrame,
  y_train: pd.Series,
  strategy: str = "hybrid",
) -> tuple[pd.DataFrame, pd.Series, dict]:
  """
  Balance training data. Skips resampling when input is already ~50/50;
  otherwise applies undersampling + SMOTEENN.
  """
  report: dict = {
    "strategy": strategy,
    "before": y_train.value_counts().to_dict(),
  }

  minority_rate = y_train.mean()
  if 0.45 <= minority_rate <= 0.55:
    report["strategy"] = "none (already 50/50)"
    report["after_smoteenn"] = y_train.value_counts().to_dict()
    report["balanced_rows"] = len(y_train)
    return X_train.copy(), y_train.copy(), report

  rus = RandomUnderSampler(
    sampling_strategy=0.2,
    random_state=RANDOM_STATE,
  )
  X_under, y_under = rus.fit_resample(X_train, y_train)

  smote_enn = SMOTEENN(random_state=RANDOM_STATE)
  X_balanced, y_balanced = smote_enn.fit_resample(X_under, y_under)

  report["after_undersample"] = pd.Series(y_under).value_counts().to_dict()
  report["after_smoteenn"] = pd.Series(y_balanced).value_counts().to_dict()
  report["balanced_rows"] = len(y_balanced)

  X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
  y_balanced = pd.Series(y_balanced, name=TARGET_COLUMN)
  return X_balanced, y_balanced, report


def run_eda(df: pd.DataFrame, output_dir: Path) -> dict:
  """Generate summary statistics and visualisations."""
  output_dir.mkdir(parents=True, exist_ok=True)
  sns.set_theme(style="whitegrid", palette="muted")

  summary: dict = {
    "shape": list(df.shape),
    "columns": list(df.columns),
    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    "fraud_rate_pct": round(df[TARGET_COLUMN].mean() * 100, 4),
    "class_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
    "transaction_type_distribution": df["type"].value_counts().to_dict(),
    "numeric_summary": df[
      ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    ].describe().round(2).to_dict(),
  }

  numeric_cols = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_change_orig",
    "balance_change_dest",
  ]

  plt.figure(figsize=(8, 5))
  fraud_counts = df[TARGET_COLUMN].value_counts().sort_index()
  labels = fraud_counts.index.map({0: "Legitimate", 1: "Fraud"})
  sns.barplot(
    x=labels,
    y=fraud_counts.values,
    hue=labels,
    legend=False,
    palette=["#4C78A8", "#E45756"],
  )
  plt.title("Combined Legitimate + Fraud Class Distribution")
  plt.ylabel("Transaction Count")
  for idx, value in enumerate(fraud_counts.values):
    plt.text(idx, value, f"{value:,}", ha="center", va="bottom")
  plt.tight_layout()
  plt.savefig(output_dir / "01_class_distribution.png", dpi=150)
  plt.close()

  if "class_label" in df.columns:
    plt.figure(figsize=(8, 5))
    label_counts = df["class_label"].value_counts()
    sns.barplot(
      x=label_counts.index,
      y=label_counts.values,
      hue=label_counts.index,
      legend=False,
      palette={"Legitimate": "#4C78A8", "Fraud": "#E45756"},
    )
    plt.title("Legitimate vs Fraud (Combined Dataset)")
    plt.ylabel("Transaction Count")
    for idx, value in enumerate(label_counts.values):
      plt.text(idx, value, f"{value:,}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_dir / "01b_combined_class_labels.png", dpi=150)
    plt.close()

  plt.figure(figsize=(9, 5))
  type_counts = df["type"].value_counts()
  sns.barplot(x=type_counts.index, y=type_counts.values, hue=type_counts.index, legend=False)
  plt.title("Transaction Type Distribution")
  plt.ylabel("Count")
  plt.xticks(rotation=20)
  plt.tight_layout()
  plt.savefig(output_dir / "02_transaction_types.png", dpi=150)
  plt.close()

  fraud_by_type = (
    df.groupby("type")[TARGET_COLUMN]
    .agg(["count", "sum", "mean"])
    .rename(columns={"sum": "fraud_count", "mean": "fraud_rate"})
    .sort_values("fraud_rate", ascending=False)
  )
  summary["fraud_by_transaction_type"] = fraud_by_type.round(6).to_dict()

  plt.figure(figsize=(9, 5))
  sns.barplot(
    data=fraud_by_type.reset_index(),
    x="type",
    y="fraud_rate",
    hue="type",
    legend=False,
    palette="rocket",
  )
  plt.title("Fraud Rate by Transaction Type")
  plt.ylabel("Fraud Rate")
  plt.xticks(rotation=20)
  plt.tight_layout()
  plt.savefig(output_dir / "03_fraud_rate_by_type.png", dpi=150)
  plt.close()

  plt.figure(figsize=(10, 6))
  sample = df.sample(n=min(100_000, len(df)), random_state=RANDOM_STATE)
  sns.scatterplot(
    data=sample,
    x="step",
    y="amount",
    hue=TARGET_COLUMN,
    alpha=0.35,
    palette={0: "#4C78A8", 1: "#E45756"},
    s=12,
  )
  plt.yscale("log")
  plt.title("Transaction Amount vs Simulation Step (sampled)")
  plt.tight_layout()
  plt.savefig(output_dir / "04_amount_vs_step.png", dpi=150)
  plt.close()

  plt.figure(figsize=(10, 5))
  fraud_sample = df.loc[df[TARGET_COLUMN] == 1, "amount"]
  legit_sample = df.loc[df[TARGET_COLUMN] == 0, "amount"].sample(
    n=min(50_000, (df[TARGET_COLUMN] == 0).sum()),
    random_state=RANDOM_STATE,
  )
  plot_df = pd.DataFrame(
    {
      "amount": pd.concat([legit_sample, fraud_sample]),
      "label": ["Legitimate"] * len(legit_sample) + ["Fraud"] * len(fraud_sample),
    }
  )
  sns.boxplot(data=plot_df, x="label", y="amount", hue="label", legend=False, palette="Set2")
  plt.yscale("log")
  plt.title("Transaction Amount by Fraud Label")
  plt.tight_layout()
  plt.savefig(output_dir / "05_amount_by_fraud_label.png", dpi=150)
  plt.close()

  corr = df[numeric_cols + [TARGET_COLUMN]].corr()
  summary["correlation_with_target"] = (
    corr[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(key=abs, ascending=False).round(4).to_dict()
  )

  plt.figure(figsize=(10, 8))
  sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
  plt.title("Correlation Matrix (Numeric Features)")
  plt.tight_layout()
  plt.savefig(output_dir / "06_correlation_heatmap.png", dpi=150)
  plt.close()

  hourly = (
    df.groupby("hour_of_day")[TARGET_COLUMN]
    .agg(["count", "sum", "mean"])
    .rename(columns={"sum": "fraud_count", "mean": "fraud_rate"})
  )
  summary["fraud_by_hour"] = hourly.round(6).to_dict()

  plt.figure(figsize=(10, 4))
  sns.lineplot(data=hourly.reset_index(), x="hour_of_day", y="fraud_rate", marker="o")
  plt.title("Fraud Rate by Hour of Day")
  plt.ylabel("Fraud Rate")
  plt.tight_layout()
  plt.savefig(output_dir / "07_fraud_rate_by_hour.png", dpi=150)
  plt.close()

  return summary


def main() -> None:
  PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
  EDA_DIR.mkdir(parents=True, exist_ok=True)

  raw_df = load_raw_data(DATA_PATH)
  cleaned_df, cleaning_report = clean_data(raw_df)
  imputed_df, missing_report = handle_missing_values(cleaned_df)

  combined_df, combine_report = combine_legitimate_and_fraud(imputed_df)
  combined_df.to_csv(PROCESSED_DIR / "combined_legitimate_fraud.csv", index=False)
  print(
    f"  Combined dataset: {combine_report['combined_rows']:,} rows "
    f"({combine_report['combined_class_labels']})"
  )

  featured_df = engineer_features(combined_df)
  eda_summary = run_eda(featured_df, EDA_DIR)

  X = featured_df[FEATURE_COLUMNS]
  y = featured_df[TARGET_COLUMN]

  X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
  )

  X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, FEATURE_COLUMNS)
  X_balanced, y_balanced, imbalance_report = address_class_imbalance(
    X_train_scaled,
    y_train,
  )

  featured_df.to_csv(PROCESSED_DIR / "featured_full.csv", index=False)
  X_train_scaled.assign(**{TARGET_COLUMN: y_train}).to_csv(
    PROCESSED_DIR / "train_scaled.csv",
    index=False,
  )
  X_test_scaled.assign(**{TARGET_COLUMN: y_test}).to_csv(
    PROCESSED_DIR / "test_scaled.csv",
    index=False,
  )
  X_balanced.assign(**{TARGET_COLUMN: y_balanced}).to_csv(
    PROCESSED_DIR / "train_balanced.csv",
    index=False,
  )

  joblib.dump(scaler, PROCESSED_DIR / "scaler.joblib")

  pipeline_report = {
    "dataset": DATA_PATH.name,
    "combine_legitimate_fraud": combine_report,
    "cleaning": cleaning_report,
    "missing_values": missing_report,
    "feature_columns": FEATURE_COLUMNS,
    "train_test_split": {
      "train_rows": len(X_train),
      "test_rows": len(X_test),
      "train_fraud_rate_pct": round(y_train.mean() * 100, 4),
      "test_fraud_rate_pct": round(y_test.mean() * 100, 4),
    },
    "class_imbalance": imbalance_report,
    "eda_summary": eda_summary,
  }

  report_path = PROCESSED_DIR / "preprocessing_eda_report.json"
  with report_path.open("w", encoding="utf-8") as f:
    json.dump(pipeline_report, f, indent=2)

  print("\nPreprocessing and EDA complete.")
  print(f"  Processed data : {PROCESSED_DIR}")
  print(f"  EDA plots      : {EDA_DIR}")
  print(f"  Report         : {report_path}")
  print(f"  Fraud rate     : {eda_summary['fraud_rate_pct']}%")
  print(f"  Balanced train : {len(X_balanced):,} rows")


if __name__ == "__main__":
  main()
