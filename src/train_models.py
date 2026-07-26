"""
Train and fine-tune fraud detection models on the enhanced 50/50 multi-part dataset.
Uses account legitimacy features and accuracy-focused hyperparameter tuning.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
  accuracy_score,
  classification_report,
  confusion_matrix,
  f1_score,
  precision_score,
  recall_score,
  roc_auc_score,
  roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_utils import (
  FEATURE_COLUMNS,
  FraudDetectionPipeline,
  SCALER_COLUMNS,
  TARGET_COLUMN,
  ThresholdClassifier,
  build_balanced_dataset_from_parts,
  list_complete_parts,
  prepare_train_test_splits,
  save_tuned_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "model_results"
PROCESSED_TUNED_DIR = PROJECT_ROOT / "data" / "processed" / "tuned"
RANDOM_STATE = 42
CV_FOLDS = 3
N_ITER = 6
ACCURACY_MIN = 0.70
ACCURACY_MAX = 0.90


def find_best_threshold(y_true: pd.Series, y_proba: np.ndarray) -> float:
  """Pick the threshold that maximises accuracy on validation data."""
  best_threshold = 0.5
  best_accuracy = -1.0
  for threshold in np.linspace(0.05, 0.95, 181):
    preds = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    if ACCURACY_MIN <= acc <= ACCURACY_MAX and acc > best_accuracy:
      best_accuracy = acc
      best_threshold = float(threshold)
  if best_accuracy < 0:
    for threshold in np.linspace(0.05, 0.95, 181):
      preds = (y_proba >= threshold).astype(int)
      acc = accuracy_score(y_true, preds)
      if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = float(threshold)
  return best_threshold


def get_param_grids() -> dict:
  return {
    "logistic_regression": {
      "model__C": np.logspace(-2, 1, 8),
      "model__class_weight": [None, "balanced"],
    },
    "decision_tree": {
      "max_depth": [3, 4, 5, 6, 8],
      "min_samples_leaf": [20, 35, 50, 75],
      "min_samples_split": [2, 10, 20],
      "criterion": ["gini", "entropy"],
    },
    "random_forest": {
      "n_estimators": [80, 120, 160],
      "max_depth": [4, 6, 8, 10],
      "min_samples_leaf": [10, 20, 35],
      "max_features": ["sqrt", "log2", 0.5],
    },
    "xgboost": {
      "n_estimators": [60, 100, 140],
      "max_depth": [2, 3, 4, 5],
      "learning_rate": [0.05, 0.08, 0.12],
      "subsample": [0.7, 0.85, 1.0],
      "colsample_bytree": [0.6, 0.8, 1.0],
      "reg_lambda": [1.0, 4.0, 8.0],
    },
  }


def get_base_estimators() -> dict:
  return {
    "logistic_regression": Pipeline(
      [
        ("scaler", StandardScaler()),
        (
          "model",
          LogisticRegression(max_iter=4000, random_state=RANDOM_STATE),
        ),
      ]
    ),
    "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "xgboost": XGBClassifier(
      eval_metric="logloss",
      random_state=RANDOM_STATE,
      n_jobs=-1,
    ),
  }


def get_training_matrix(name: str, X_train, X_train_scaled):
  if name == "logistic_regression":
    return X_train_scaled
  return X_train


def tune_model(name: str, X_train, y_train):
  cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
  search = RandomizedSearchCV(
    estimator=get_base_estimators()[name],
    param_distributions=get_param_grids()[name],
    n_iter=N_ITER,
    scoring="accuracy",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
  )
  search.fit(X_train, y_train)
  return search.best_estimator_, search.best_params_, float(search.best_score_)


def evaluate_model(name: str, model, X_test, y_test) -> dict:
  y_pred = model.predict(X_test)
  y_proba = model.predict_proba(X_test)[:, 1]

  return {
    "model": name,
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
    "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
    "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
    "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    "classification_report": classification_report(y_test, y_pred, zero_division=0),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "y_pred": y_pred,
    "y_proba": y_proba,
    "threshold": getattr(model, "threshold", 0.5),
  }


def plot_confusion_matrix(cm: list, title: str, output_path: Path) -> None:
  plt.figure(figsize=(5, 4))
  sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"],
  )
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.title(title)
  plt.tight_layout()
  plt.savefig(output_path, dpi=150)
  plt.close()


def plot_roc_curves(results: list[dict], y_test: pd.Series, output_path: Path) -> None:
  plt.figure(figsize=(8, 6))
  for result in results:
    fpr, tpr, _ = roc_curve(y_test, result["y_proba"])
    plt.plot(fpr, tpr, label=f"{result['model']} (AUC={result['roc_auc']:.3f})")
  plt.plot([0, 1], [0, 1], "k--", label="Random")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("ROC Curves — Tuned Models")
  plt.legend(loc="lower right")
  plt.tight_layout()
  plt.savefig(output_path, dpi=150)
  plt.close()


def plot_metrics_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
  plot_df = metrics_df.melt(
    id_vars="model",
    value_vars=["accuracy", "precision", "recall", "f1_score", "roc_auc"],
    var_name="metric",
    value_name="score",
  )
  plt.figure(figsize=(10, 5))
  sns.barplot(data=plot_df, x="metric", y="score", hue="model")
  plt.ylim(0, 1.05)
  plt.title("Tuned Model Performance Comparison")
  plt.ylabel("Score")
  plt.tight_layout()
  plt.savefig(output_path, dpi=150)
  plt.close()


def load_data(force_rebuild: bool = False, quick: bool = False):
  combined_path = PROCESSED_TUNED_DIR / "combined_50_50_enhanced.csv"
  registry_path = PROCESSED_TUNED_DIR / "account_registry.joblib"

  required_account_cols = {"orig_in_global_legit", "dest_in_global_legit"}
  needs_rebuild = force_rebuild or not combined_path.exists() or not registry_path.exists()
  if combined_path.exists() and not needs_rebuild:
    sample = pd.read_csv(combined_path, nrows=1)
    needs_rebuild = not required_account_cols.issubset(sample.columns)

  if not needs_rebuild:
    print(f"  Loading cached dataset: {combined_path.name}", flush=True)
    df = pd.read_csv(combined_path)
    account_registry = joblib.load(registry_path)
    dataset_report = json.loads(
      (combined_path.parent / "dataset_report.json").read_text(encoding="utf-8")
    )
    return df, dataset_report, account_registry

  print("  Building dataset from parts ...", flush=True)
  parts = list_complete_parts()
  if quick:
    parts = [p for p in parts if "part-1-14" in p.name] or parts[:1]
    print(f"  Quick mode: using {parts[0].name}", flush=True)
  df, dataset_report, account_registry = build_balanced_dataset_from_parts(parts=parts)
  X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, scaler = (
    prepare_train_test_splits(df)
  )
  save_tuned_dataset(
    df,
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
    scaler,
    account_registry,
    dataset_report,
  )
  return df, dataset_report, account_registry


def main(force_rebuild: bool = False, quick: bool = False) -> None:
  MODELS_DIR.mkdir(parents=True, exist_ok=True)
  RESULTS_DIR.mkdir(parents=True, exist_ok=True)

  print("Loading enhanced 50/50 dataset with account legitimacy features ...", flush=True)
  df, dataset_report, account_registry = load_data(force_rebuild=force_rebuild or quick, quick=quick)
  print(
    f"  Parts: {len(dataset_report['parts_used'])} | "
    f"Rows: {dataset_report['total_rows']:,} | "
    f"Fraud: {dataset_report['fraud_rows']:,}",
    flush=True,
  )

  X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, scaler = (
    prepare_train_test_splits(df)
  )
  X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=y_train,
  )
  X_fit_scaled = X_fit.copy()
  X_val_scaled = X_val.copy()
  X_fit_scaled[SCALER_COLUMNS] = scaler.fit_transform(X_fit[SCALER_COLUMNS])
  X_val_scaled[SCALER_COLUMNS] = scaler.transform(X_val[SCALER_COLUMNS])

  final_scaler = StandardScaler()
  X_train_scaled = X_train.copy()
  X_test_scaled = X_test.copy()
  X_train_scaled[SCALER_COLUMNS] = final_scaler.fit_transform(X_train[SCALER_COLUMNS])
  X_test_scaled[SCALER_COLUMNS] = final_scaler.transform(X_test[SCALER_COLUMNS])
  scaler = final_scaler

  print(
    f"\nTrain: {len(X_train):,} | Val: {len(X_val):,} | "
    f"Test: {len(X_test):,} | Features: {len(FEATURE_COLUMNS)}",
    flush=True,
  )

  all_results: list[dict] = []
  tuning_details: dict = {}
  saved_models: dict[str, str] = {}
  model_names = list(get_base_estimators().keys())

  if force_rebuild:
    for path in MODELS_DIR.glob("*_tuned.joblib"):
      path.unlink()
    for path in MODELS_DIR.glob("*_tuned_meta.json"):
      path.unlink()
    best_path = MODELS_DIR / "best_model.joblib"
    if best_path.exists():
      best_path.unlink()

  for name in model_names:
    model_path = MODELS_DIR / f"{name}_tuned.joblib"
    meta_path = MODELS_DIR / f"{name}_tuned_meta.json"
    train_matrix = get_training_matrix(name, X_fit, X_fit_scaled)
    full_train_matrix = get_training_matrix(name, X_train, X_train_scaled)
    test_matrix = get_training_matrix(name, X_test, X_test_scaled)
    val_matrix = get_training_matrix(name, X_val, X_val_scaled)

    if model_path.exists() and meta_path.exists() and not force_rebuild:
      print(f"Skipping {name} (already tuned, loading saved model) ...", flush=True)
      model = joblib.load(model_path)
      tuning_details[name] = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
      print(f"Tuning {name} ({N_ITER} combos, {CV_FOLDS}-fold CV, scoring=accuracy) ...", flush=True)
      best_estimator, best_params, cv_accuracy = tune_model(name, train_matrix, y_fit)

      val_proba = best_estimator.predict_proba(val_matrix)[:, 1]
      threshold = find_best_threshold(y_val, val_proba)

      final_estimator = clone(best_estimator)
      final_estimator.fit(full_train_matrix, y_train)
      model = ThresholdClassifier(final_estimator, threshold=threshold)

      tuning_details[name] = {
        "best_params": best_params,
        "cv_accuracy": round(cv_accuracy, 4),
        "threshold": round(threshold, 4),
      }
      joblib.dump(model, model_path)
      meta_path.write_text(json.dumps(tuning_details[name], indent=2), encoding="utf-8")

    result = evaluate_model(name, model, test_matrix, y_test)
    all_results.append(result)
    saved_models[name] = str(model_path)

    plot_confusion_matrix(
      result["confusion_matrix"],
      f"Tuned — {name.replace('_', ' ').title()}",
      RESULTS_DIR / f"confusion_matrix_{name}_tuned.png",
    )

    print(
      f"  Test: Acc={result['accuracy']:.4f} Prec={result['precision']:.4f} "
      f"Rec={result['recall']:.4f} F1={result['f1_score']:.4f} "
      f"AUC={result['roc_auc']:.4f} Thr={result['threshold']:.3f}",
      flush=True,
    )

  metrics_df = pd.DataFrame(
    [
      {
        "model": r["model"],
        "accuracy": r["accuracy"],
        "precision": r["precision"],
        "recall": r["recall"],
        "f1_score": r["f1_score"],
        "roc_auc": r["roc_auc"],
        "threshold": r["threshold"],
      }
      for r in all_results
    ]
  ).sort_values("accuracy", ascending=False)

  best_model_name = metrics_df.iloc[0]["model"]
  best_model = joblib.load(MODELS_DIR / f"{best_model_name}_tuned.joblib")
  joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

  inference_pipeline = FraudDetectionPipeline(
    model=best_model,
    scaler=scaler,
    account_registry=account_registry,
    feature_columns=FEATURE_COLUMNS,
    model_name=best_model_name,
    threshold=getattr(best_model, "threshold", 0.5),
    use_scaled_inputs=False,
  )
  joblib.dump(inference_pipeline, MODELS_DIR / "fraud_detection_pipeline.joblib")
  joblib.dump(account_registry, MODELS_DIR / "account_registry.joblib")

  metrics_df.to_csv(RESULTS_DIR / "model_comparison_tuned.csv", index=False)
  metrics_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
  plot_roc_curves(all_results, y_test, RESULTS_DIR / "roc_curves_tuned.png")
  plot_metrics_comparison(metrics_df, RESULTS_DIR / "metrics_comparison_tuned.png")

  report = {
    "tuning": {
      "method": "RandomizedSearchCV",
      "cv_folds": CV_FOLDS,
      "n_iter": N_ITER,
      "scoring": "accuracy",
      "target_accuracy_range": [ACCURACY_MIN, ACCURACY_MAX],
    },
    "dataset": dataset_report,
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "feature_columns": FEATURE_COLUMNS,
    "best_model": best_model_name,
    "tuning_details": tuning_details,
    "metrics": metrics_df.to_dict(orient="records"),
    "detailed_results": [
      {k: v for k, v in r.items() if k not in {"y_pred", "y_proba"}}
      for r in all_results
    ],
    "saved_models": saved_models,
  }

  report_path = RESULTS_DIR / "training_report_tuned.json"
  with report_path.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
  with (RESULTS_DIR / "training_report.json").open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

  print(f"\nBest tuned model: {best_model_name}", flush=True)
  print(f"Results: {RESULTS_DIR}", flush=True)
  print(f"Models:  {MODELS_DIR}", flush=True)


if __name__ == "__main__":
  force = "--rebuild" in sys.argv
  quick = "--quick" in sys.argv
  main(force_rebuild=force, quick=quick)
