"""Fraud Detection System — Flask web application."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, flash, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_utils import (
  DEPLOY_PIPELINE_NAME,
  FraudDetectionPipeline,
  is_valid_registry_db,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-in-production")

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "model_results"

_pipeline: FraudDetectionPipeline | None = None
_metrics_df: pd.DataFrame | None = None
_training_report: dict | None = None


def _resolve_registry_path(registry: object) -> None:
  db_path = getattr(registry, "_db_path", None)
  if not db_path:
    return
  path = Path(db_path)
  if not path.is_absolute():
    registry._db_path = str(MODELS_DIR / path.name)


def load_pipeline() -> FraudDetectionPipeline | None:
  deploy_path = MODELS_DIR / DEPLOY_PIPELINE_NAME
  legacy_path = MODELS_DIR / "fraud_detection_pipeline.joblib"

  if deploy_path.exists():
    pipeline = joblib.load(deploy_path)
    _resolve_registry_path(pipeline.account_registry)
    if pipeline.account_registry.uses_sqlite():
      db_path = Path(pipeline.account_registry._db_path)
      if not is_valid_registry_db(db_path):
        logger.error(
          "Deploy pipeline loaded but registry DB is missing or invalid at %s",
          db_path,
        )
        return None
      pipeline.account_registry.ensure_ready()
    return pipeline

  if os.environ.get("FLASK_ENV") == "production":
    print(
      "WARNING: Deploy pipeline missing. Run: python scripts/export_deploy_artifacts.py",
      flush=True,
    )
    return None

  if legacy_path.exists():
    return joblib.load(legacy_path)
  return None


def get_pipeline() -> FraudDetectionPipeline | None:
  global _pipeline
  if _pipeline is None:
    _pipeline = load_pipeline()
  return _pipeline


def get_model_metrics() -> pd.DataFrame | None:
  global _metrics_df
  if _metrics_df is None:
    for name in ("model_comparison_latest.csv", "model_comparison_tuned.csv", "model_comparison.csv"):
      metrics_path = RESULTS_DIR / name
      if metrics_path.exists():
        _metrics_df = pd.read_csv(metrics_path)
        break
  return _metrics_df


def get_training_report() -> dict:
  global _training_report
  if _training_report is None:
    report_path = RESULTS_DIR / "training_report_tuned.json"
    if not report_path.exists():
      report_path = RESULTS_DIR / "training_report.json"
    if report_path.exists():
      _training_report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
      _training_report = {}
  return _training_report


@app.route("/health")
def health():
  pipeline = get_pipeline()
  registry_ready = True
  registry_path = None
  if pipeline is not None and pipeline.account_registry.uses_sqlite():
    registry_path = pipeline.account_registry._db_path
    registry_ready = is_valid_registry_db(registry_path)

  healthy = pipeline is not None and registry_ready
  status_code = 200 if healthy else 503
  return jsonify(
    {
      "status": "ok" if healthy else "degraded",
      "model_loaded": pipeline is not None,
      "registry_ready": registry_ready,
      "registry_path": registry_path,
    }
  ), status_code


@app.route("/")
def index():
  pipeline = get_pipeline()
  training_report = get_training_report()
  metrics_df = get_model_metrics()
  best_model_key = training_report.get("best_model", "unknown")
  best_model = best_model_key.replace("_", " ").title()
  metrics = metrics_df.to_dict(orient="records") if metrics_df is not None else []

  best_accuracy = None
  best_f1 = None
  if metrics_df is not None and not metrics_df.empty:
    best_row = metrics_df.sort_values("accuracy", ascending=False).iloc[0]
    if best_model_key == "unknown":
      best_model = best_row["model"].replace("_", " ").title()
    best_accuracy = float(best_row["accuracy"])
    best_f1 = float(best_row["f1_score"])

  dataset = training_report.get("dataset", {})
  summary_stats = []
  if dataset.get("total_rows"):
    summary_stats.append({"label": "Evaluation records", "value": f"{dataset['total_rows']:,}"})
  if dataset.get("fraud_rows"):
    summary_stats.append({"label": "Fraud samples", "value": f"{dataset['fraud_rows']:,}"})
  summary_stats.append({"label": "Models benchmarked", "value": str(len(metrics)) if metrics else "4"})
  if best_accuracy is not None:
    summary_stats.append({"label": "Best accuracy", "value": f"{best_accuracy * 100:.1f}%"})

  return render_template(
    "index.html",
    best_model=best_model,
    best_accuracy=best_accuracy,
    best_f1=best_f1,
    metrics=metrics,
    model_ready=pipeline is not None,
    summary_stats=summary_stats,
  )


@app.route("/predict", methods=["GET", "POST"])
def predict():
  pipeline = get_pipeline()
  if pipeline is None:
    flash("Model not found. Run `python scripts/export_deploy_artifacts.py` after training.", "error")
    return render_template("predict.html", model_ready=False, result=None)

  result = None
  if request.method == "POST":
    payload = None
    try:
      payload = {
        "step": float(request.form.get("step", 0)),
        "type": request.form.get("type", "PAYMENT").upper(),
        "amount": float(request.form.get("amount", 0)),
        "nameOrig": request.form.get("nameOrig", "C123456789").strip(),
        "oldbalanceOrg": float(request.form.get("oldbalanceOrg", 0)),
        "newbalanceOrig": float(request.form.get("newbalanceOrig", 0)),
        "nameDest": request.form.get("nameDest", "M123456789").strip(),
        "oldbalanceDest": float(request.form.get("oldbalanceDest", 0)),
        "newbalanceDest": float(request.form.get("newbalanceDest", 0)),
        "isFlaggedFraud": int(request.form.get("isFlaggedFraud", 0)),
      }
      result = pipeline.predict(payload)[0]
    except (TypeError, ValueError) as exc:
      flash(f"Invalid input: {exc}", "error")
    except Exception as exc:
      logger.exception("Prediction failed for payload: %s", payload)
      flash(
        "Analysis could not be completed. The account registry may be unavailable "
        f"on the server. Details: {exc}",
        "error",
      )

  return render_template("predict.html", model_ready=True, result=result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
  pipeline = get_pipeline()
  if pipeline is None:
    return jsonify({"error": "Model not trained yet."}), 503

  payload = request.get_json(silent=True)
  if not payload:
    return jsonify({"error": "JSON body required."}), 400

  try:
    result = pipeline.predict(payload)
    if isinstance(payload, dict):
      return jsonify(result[0])
    return jsonify(result)
  except (TypeError, ValueError, KeyError) as exc:
    return jsonify({"error": str(exc)}), 400
  except Exception as exc:
    logger.exception("API prediction failed")
    return jsonify({"error": str(exc)}), 500


@app.route("/api/metrics")
def api_metrics():
  metrics_df = get_model_metrics()
  if metrics_df is None:
    return jsonify({"error": "Metrics not available."}), 404
  return jsonify(metrics_df.to_dict(orient="records"))


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  debug = os.environ.get("FLASK_ENV", "development") != "production"
  app.run(debug=debug, host="0.0.0.0", port=port)
