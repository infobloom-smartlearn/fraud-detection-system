"""Fraud Detection System — Flask web application."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, flash, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_utils import FraudDetectionPipeline

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-in-production")

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "model_results"


def load_pipeline() -> FraudDetectionPipeline | None:
  pipeline_path = MODELS_DIR / "fraud_detection_pipeline.joblib"
  if pipeline_path.exists():
    return joblib.load(pipeline_path)
  return None


def load_model_metrics() -> pd.DataFrame | None:
  for name in ("model_comparison_latest.csv", "model_comparison_tuned.csv", "model_comparison.csv"):
    metrics_path = RESULTS_DIR / name
    if metrics_path.exists():
      return pd.read_csv(metrics_path)
  return None


def load_training_report() -> dict:
  report_path = RESULTS_DIR / "training_report_tuned.json"
  if not report_path.exists():
    report_path = RESULTS_DIR / "training_report.json"
  if report_path.exists():
    return json.loads(report_path.read_text(encoding="utf-8"))
  return {}


pipeline = load_pipeline()
metrics_df = load_model_metrics()
training_report = load_training_report()


@app.route("/health")
def health():
  return jsonify({"status": "ok", "model_loaded": pipeline is not None}), 200


@app.route("/")
def index():
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
  if pipeline is None:
    flash("Model not found. Run `python src/train_models.py --rebuild` first.", "error")
    return render_template("predict.html", model_ready=False, result=None)

  result = None
  if request.method == "POST":
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

  return render_template("predict.html", model_ready=True, result=result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
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


@app.route("/api/metrics")
def api_metrics():
  if metrics_df is None:
    return jsonify({"error": "Metrics not available."}), 404
  return jsonify(metrics_df.to_dict(orient="records"))


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  debug = os.environ.get("FLASK_ENV", "development") != "production"
  app.run(debug=debug, host="0.0.0.0", port=port)
