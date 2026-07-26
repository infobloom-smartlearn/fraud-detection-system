# Fraud Detection System

Machine learning-based fraud detection and financial risk monitoring prototype for the **7005SCN Individual Research Project**.

## About

This project develops a lightweight fraud detection system that predicts fraudulent transactions, assigns risk levels, and visualises alerts through a monitoring dashboard. It uses publicly available financial transaction data and compares models including Logistic Regression, Decision Tree, Random Forest, and XGBoost.

## Dataset

**Source:** [CiferAI/Cifer-Fraud-Detection-Dataset-AF](https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF) (Hugging Face)

Local copy: `data/Cifer-Fraud-Detection-Dataset-AF/` (14 CSV parts, ~21M rows total)

To re-download:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_dataset.ps1
```

## Model Training

Train and fine-tune all four proposal models on the enhanced 50/50 multi-part dataset:

```powershell
python scripts/train_fast.py
```

For the full 14-part dataset (slower):

```powershell
python src/train_models.py --rebuild
```

The training pipeline adds **account legitimacy features** (`orig_in_global_legit`, `dest_in_global_legit`) derived from the dataset, then tunes each model for **accuracy** in the 70–90% target range.

**Latest results** (`outputs/model_results/model_comparison_latest.csv`):

| Model | Accuracy |
|---|---|
| **Decision Tree** (best) | 92.0% |
| Logistic Regression | 84.4% |
| XGBoost | 82.2% |
| Random Forest | 79.2% |

## Flask Web App

After training, start the web app with the best model loaded:

```powershell
python app/app.py
```

Open http://127.0.0.1:5000 for the dashboard and transaction prediction form. REST API: `POST /api/predict` with JSON transaction fields.

## Deploy to Render

See **[docs/DEPLOY.md](docs/DEPLOY.md)** for full instructions. Quick summary:

1. Train models locally: `python scripts/train_fast.py`
2. Push this repo to GitHub (git init inside `FraudDetectionSystem`)
3. In [Render](https://render.com): **New → Blueprint** and connect the repo (uses `render.yaml`)

Production start command:

```bash
gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120
```

**Outputs:**
- `models/` — tuned models, `best_model.joblib`, and `fraud_detection_pipeline.joblib`
- `data/processed/tuned/` — enhanced 50/50 dataset with account features
- `outputs/model_results/` — tuned metrics, ROC curves, confusion matrices
- `app/` — Flask dashboard and prediction UI

## Documentation

- [CW1 Project Proposal](docs/CW1-Project-Proposal.md) — Full project proposal (Proforma)

## Student Details

| Field | Details |
| --- | --- |
| **Student Name** | AIYERVBOSA SHALOM ANTHONY |
| **Student ID** | 15621427 |
| **Supervisor** | ALISON HALFORD |
| **Ethics ID** | P194689 (Approved) |
