# Deploy to Render

This guide deploys the Flask fraud detection prototype as a Render **Web Service**.

## What gets deployed

| Included | Excluded (via `.gitignore`) |
|---|---|
| `app/`, `src/dataset_utils.py`, `wsgi.py` | Raw dataset CSVs (`data/`) |
| `models/fraud_detection_pipeline_deploy.joblib` (~4 KB) | Full in-memory pipeline (~45 MB) |
| `models/account_registry.db.gz` (compressed registry) | Uncompressed `account_registry.db` |
| `outputs/model_results/model_comparison_latest.csv` | EDA plots |
| `outputs/model_results/training_report_tuned.json` | |

## Why deploy artefacts?

Render's **free tier has 512 MB RAM**. The full `fraud_detection_pipeline.joblib` loads ~1.5 million account IDs into Python sets and exceeds that limit. Deploy artefacts use:

- A **slim pipeline** (`fraud_detection_pipeline_deploy.joblib`) — model + scaler only
- A **SQLite registry** (`account_registry.db`) — account lookups from disk, minimal RAM

## Prerequisites

1. Train models and export deploy artefacts locally:

   ```powershell
   python scripts/train_fast.py
   python scripts/export_deploy_artifacts.py
   python scripts/compress_registry_db.py
   ```

2. A [GitHub](https://github.com) account  
3. A [Render](https://render.com) account  

## Step 1 — Push the project to GitHub

From the project root (`FraudDetectionSystem`):

```powershell
git init
git add .
git status
git commit -m "Prepare fraud detection app for Render deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/FraudDetectionSystem.git
git push -u origin main
```

> **Important:** Initialise git inside `FraudDetectionSystem`, not your user home folder.  
> Commit `fraud_detection_pipeline_deploy.joblib` and `account_registry.db.gz` (not the 45 MB pipeline).

The compressed registry is ~47 MB — **under GitHub's 100 MB file limit**. Store it as a **regular git file** (do **not** use Git LFS; free LFS quotas are easy to exceed).

## Step 2 — Create the Render service

### Option A — Blueprint (recommended)

1. Open [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**
2. Connect your GitHub repo
3. Render reads `render.yaml` and creates the web service automatically
4. Click **Apply**

The build command decompresses the registry DB:

`pip install -r requirements-prod.txt && python scripts/decompress_registry_db.py`

### Option B — Manual setup

| Setting | Value |
|---|---|
| **Build Command** | `pip install -r requirements-prod.txt && python scripts/decompress_registry_db.py` |
| **Start Command** | `gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 120` |
| **Health Check Path** | `/health` |

Environment variables: `PYTHON_VERSION=3.12.0`, `FLASK_ENV=production`, `SECRET_KEY` (auto-generate).

## Step 3 — Verify deployment

- Health: `https://YOUR-SERVICE.onrender.com/health` → `{"status":"ok","model_loaded":true}`
- Overview: `https://YOUR-SERVICE.onrender.com/`

## Local production test

```powershell
pip install -r requirements-prod.txt
python scripts/decompress_registry_db.py
$env:FLASK_ENV="production"
gunicorn wsgi:app --bind 0.0.0.0:5000 --workers 1 --threads 1 --timeout 120
```

## Troubleshooting

| Issue | Fix |
|---|---|
| `Out of memory (used over 512Mi)` | Ensure deploy artefacts are committed, not `fraud_detection_pipeline.joblib` |
| `InconsistentVersionWarning` sklearn | Production pins `scikit-learn==1.8.0` in `requirements-prod.txt` |
| `model_loaded: false` | Run export + compress locally; push `*_deploy.joblib` and `account_registry.db.gz` |
| `registry_ready: false` on `/health` | Build failed to decompress registry; check Render build logs |
| GitHub LFS budget exceeded | Remove LFS: delete `.gitattributes`, run `git lfs untrack models/account_registry.db.gz`, re-add the file as regular git |
| GitHub rejects push (>100 MB) | Run `compress_registry_db.py` to shrink the archive |
| Build timeout | Registry decompress adds ~1 min on first build; normal |
