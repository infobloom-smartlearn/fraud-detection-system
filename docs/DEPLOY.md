# Deploy to Render

This guide deploys the Flask fraud detection prototype as a Render **Web Service**.

## What gets deployed

| Included | Excluded (via `.gitignore`) |
|---|---|
| `app/`, `src/dataset_utils.py`, `wsgi.py` | Raw dataset CSVs (`data/`) |
| `models/fraud_detection_pipeline.joblib` (~45 MB) | Processed training CSVs |
| `models/best_model.joblib`, `models/decision_tree_tuned.joblib` | Redundant model copies |
| `outputs/model_results/model_comparison_latest.csv` | EDA plots |
| `outputs/model_results/training_report_tuned.json` | |

## Prerequisites

1. Trained model artefacts (run locally first):

   ```powershell
   python scripts/train_fast.py
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
> The `.gitignore` excludes dataset files but **includes** the trained pipeline (~45 MB).

If GitHub rejects a file over 100 MB, use [Git LFS](https://git-lfs.github.com/) for `models/fraud_detection_pipeline.joblib`.

## Step 2 — Create the Render service

### Option A — Blueprint (recommended)

1. Open [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**
2. Connect your GitHub repo
3. Render reads `render.yaml` and creates the web service automatically
4. Click **Apply**

### Option B — Manual setup

1. **New** → **Web Service** → connect your repo  
2. Configure:

   | Setting | Value |
   |---|---|
   | **Language** | Python 3 |
   | **Build Command** | `pip install -r requirements-prod.txt` |
   | **Start Command** | `gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120` |
   | **Health Check Path** | `/health` |

3. **Environment variables**

   | Key | Value |
   |---|---|
   | `PYTHON_VERSION` | `3.12.0` |
   | `FLASK_ENV` | `production` |
   | `SECRET_KEY` | Generate a random secret (Render can auto-generate) |

4. Choose **Free** plan (or paid for always-on) → **Create Web Service**

## Step 3 — Verify deployment

After the build completes (first deploy may take 3–5 minutes):

- App URL: `https://YOUR-SERVICE.onrender.com`
- Health check: `https://YOUR-SERVICE.onrender.com/health`  
  Expected: `{"status":"ok","model_loaded":true}`

## Local production test

Test the same command Render uses:

```powershell
pip install -r requirements-prod.txt
gunicorn wsgi:app --bind 0.0.0.0:5000 --workers 1 --timeout 120
```

Open http://127.0.0.1:5000

## Free tier notes

- Services **sleep after ~15 minutes** of inactivity; the first request after idle can take 30–60 seconds (model load + cold start).
- Free instances have **512 MB RAM** — the app uses **1 Gunicorn worker** to avoid loading the model twice.
- Upgrade to a paid instance for always-on production use.

## Troubleshooting

| Issue | Fix |
|---|---|
| `model_loaded: false` | Ensure `models/fraud_detection_pipeline.joblib` was committed and pushed |
| Build timeout | Use `requirements-prod.txt` (not full `requirements.txt`) |
| Worker timeout on start | Increase Gunicorn `--timeout` (already set to 120s) |
| Out of memory | Keep `--workers 1`; remove extra model files from the repo |
