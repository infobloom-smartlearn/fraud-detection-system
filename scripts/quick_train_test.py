"""Quick training experiment."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset_utils import FEATURE_COLUMNS, TARGET_COLUMN, build_balanced_dataset_from_parts, prepare_train_test_splits

df, _ = build_balanced_dataset_from_parts()
X_train, y_train, X_test, y_test, X_train_s, X_test_s, _ = prepare_train_test_splits(df)

models = {
    "lr": LogisticRegression(max_iter=2000, random_state=42),
    "dt": DecisionTreeClassifier(max_depth=16, random_state=42),
    "rf": RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1),
    "xgb": XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=42),
}

for name, model in models.items():
    if name == "lr":
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    print(name, accuracy_score(y_test, pred))
