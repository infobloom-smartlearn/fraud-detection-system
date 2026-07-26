"""Validate entity features improve accuracy."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset_utils import (
  ENTITY_FEATURE_COLUMNS,
  FEATURE_COLUMNS,
  TARGET_COLUMN,
  add_entity_features,
  build_balanced_dataset_from_parts,
  engineer_features,
)

ALL_FEATURES = FEATURE_COLUMNS + ENTITY_FEATURE_COLUMNS

print("Building dataset (part 1 only for speed)...")
parts = list(Path("data/Cifer-Fraud-Detection-Dataset-AF").glob("*part-1-14.csv"))
df, _ = build_balanced_dataset_from_parts(parts=parts)
print(f"Rows: {len(df):,}")

train_df, test_df = train_test_split(
  df,
  test_size=0.2,
  random_state=42,
  stratify=df[TARGET_COLUMN],
)

train_df, entity_mapper = add_entity_features(train_df)
test_df, _ = add_entity_features(test_df, mapper=entity_mapper)

X_train = train_df[ALL_FEATURES]
y_train = train_df[TARGET_COLUMN]
X_test = test_df[ALL_FEATURES]
y_test = test_df[TARGET_COLUMN]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
  "lr": LogisticRegression(max_iter=2000, random_state=42),
  "dt": DecisionTreeClassifier(max_depth=14, random_state=42),
  "rf": RandomForestClassifier(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1),
  "xgb": XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42,
  ),
}

for name, model in models.items():
  if name == "lr":
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
  else:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
  print(f"{name}: {accuracy_score(y_test, pred):.4f}")
