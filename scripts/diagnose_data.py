"""Quick dataset diagnostics."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset_utils import RAW_COLUMNS, clean_dataframe

path = Path(__file__).resolve().parents[1] / "data" / "Cifer-Fraud-Detection-Dataset-AF" / "Cifer-Fraud-Detection-Dataset-AF-part-1-14.csv"
df = pd.read_csv(path, usecols=RAW_COLUMNS, nrows=200_000)
df = clean_dataframe(df)
df["orig_err"] = (df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]).abs()
df["dest_err"] = (df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]).abs()

for label in [0, 1]:
    sub = df[df["isFraud"] == label]
    orig_ok = (sub["orig_err"] < 0.01).mean()
    dest_ok = (sub["dest_err"] < 0.01).mean()
    print(f"isFraud={label}: orig_err~0={orig_ok:.4f}, dest_err~0={dest_ok:.4f}")

for t in sorted(df["type"].unique()):
    sub = df[df["type"] == t]
    print(f"{t}: fraud_rate={sub['isFraud'].mean():.6f}, n={len(sub)}")

print("flagged:", df.groupby("isFlaggedFraud")["isFraud"].mean().to_dict())
print("amount>200k count:", int((df["amount"] > 200_000).sum()))
