"""Bloc 4 - Évaluation sur train ET test (l'écart = signal de surapprentissage).

Lit  : models/model.pkl + data/split/*
Écrit: reports/metrics.json
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SPLIT_DIR = Path("data/split")
MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("reports/metrics.json")


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    metrics = {}
    for part in ("train", "test"):
        X = pd.read_csv(SPLIT_DIR / f"X_{part}.csv")
        y = pd.read_csv(SPLIT_DIR / f"y_{part}.csv").squeeze("columns")
        metrics[part] = compute_metrics(y, model.predict(X))

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    for part, m in metrics.items():
        print(f"[eval] {part:5s} RMSE={m['rmse']:,.0f}  MAE={m['mae']:,.0f}  R2={m['r2']:.4f}")


if __name__ == "__main__":
    main()
