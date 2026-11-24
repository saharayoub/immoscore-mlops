"""Bloc 2 - Découpage train / test stratifié par binning de la cible.

Lit  : data/processed/dataset.csv + params.yaml (split.test_size, split.n_bins, split.seed)
Écrit: data/split/X_train.csv, X_test.csv, y_train.csv, y_test.csv
"""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

IN_PATH = Path("data/processed/dataset.csv")
OUT_DIR = Path("data/split")
TARGET = "SalePrice"


def load_params(path: str = "params.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def split(df: pd.DataFrame, test_size: float, n_bins: int, seed: int):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    return train_test_split(X, y, test_size=test_size, stratify=bins, random_state=seed)


def main() -> None:
    p = load_params()["split"]
    df = pd.read_csv(IN_PATH)
    X_train, X_test, y_train, y_test = split(df, p["test_size"], p["n_bins"], p["seed"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(OUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUT_DIR / "y_test.csv", index=False)
    print(f"[split] train={len(X_train)} test={len(X_test)} (seed={p['seed']}, bins={p['n_bins']})")


if __name__ == "__main__":
    main()
