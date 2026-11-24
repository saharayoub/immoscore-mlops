"""Bloc 3 - Entraînement d'une régression linéaire.

Lit  : data/split/X_train.csv, y_train.csv
Écrit: models/model.pkl
"""
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

SPLIT_DIR = Path("data/split")
MODEL_PATH = Path("models/model.pkl")


def train(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model


def main() -> None:
    X_train = pd.read_csv(SPLIT_DIR / "X_train.csv")
    y_train = pd.read_csv(SPLIT_DIR / "y_train.csv").squeeze("columns")

    model = train(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[train] modèle sauvegardé dans {MODEL_PATH}")


if __name__ == "__main__":
    main()
