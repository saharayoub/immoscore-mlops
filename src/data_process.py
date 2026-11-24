"""Bloc 1 - Préparation des données.

Lit  : data/raw/data.csv + params.yaml (process.corr_threshold)
Écrit: data/processed/dataset.csv
"""
from pathlib import Path

import pandas as pd
import yaml

RAW_PATH = Path("data/raw/data.csv")
OUT_PATH = Path("data/processed/dataset.csv")
TARGET = "SalePrice"


def load_params(path: str = "params.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ne garde que les colonnes numériques et retire l'identifiant."""
    num = df.select_dtypes(include="number")
    return num.drop(columns=["Id"], errors="ignore")


def select_by_correlation(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Garde les variables dont |corr(x, SalePrice)| > threshold, plus la cible."""
    corr = df.corr()[TARGET].abs()
    keep = corr[corr > threshold].index.tolist()
    if TARGET not in keep:
        keep.append(TARGET)
    return df[keep]


def process(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = select_numeric(df)
    df = select_by_correlation(df, threshold)
    return df.dropna()  # supprime les lignes incomplètes


def main() -> None:
    params = load_params()
    threshold = params["process"]["corr_threshold"]

    df = pd.read_csv(RAW_PATH)
    out = process(df, threshold)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"[process] seuil={threshold} -> {out.shape[1] - 1} variables, {len(out)} lignes")


if __name__ == "__main__":
    main()
