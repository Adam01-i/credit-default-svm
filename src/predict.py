"""
Charge le modèle entraîné (models/model.pkl) et prédit le risque de défaut
de paiement pour de nouveaux clients fournis dans un CSV.

Le CSV d'entrée doit contenir les mêmes colonnes explicatives que les
données d'entraînement (toutes les colonnes sauf ID et
default.payment.next.month). Contrairement au notebook original
(model_svm_test.ipynb), qui construisait un tableau numpy à la main sans
noms de colonnes (source du warning sklearn "X does not have valid
feature names"), ce script lit un vrai CSV et vérifie que les colonnes
correspondent exactement à celles utilisées à l'entraînement.

Usage:
    python src/predict.py --clients data/new_clients_example.csv
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", required=True, help="CSV des nouveaux clients")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Chemin du modèle entraîné")
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    model, scaler, feature_names = bundle["model"], bundle["scaler"], bundle["feature_names"]

    clients = pd.read_csv(args.clients)

    missing = set(feature_names) - set(clients.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {args.clients} par rapport à l'entraînement : {missing}"
        )

    X_new = clients[feature_names]  # même ordre de colonnes qu'à l'entraînement
    X_scaled = scaler.transform(X_new)
    predictions = model.predict(X_scaled)

    for i, pred in enumerate(predictions):
        label = "DÉFAUT PROBABLE" if pred == 1 else "pas de défaut prévu"
        print(f"Client {i} -> {pred} ({label})")


if __name__ == "__main__":
    main()
