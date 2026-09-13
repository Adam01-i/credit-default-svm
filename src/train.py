"""
Entraînement d'un modèle SVM pour prédire le défaut de paiement le mois
suivant (dataset "default of credit card clients", UCI/Kaggle).

Usage:
    python src/train.py --data data/UCI_Credit_Card.csv
    python src/train.py --data data/sample_credit_data.csv --quick

Corrections apportées par rapport au notebook original :
- StandardScaler fit uniquement sur le train (voir data_prep.py) -> plus de
  fuite de données.
- random_state fixé partout (split ET modèle) -> résultats reproductibles.
- La grille d'hyperparamètres (param_grid), définie mais jamais utilisée
  dans le notebook original, est maintenant réellement passée à GridSearchCV.
- class_weight="balanced" pour compenser le déséquilibre des classes
  (~22% de défauts dans le dataset original) : sans cela, le modèle a
  tendance à sous-détecter la classe minoritaire (rappel de 33% seulement
  dans la version d'origine, cf. README).
- Le split train/test est stratifié (stratify=y).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from data_prep import RANDOM_STATE, load_dataset, split_and_scale

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"


def build_param_grid(quick: bool) -> dict:
    if quick:
        # Grille réduite pour un test rapide (utile en local / CI / échantillon)
        return {"C": [1, 10], "kernel": ["rbf"], "gamma": ["scale"]}
    # Grille complète (celle définie -mais jamais utilisée- dans le notebook original)
    return {"C": [0.1, 10, 100], "kernel": ["rbf", "linear"], "gamma": [0.01, 0.1, 1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Chemin vers le CSV d'entraînement")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Grille d'hyperparamètres réduite (recommandé pour un jeu de "
        "données d'échantillon ou un test rapide ; sur les 30 000 lignes du "
        "dataset complet, la grille complète est très coûteuse en temps de "
        "calcul avec un SVM à noyau RBF).",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Entraîne un seul SVC (C=1, kernel=rbf) sans GridSearchCV.",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    print(f"Chargement des données depuis {args.data} ...")
    df = load_dataset(args.data)
    print(f"{df.shape[0]} lignes, {df.shape[1]} colonnes.")
    print("Taux de défaut :", df["default.payment.next.month"].mean().round(3))

    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)

    if args.no_search:
        print("Entraînement d'un SVC simple (sans recherche d'hyperparamètres)...")
        model = SVC(C=1, kernel="rbf", gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        best_params = model.get_params()
    else:
        param_grid = build_param_grid(args.quick)
        print(f"GridSearchCV sur la grille : {param_grid}")
        base_model = SVC(class_weight="balanced", random_state=RANDOM_STATE)
        search = GridSearchCV(
            base_model, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        print("Meilleurs paramètres :", best_params)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(classification_report(y_test, y_pred))
    print("Matrice de confusion :", cm)

    # Sauvegarde du modèle + scaler + noms des colonnes (nécessaires pour predict.py)
    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "feature_names": feature_names}, f)

    metrics = {
        "best_params": str(best_params),
        "classification_report": report,
        "confusion_matrix": cm,
    }
    with open(OUTPUTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModèle sauvegardé dans {MODELS_DIR / 'model.pkl'}")
    print(f"Métriques sauvegardées dans {OUTPUTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
