"""
Chargement et préparation des données pour la prédiction de défaut de paiement.

Bug corrigé par rapport à la version originale (SVM_Classification.ipynb) :
le StandardScaler était entraîné (fit) sur l'ensemble des données AVANT la
séparation train/test. Le scaler "voyait" donc la moyenne/variance de
l'ensemble de test au moment de l'entraînement du modèle : c'est une fuite
de données (data leakage) qui gonfle artificiellement les performances
mesurées. Ici, le scaler est fit UNIQUEMENT sur X_train, puis appliqué
(transform) à X_train et X_test séparément.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = "default.payment.next.month"
ID_COL = "ID"
RANDOM_STATE = 42


def load_dataset(path: str) -> pd.DataFrame:
    """Charge le CSV brut (schéma UCI 'default of credit card clients')."""
    df = pd.read_csv(path)
    missing = {ID_COL, TARGET_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes attendues manquantes dans {path}: {missing}")
    return df


def split_and_scale(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """Sépare features/cible, split train/test, puis standardise SANS fuite.

    Retourne X_train, X_test, y_train, y_test, scaler (le scaler doit être
    sauvegardé avec le modèle pour transformer les futures données de la
    même façon).
    """
    X = df.drop(columns=[TARGET_COL, ID_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # préserve le taux de défaut dans train et test
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit UNIQUEMENT sur train
    X_test_scaled = scaler.transform(X_test)  # transform seulement sur test

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()
