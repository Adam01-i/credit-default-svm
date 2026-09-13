# Prédiction de défaut de paiement — SVM

Modèle de classification binaire (SVM) qui prédit si un client de carte de
crédit sera en défaut de paiement le mois suivant, à partir de son historique
de facturation et de remboursement.

## Pourquoi ce projet

Projet réalisé dans le cadre de mon Master 1 Système d'Information, pour
mettre en pratique un pipeline de classification supervisée complet :
prétraitement, standardisation, entraînement SVM, recherche
d'hyperparamètres, évaluation, sérialisation du modèle et prédiction sur de
nouvelles données.

## Jeu de données

**Default of Credit Card Clients** (UCI Machine Learning Repository /
Kaggle) : 30 000 clients d'une banque taïwanaise, 24 variables explicatives
(limite de crédit, âge, historique de paiement sur 6 mois, montants des
factures et des remboursements) et une variable cible binaire
(`default.payment.next.month`).

⚠️ **Le fichier de données réel n'est pas inclus dans ce dépôt** (30 000
lignes, non fourni au moment de la refonte). Pour le télécharger : cherchez
*"Default of Credit Card Clients Dataset"* sur Kaggle ou l'UCI ML
Repository, puis placez le CSV dans `data/UCI_Credit_Card.csv`.

Un petit échantillon **synthétique** (`data/sample_credit_data.csv`, 600
lignes générées aléatoirement avec le même schéma de colonnes) est fourni
pour vérifier que le pipeline s'exécute correctement de bout en bout sans
dépendre du vrai dataset. Les métriques obtenues avec cet échantillon sont
artificielles et ne reflètent pas les performances réelles.

## Ce qui a été corrigé par rapport à la version d'origine

Le notebook original (`SVM_Classification.ipynb`) contenait plusieurs
problèmes, corrigés dans ce dépôt (détails dans les docstrings de
`src/data_prep.py` et `src/train.py`) :

1. **Fuite de données (data leakage)** : le `StandardScaler` était entraîné
   sur l'ensemble des données *avant* la séparation train/test. Le scaler
   avait donc déjà "vu" les statistiques de l'ensemble de test au moment de
   l'entraînement — ce qui biaise l'évaluation. Corrigé : le scaler est
   maintenant `fit` uniquement sur `X_train`.
2. **Pas de `random_state`** sur le `SVC` (seul le split en avait un) :
   résultats non reproductibles d'une exécution à l'autre. Corrigé.
3. **`param_grid` défini mais jamais utilisé** : la grille d'hyperparamètres
   était écrite dans le notebook mais aucun `GridSearchCV` n'était
   réellement lancé — le modèle final restait le `SVC()` par défaut.
   Corrigé : `GridSearchCV` est maintenant exécuté (`src/train.py`, options
   `--quick` / `--no-search`).
4. **Déséquilibre de classes non traité** (~22 % de défauts dans le vrai
   dataset) : le modèle d'origine avait un rappel de seulement 33 % sur la
   classe minoritaire (clients à risque). Ajout de `class_weight="balanced"`.
5. **`model_svm_test.ipynb`** construisait les nouveaux clients à la main
   dans un tableau numpy sans noms de colonnes (d'où le
   `UserWarning: X does not have valid feature names`). Remplacé par
   `src/predict.py`, qui lit un vrai CSV et vérifie la correspondance des
   colonnes.
6. Chemins de fichiers codés en dur (`model.pkl` dans le dossier courant) →
   chemins relatifs à la racine du projet (`models/model.pkl`).

## Structure du projet

```
credit-default-svm/
├── data/
│   ├── sample_credit_data.csv       # échantillon synthétique (tests)
│   └── new_clients_example.csv      # exemple pour predict.py
├── src/
│   ├── data_prep.py                 # chargement + split + scaling (sans fuite)
│   ├── train.py                     # entraînement + GridSearchCV + sauvegarde
│   └── predict.py                   # prédiction sur de nouveaux clients
├── models/                          # modèle entraîné (model.pkl, généré)
├── outputs/                         # métriques générées (metrics.json)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

Entraînement (grille d'hyperparamètres réduite, recommandé sur l'échantillon) :

```bash
python src/train.py --data data/sample_credit_data.csv --quick
```

Sur le vrai dataset (30 000 lignes), la grille complète (`C`: 0.1/10/100,
`kernel`: rbf/linéaire, `gamma`: 0.01/0.1/1) est coûteuse en temps de calcul
avec un noyau RBF — comptez plusieurs dizaines de minutes selon la machine :

```bash
python src/train.py --data data/UCI_Credit_Card.csv
```

Prédiction sur de nouveaux clients :

```bash
python src/predict.py --clients data/new_clients_example.csv
```

## Résultats obtenus (sur l'échantillon synthétique de test)

Pipeline validé de bout en bout — voir `outputs/metrics.json` pour le détail.
Sur cet échantillon (600 clients, classes équilibrées artificiellement) :
accuracy ≈ 0.86, f1-score ≈ 0.86 sur les deux classes. **Ces chiffres ne
sont pas représentatifs du vrai dataset** (déséquilibré, patterns réels
différents) — à recalculer une fois `UCI_Credit_Card.csv` en place.

## Stack technique

Python, pandas, scikit-learn (SVM, StandardScaler, GridSearchCV), pickle.

## Licence

MIT — voir [LICENSE](./LICENSE).
