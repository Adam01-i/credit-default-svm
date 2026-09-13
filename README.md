# Credit Default SVM

> Pipeline de machine learning pour estimer le risque de défaut de paiement
> d'un client de carte de crédit le mois suivant.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-SVM-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-2ea44f.svg)](LICENSE)

Ce projet met en œuvre un pipeline de classification binaire complet avec un
SVM (*Support Vector Machine*) : préparation des données, séparation
stratifiée, standardisation, recherche d'hyperparamètres, évaluation,
sérialisation du modèle et prédiction sur de nouveaux clients.

## Sommaire

- [Objectif](#objectif)
- [Jeu de données](#jeu-de-données)
- [Points techniques](#points-techniques)
- [Installation](#installation)
- [Entraînement](#entraînement)
- [Prédiction](#prédiction)
- [Résultats](#résultats)
- [Structure](#structure)
- [Limites](#limites)
- [Licence](#licence)

## Objectif

La cible `default.payment.next.month` indique si le client a connu un défaut
de paiement le mois suivant. Le modèle exploite notamment :

- la limite de crédit et les informations générales du client ;
- l'historique des statuts de paiement sur six mois ;
- les montants des factures sur six mois ;
- les montants des remboursements sur six mois.

Le projet est conçu comme un exemple reproductible de classification supervisée
sur des données tabulaires. Il ne constitue pas un système de décision de
crédit prêt pour la production.

## Jeu de données

Le projet utilise le jeu de données **Default of Credit Card Clients**,
disponible sur le [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
et sur Kaggle. Il contient 30 000 clients et 24 variables explicatives,
auxquelles s'ajoutent `ID` et la variable cible.

Télécharger le fichier puis le placer exactement ici :

```text
data/UCI_Credit_Card.csv
```

Le dépôt contient également :

- `data/sample_credit_data.csv` : échantillon synthétique pour tester le
   pipeline rapidement ;
- `data/new_clients_example.csv` : exemple de fichier destiné à la prédiction.

Le fichier d'entraînement doit contenir au minimum les colonnes `ID` et
`default.payment.next.month`. Les colonnes d'entrée sont conservées dans leur
ordre d'entraînement pour éviter les erreurs de correspondance.

## Points techniques

- **Prévention de la fuite de données** : le `StandardScaler` est ajusté
   uniquement sur `X_train`, puis appliqué à `X_test` et aux nouveaux clients.
- **Séparation reproductible** : split train/test stratifié avec
   `random_state=42`.
- **Déséquilibre des classes** : `class_weight="balanced"` est utilisé pour
   mieux prendre en compte les clients en défaut.
- **Recherche d'hyperparamètres** : `GridSearchCV` optimise `C`, `kernel` et
   `gamma` selon le score F1.
- **Artefact complet** : le modèle, le scaler et les noms de variables sont
   sauvegardés ensemble dans `models/model.pkl`.
- **Validation des entrées** : le script de prédiction vérifie la présence des
   variables attendues avant de produire un résultat.

## Installation

### Cloner le projet

Cloner le dépôt GitHub puis se placer dans son répertoire :

```bash
git clone https://github.com/Adam01-i/credit-default-svm.git
cd credit-default-svm
```

Le dataset UCI n'est pas inclus dans Git. Après le clonage, télécharger
`UCI_Credit_Card.csv` et le placer dans `data/UCI_Credit_Card.csv` comme
indiqué dans la section [Jeu de données](#jeu-de-données).

Lien de telechargement du dataset : [UCI_Credit_Card.csv](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?resource=download)

### Installer les dépendances

Depuis la racine du dépôt :

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Sous Windows PowerShell, l'activation devient :

```powershell
venv\Scripts\Activate.ps1
```

## Entraînement

### Vérification rapide

Utiliser l'échantillon synthétique pour vérifier l'installation :

```bash
python3 src/train.py --data data/sample_credit_data.csv --quick
```

### Entraînement recommandé sur le vrai dataset

La grille réduite permet d'obtenir rapidement un premier modèle :

```bash
python3 src/train.py --data data/UCI_Credit_Card.csv --quick
```

Pour entraîner un seul SVM sans recherche d'hyperparamètres :

```bash
python3 src/train.py --data data/UCI_Credit_Card.csv --no-search
```

La grille complète est disponible sans option. Elle est plus coûteuse, en
particulier avec le noyau RBF :

```bash
python3 src/train.py --data data/UCI_Credit_Card.csv
```

Après l'entraînement, deux fichiers sont générés ou mis à jour :

```text
models/model.pkl       # modèle, scaler et variables attendues
outputs/metrics.json   # rapport de classification et matrice de confusion
```

## Prédiction

Le fichier de nouveaux clients doit contenir les 24 variables explicatives
utilisées à l'entraînement. La colonne `ID` est facultative pour la
prédiction, tandis que la cible ne doit pas être fournie.

Exécuter une prédiction avec le fichier d'exemple :

```bash
python3 src/predict.py --clients data/new_clients_example.csv
```

Pour utiliser un modèle situé ailleurs :

```bash
python3 src/predict.py \
   --model chemin/vers/model.pkl \
   --clients chemin/vers/nouveaux_clients.csv
```

Exemple de sortie :

```text
Client 0 -> 1 (DÉFAUT PROBABLE)
Client 1 -> 0 (pas de défaut prévu)
```

La sortie est une prédiction statistique, pas une certitude ni une décision
automatisée de crédit.

## Résultats

Un entraînement avec `--quick` sur le dataset UCI a produit les résultats de
test suivants :

| Indicateur | Classe 0 | Classe 1 |
| --- | ---: | ---: |
| Précision | 0,87 | 0,49 |
| Rappel | 0,84 | 0,56 |
| F1-score | 0,85 | 0,53 |

La précision globale obtenue est de **0,78**. La classe `1` correspond au
défaut de paiement et représente environ 22 % des observations ; le rappel de
la classe à risque est donc plus pertinent que l'accuracy seule.

Les métriques détaillées et la matrice de confusion sont disponibles dans
[`outputs/metrics.json`](outputs/metrics.json). Elles peuvent varier si le
dataset, les versions des dépendances ou les paramètres d'entraînement
changent.

## Structure

```text
credit-default-svm/
├── data/
│   ├── UCI_Credit_Card.csv       # dataset réel, à fournir localement
│   ├── sample_credit_data.csv    # échantillon synthétique
│   └── new_clients_example.csv   # données d'exemple pour la prédiction
├── models/
│   └── model.pkl                 # artefact généré après entraînement
├── outputs/
│   └── metrics.json              # métriques générées après entraînement
├── src/
│   ├── data_prep.py              # chargement, split et standardisation
│   ├── train.py                  # entraînement et évaluation
│   └── predict.py                # prédiction sur de nouveaux clients
├── requirements.txt
├── LICENSE
└── README.md
```

## Limites et bonnes pratiques

- Les performances présentées sont mesurées sur un split train/test unique.
   Une validation externe et une analyse de stabilité seraient nécessaires
   avant toute utilisation réelle.
- Les données peuvent contenir des biais historiques. Les prédictions ne
   doivent pas être utilisées seules pour accorder ou refuser un crédit.
- Le modèle n'expose pas de probabilité calibrée : `1` signifie que le SVM
   classe le client dans la catégorie de risque, pas qu'il y a 100 % de chance
   de défaut.
- Le dataset réel n'est pas redistribué dans le dépôt ; il doit être obtenu
   auprès de sa source officielle en respectant ses conditions d'utilisation.

## Technologies

Python · pandas · NumPy · scikit-learn · SVC · StandardScaler · GridSearchCV

## Licence

Ce projet est distribué sous licence MIT. Voir [`LICENSE`](LICENSE).
