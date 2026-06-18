# mlproject/config.py
# =============================================================================
# Configuration du projet DPE — Prédiction des passoires thermiques
#
# Problématique : Classification binaire
#   - Cible    : is_passoire (1 = passoire thermique F/G, 0 = non-passoire)
#   - Contexte : Prédire si un logement neuf est une passoire thermique
#                pour anticiper les risques de non-conformité réglementaire
#                et orienter les actions de rénovation.
# =============================================================================

import os
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CSV_PATH = DATA_DIR / "dpe_silver_data.csv"
MODELS_DIR = ROOT_DIR / "models"

# ── Cible ─────────────────────────────────────────────────────────────────────
# Colonne binaire créée depuis etiquette_dpe :
#   etiquette_dpe ∈ {F, G} → is_passoire = 1
#   etiquette_dpe ∈ {A, B, C, D, E} → is_passoire = 0
TARGET = "is_passoire"
POSITIVE_LABEL = 1  # passoire = classe positive (ce qu'on veut détecter)

# ── Features ──────────────────────────────────────────────────────────────────
# Sélection depuis le Silver DPE
# Règle : pas de data leakage → on exclut les colonnes qui dérivent
# directement de l'étiquette (conso_5_usages_par_m2_ep est la base du calcul DPE)

NUMERICAL_FEATURES = [
    "surface_habitable_logement",  # surface du logement (m²)
    "hauteur_sous_plafond",  # hauteur sous plafond (m)
    "ubat_w_par_m2_k",  # coefficient de déperdition thermique (W/m²K)
    "annee_construction",  # année de construction
    "nombre_niveau_logement",  # nombre de niveaux
    "conso_chauffage_ep",  # consommation chauffage énergie primaire
    "conso_ecs_ep",  # consommation ECS énergie primaire
    "emission_ges_chauffage",  # émissions GES chauffage
    "cout_chauffage",  # coût chauffage annuel (€)
    "cout_total_5_usages",  # coût total annuel (€)
]

CATEGORICAL_FEATURES = [
    "type_batiment",  # maison / appartement / immeuble
    "periode_construction",  # tranche de période de construction
    "type_energie_principale_chauffage",  # gaz / électricité / fioul / etc.
    "qualite_isolation_enveloppe",  # très bonne / bonne / insuffisante / etc.
    "qualite_isolation_murs",
    "qualite_isolation_menuiseries",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ── Split ─────────────────────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "dpe-passoire-prediction"
REGISTERED_MODEL = "dpe-passoire-classifier"

# ── Métriques cibles ──────────────────────────────────────────────────────────
# Le recall est prioritaire : on préfère manquer un bon logement
# plutôt que de rater une passoire (coût réglementaire élevé)
PRIMARY_METRIC = "recall"
METRICS = ["f1", "roc_auc", "precision", "recall"]
