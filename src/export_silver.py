"""
scripts/export_silver.py
────────────────────────
Exporte un échantillon du Silver DPE vers un CSV pour le projet MLOps.

Ce script lit le Parquet Silver du projet DPE Analytics Platform
et produit un CSV prêt à l'emploi pour le package mlproject.

Transformations appliquées :
  - Création de la colonne cible binaire `is_passoire`
  - Sélection des features définies dans mlproject/config.py
  - Échantillonnage stratifié pour équilibrer les classes
  - Export CSV dans data/dpe_silver_sample.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
SILVER_PATH = Path(os.getenv("DATA_SOURCE"))
OUTPUT_CSV  = ROOT_DIR / "data" / "dpe_silver_data.csv"

sys.path.insert(0, str(ROOT_DIR))
from src.config import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET
)

SAMPLE_SIZE = 200_000 
RANDOM_STATE = 42

def main():
    df = pd.read_parquet(SILVER_PATH)

    # Création de la cible binaire
    df[TARGET] = df["etiquette_dpe"].isin(["E","F", "G"]).astype(int)

    # ── Sélection des colonnes utiles
    cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    cols_available = [c for c in cols if c in df.columns]

    df = df[cols_available].dropna(subset=[TARGET])

    # ── Échantillonnage stratifié 
    passoires = df[df[TARGET] == 1]
    non_passoires = df[df[TARGET] == 0].sample(
        n=min(SAMPLE_SIZE, len(df[df[TARGET] == 0])),
        random_state=RANDOM_STATE
    )
    df_sample = pd.concat([passoires, non_passoires]).sample(frac=1, random_state=RANDOM_STATE)

    # ── Export
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    main()