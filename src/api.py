"""API d'inférence du modèle de classification DPE (FastAPI).

Séance 12 - TP FastAPI
    Expose le modèle entraîné via deux endpoints :
        GET  /health      → statut de l'API
        POST /predict     → prédiction passoire thermique
        GET  /model-info  → version du modèle servi (bonus S12-5)

Lancement :
    make api
    PYTHONPATH=. uv run uvicorn src.api:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Stockage global du modèle (chargé une seule fois au démarrage)
ml: dict = {}


# ── S12-3 : lifespan — chargement/déchargement du modèle ─────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_path = MODELS_DIR / "model.joblib"
    logger.info("Chargement du modèle depuis %s", model_path)
    ml["model"] = joblib.load(model_path)
    logger.info("Modèle chargé avec succès.")
    yield
    ml.clear()
    logger.info("Modèle déchargé.")


app = FastAPI(
    title="DPE Passoire Thermique — API de prédiction",
    description=(
        "Prédit si un logement neuf est une passoire thermique (étiquette F ou G). "
        "Classe positive (1) = passoire thermique, classe négative (0) = non-passoire."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── S12-1 : schéma d'entrée ───────────────────────────────────────────────────

class Features(BaseModel):
    """Caractéristiques d'un logement pour la prédiction DPE."""

    # Numériques
    surface_habitable_logement: float = Field(..., ge=0, description="Surface habitable (m²)")
    hauteur_sous_plafond: float = Field(..., ge=0, description="Hauteur sous plafond (m)")
    ubat_w_par_m2_k: float = Field(..., ge=0, description="Coefficient de déperdition thermique (W/m²K)")
    annee_construction: int = Field(..., ge=1800, le=2030, description="Année de construction")
    nombre_niveau_logement: float = Field(..., ge=0, description="Nombre de niveaux")
    conso_chauffage_ep: float = Field(..., description="Consommation chauffage énergie primaire (kWh/m²/an)")
    conso_ecs_ep: float = Field(..., description="Consommation ECS énergie primaire (kWh/m²/an)")
    emission_ges_chauffage: float = Field(..., ge=0, description="Émissions GES chauffage (kgCO2/m²/an)")
    cout_chauffage: float = Field(..., ge=0, description="Coût chauffage annuel (€)")
    cout_total_5_usages: float = Field(..., ge=0, description="Coût total annuel 5 usages (€)")

    # Catégorielles ordonnées
    periode_construction: str = Field(
        ...,
        description="Période de construction",
        examples=["avant 1948", "1948-1974", "2006-2012", "2013-2021", "après 2021"],
    )
    qualite_isolation_enveloppe: str = Field(
        ...,
        description="Qualité isolation enveloppe",
        examples=["insuffisante", "moyenne", "bonne", "très bonne"],
    )
    qualite_isolation_murs: str = Field(
        ...,
        description="Qualité isolation murs",
        examples=["insuffisante", "moyenne", "bonne", "très bonne"],
    )
    qualite_isolation_menuiseries: str = Field(
        ...,
        description="Qualité isolation menuiseries",
        examples=["insuffisante", "moyenne", "bonne", "très bonne"],
    )

    # Catégorielles nominales
    type_batiment: str = Field(
        ...,
        description="Type de bâtiment",
        examples=["maison", "appartement", "immeuble"],
    )
    type_energie_principale_chauffage: str = Field(
        ...,
        description="Type d'énergie principale de chauffage",
        examples=["gaz naturel", "électricité", "fioul domestique", "bois"],
    )
    zone_climatique: str = Field(..., description="Zone climatique (ex. H1, H2, H3)")
    type_ventilation: str = Field(..., description="Type de ventilation (ex. VMC simple flux)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "surface_habitable_logement": 85.0,
                    "hauteur_sous_plafond": 2.5,
                    "ubat_w_par_m2_k": 1.2,
                    "annee_construction": 1975,
                    "nombre_niveau_logement": 2.0,
                    "conso_chauffage_ep": 250.0,
                    "conso_ecs_ep": 45.0,
                    "emission_ges_chauffage": 52.0,
                    "cout_chauffage": 1800.0,
                    "cout_total_5_usages": 2400.0,
                    "periode_construction": "1948-1974",
                    "qualite_isolation_enveloppe": "insuffisante",
                    "qualite_isolation_murs": "insuffisante",
                    "qualite_isolation_menuiseries": "moyenne",
                    "type_batiment": "maison",
                    "type_energie_principale_chauffage": "fioul domestique",
                    "zone_climatique": "H1c",
                    "type_ventilation": "ventilation naturelle",
                }
            ]
        }
    }


# ── S12-2 : schéma de sortie ──────────────────────────────────────────────────

class PredictionOut(BaseModel):
    """Résultat de la prédiction."""
    prediction: int = Field(..., description="Classe prédite : 1 = passoire thermique, 0 = non-passoire")
    probability: float = Field(..., description="Probabilité d'être une passoire thermique")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Monitoring"])
def health() -> dict:
    """Vérifie que l'API est opérationnelle."""
    return {"status": "ok"}


# S12-4 : endpoint de prédiction
@app.post("/predict", response_model=PredictionOut, tags=["Prédiction"])
def predict(features: Features) -> PredictionOut:
    """Prédit si un logement est une passoire thermique (F ou G)."""
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    row = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))


# S12-5 bonus : infos sur le modèle servi
@app.get("/model-info", tags=["Monitoring"])
def model_info() -> dict:
    """Retourne la version du modèle actuellement servi."""
    return {
        "version": os.environ.get("MODEL_VERSION", "unknown"),
        "model_loaded": "model" in ml,
    }