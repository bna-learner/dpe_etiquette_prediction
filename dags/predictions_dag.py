"""DAG Airflow - trafic de prévisions quotidien.

Séance 17 - TP Airflow (suite)
    Planifié tous les jours à 10h : échantillonne 20 lignes et les envoie
    en POST /predict pour simuler un flux de prévisions en production.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

N_PREDICTIONS = 20
API_URL = "http://api:8000"

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_send_predictions(**context) -> None:
    """S17-6/7 : échantillonner N lignes et les envoyer à /predict."""
    import httpx
    from src.config import TARGET
    from src.data import load_data
    from src.feature import na_handle

    df = load_data()
    df_clean = na_handle(df)
    features = df_clean.drop(columns=[TARGET])

    # S17-6 : échantillonnage
    sample = features.sample(n=N_PREDICTIONS, random_state=None)

    # S17-7 : envoi à l'API
    success = 0
    errors = 0

    with httpx.Client(base_url=API_URL, timeout=10.0) as client:
        # Vérification santé
        client.get("/health").raise_for_status()
        logger.info("API opérationnelle : %s", API_URL)

        for _, row in sample.iterrows():
            # json.loads(to_json()) garantit des types JSON natifs (pas de numpy)
            payload = json.loads(row.to_json())
            # annee_construction doit être un int
            if "annee_construction" in payload:
                payload["annee_construction"] = int(payload["annee_construction"])
            try:
                response = client.post("/predict", json=payload)
                response.raise_for_status()
                result = response.json()
                logger.debug(
                    "prediction=%d probabilité=%.4f",
                    result["prediction"],
                    result["probability"],
                )
                success += 1
            except httpx.HTTPError as exc:
                logger.warning("Erreur prédiction : %s", exc)
                errors += 1

    logger.info(
        "%d prévisions envoyées à %s (succès=%d, erreurs=%d)",
        N_PREDICTIONS, API_URL, success, errors,
    )

    if errors > 0:
        raise RuntimeError(f"{errors} prédictions ont échoué sur {N_PREDICTIONS}.")


# S17-8 : planifié tous les jours à 10h
with DAG(
    dag_id="daily_predictions",
    description="Envoie 20 prévisions par jour à l'API (trafic simulé)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="0 10 * * *",
    catchup=False,
    tags=["classification", "predictions"],
) as dag:
    send_predictions = PythonOperator(
        task_id="send_predictions",
        python_callable=task_send_predictions,
    )