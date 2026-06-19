"""Client de test de l'API FastAPI — Séance 15.

Envoie des requêtes de prédiction à l'endpoint /predict et affiche les résultats.

Lancement :
    PYTHONPATH=. uv run python scripts/predict_client.py
    PYTHONPATH=. uv run python scripts/predict_client.py --url http://localhost:8000
    PYTHONPATH=. uv run python scripts/predict_client.py --n 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import httpx
import pandas as pd

from src.config import CSV_PATH
from src.data import load_data
from src.feature import na_handle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Colonnes attendues par l'API (même ordre que Features dans api.py)
API_FEATURES = [
    "surface_habitable_logement",
    "hauteur_sous_plafond",
    "ubat_w_par_m2_k",
    "annee_construction",
    "nombre_niveau_logement",
    "conso_chauffage_ep",
    "conso_ecs_ep",
    "emission_ges_chauffage",
    "cout_chauffage",
    "cout_total_5_usages",
    "type_batiment",
    "periode_construction",
    "type_energie_principale_chauffage",
    "qualite_isolation_enveloppe",
    "qualite_isolation_murs",
    "qualite_isolation_menuiseries",
]


def check_health(client: httpx.Client, url: str) -> bool:
    """Vérifier que l'API est opérationnelle."""
    try:
        resp = client.get(f"{url}/health", timeout=5.0)
        resp.raise_for_status()
        logger.info("Health check : %s", resp.json())
        return True
    except httpx.HTTPError as exc:
        logger.error("API inaccessible : %s", exc)
        return False


def sample_rows(n: int) -> pd.DataFrame:
    """Extraire n lignes aléatoires du dataset pour les envoyer à l'API."""
    df = load_data()
    df_clean = na_handle(df)
    return df_clean[API_FEATURES].dropna().sample(n=n, random_state=42)


def predict_batch(client: httpx.Client, url: str, rows: pd.DataFrame) -> list[dict]:
    """Envoyer chaque ligne à /predict et collecter les résultats."""
    results = []
    for i, (_, row) in enumerate(rows.iterrows()):
        payload = row.to_dict()
        # annee_construction doit être un int
        payload["annee_construction"] = int(payload["annee_construction"])

        try:
            resp = client.post(f"{url}/predict", json=payload, timeout=10.0)
            resp.raise_for_status()
            result = resp.json()
            result["row_index"] = i
            results.append(result)
            logger.info(
                "Ligne %d → prediction=%d | probabilité=%.4f",
                i,
                result["prediction"],
                result["probability"],
            )
        except httpx.HTTPError as exc:
            logger.error("Erreur ligne %d : %s", i, exc)
            results.append({"row_index": i, "error": str(exc)})

    return results


def print_summary(results: list[dict]) -> None:
    """Afficher un résumé des prédictions."""
    valid = [r for r in results if "prediction" in r]
    passoires = sum(1 for r in valid if r["prediction"] == 1)
    non_passoires = len(valid) - passoires
    errors = len(results) - len(valid)

    print("\n── Résumé des prédictions ──")
    print(f"  Total envoyé    : {len(results)}")
    print(f"  Succès          : {len(valid)}")
    print(f"  Erreurs         : {errors}")
    print(f"  Passoires (1)   : {passoires}")
    print(f"  Non-passoires (0): {non_passoires}")
    if valid:
        avg_proba = sum(r["probability"] for r in valid) / len(valid)
        print(f"  Probabilité moy.: {avg_proba:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="URL de l'API")
    parser.add_argument("--n", type=int, default=5, help="Nombre de lignes à tester")
    parser.add_argument(
        "--output", default=None, help="Fichier JSON de sortie (optionnel)"
    )
    args = parser.parse_args()

    with httpx.Client() as client:
        # 1. Health check
        if not check_health(client, args.url):
            sys.exit(1)

        # 2. Modèle info
        try:
            resp = client.get(f"{args.url}/model-info", timeout=5.0)
            logger.info("Modèle servi : %s", resp.json())
        except httpx.HTTPError:
            pass

        # 3. Échantillon de données
        logger.info("Chargement de %d lignes depuis %s", args.n, CSV_PATH)
        rows = sample_rows(args.n)

        # 4. Prédictions
        logger.info("Envoi des requêtes à %s/predict...", args.url)
        results = predict_batch(client, args.url, rows)

        # 5. Résumé
        print_summary(results)

        # 6. Export JSON optionnel
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info("Résultats sauvegardés dans %s", args.output)


if __name__ == "__main__":
    main()