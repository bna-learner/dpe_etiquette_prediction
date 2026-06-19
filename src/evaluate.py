"""Évaluation automatisée et validation du modèle.

Séance 11 - TP Tests Données & Modèle

Lancement :
    PYTHONPATH=. uv run python -m src.evaluate
    PYTHONPATH=. uv run python -m src.evaluate --model-uri models:/dpe-passoire-classifier/1
    PYTHONPATH=. uv run python -m src.evaluate --no-validate
    EVAL_ROC_AUC_MIN=0.9999 PYTHONPATH=. uv run python -m src.evaluate  # porte qualité en échec
"""
from __future__ import annotations

import argparse
import logging

import mlflow
import mlflow.data
import mlflow.models
from mlflow.exceptions import MlflowException
from mlflow.models import MetricThreshold

from src.config import (
    CSV_PATH,
    EVAL_F1_MIN,
    EVAL_ROC_AUC_MIN,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL,
    TARGET,
)
from src.data import load_data, split
from src.feature import na_handle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def latest_model_uri() -> str:
    """Résoudre l'URI de la dernière version enregistrée de REGISTERED_MODEL."""
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    if not versions:
        raise RuntimeError(
            f"Aucune version enregistrée pour '{REGISTERED_MODEL}'. "
            "Lancez d'abord un entraînement (make train)."
        )
    latest = max(versions, key=lambda v: int(v.version))
    uri = f"models:/{REGISTERED_MODEL}/{latest.version}"
    logger.info("Dernière version trouvée : %s", uri)
    return uri


# ── S11-1 : seuils de validation ──────────────────────────────────────────────

def build_thresholds() -> dict[str, MetricThreshold]:
    """Construire les seuils de validation depuis la configuration."""
    return {
        "roc_auc": MetricThreshold(threshold=EVAL_ROC_AUC_MIN, greater_is_better=True),
        "f1_score": MetricThreshold(threshold=EVAL_F1_MIN, greater_is_better=True),
    }


# ── S11-2 & S11-3 : évaluation + porte qualité ────────────────────────────────

def evaluate_model(model_uri: str | None = None, validate: bool = True):
    """Évaluer un modèle du registry et, optionnellement, valider les seuils."""
    df = load_data()
    df_clean = na_handle(df)
    _, x_test, _, y_test = split(df_clean)

    # mlflow.evaluate attend un seul DataFrame contenant features + cible
    eval_df = x_test.copy()
    eval_df[TARGET] = y_test.values

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    model_uri = model_uri or latest_model_uri()
    logger.info("Évaluation de %s", model_uri)

    with mlflow.start_run(run_name="evaluate"):
        # S11-2a : traçabilité — logger le jeu d'évaluation comme dataset MLflow
        dataset = mlflow.data.from_pandas(  # type: ignore[attr-defined]
            eval_df,
            source=str(CSV_PATH),
            targets=TARGET,
            name="eval",
        )
        mlflow.log_input(dataset, context="evaluation")

        # S11-2b : évaluation complète (métriques + artefacts auto)
        result = mlflow.models.evaluate(
            model_uri,
            data=eval_df,
            targets=TARGET,
            model_type="classifier",
            evaluators=["default"],
        )
        logger.info(
            "f1_score=%.4f  roc_auc=%.4f",
            result.metrics["f1_score"],
            result.metrics["roc_auc"],
        )

        # S11-3 : porte qualité
        if validate:
            logger.info(
                "Porte qualité — seuils : roc_auc>=%.3f, f1_score>=%.3f",
                EVAL_ROC_AUC_MIN,
                EVAL_F1_MIN,
            )
            mlflow.validate_evaluation_results(build_thresholds(), result)
            logger.info("Porte qualité : OK ✓")

        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-uri", default=None)
    parser.add_argument("--no-validate", dest="validate", action="store_false")
    args = parser.parse_args()

    try:
        evaluate_model(model_uri=args.model_uri, validate=args.validate)
    except MlflowException as exc:
        logger.error("Porte qualité échouée : %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()