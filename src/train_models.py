"""Entraînement et optimisation de plusieurs modèles de classification (AutoML + SHAP).

Séance 7 - TP AutoML & SHAP
    Ce module compare trois familles de modèles (Random Forest, XGBoost,
    LightGBM), chacune optimisée par recherche d'hyperparamètres en grille
    (GridSearchCV), et persiste le meilleur dans `models/model.joblib`.

Lancement :
    make train-models
    PYTHONPATH=. uv run python -m src.train_models
    PYTHONPATH=. uv run python -m src.train_models --cv 3 --scoring recall
    PYTHONPATH=. uv run python -m src.train_models --no-mlflow
"""
from __future__ import annotations

from src.evaluation import log_shap_summary
import argparse
import logging
import warnings
from dataclasses import dataclass
from typing import cast

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# ── S7-1 : imports ────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import (
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    REGISTERED_MODEL,
    RANDOM_STATE,
)
from src.data import load_data, split
from src.feature import build_preprocessor, na_handle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Spécification d'un modèle à optimiser."""
    name: str
    estimator: ClassifierMixin
    param_grid: dict


@dataclass
class FitResult:
    """Résultat d'optimisation d'un modèle."""
    name: str
    best_estimator: Pipeline
    best_params: dict
    cv_score: float
    f1: float
    roc_auc: float
    preds: np.ndarray


# ── S7-2 : définition des modèles et grilles ──────────────────────────────────

def build_model_specs() -> list[ModelSpec]:
    """Construire la liste des modèles à optimiser.

    Notes sur les choix de grilles pour le cas DPE (classes très déséquilibrées) :
    - class_weight="balanced" sur RF pour gérer le déséquilibre (~0.3% passoires)
    - scale_pos_weight sur XGBoost : ratio négatifs/positifs ≈ 358
    - is_unbalance sur LightGBM : équivalent automatique du class_weight balanced
    """
    return [
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
        ModelSpec(
            name="xgboost",
            estimator=XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
                scale_pos_weight=358,   # ~36 522 / 102 — compense le déséquilibre
            ),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.1, 0.01],
            },
        ),
        ModelSpec(
            name="lightgbm",
            estimator=LGBMClassifier(
                random_state=RANDOM_STATE,
                verbose=-1,
                is_unbalance=True,
            ),
            param_grid={
                "clf__n_estimators": [100, 200],
                "clf__num_leaves": [31, 63],
                "clf__learning_rate": [0.1, 0.01],
            },
        ),
    ]


# ── Assemblage pipeline ────────────────────────────────────────────────────────

def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    """Assembler le preprocessing et un classifieur dans un pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", estimator),
        ]
    )


# ── S7-3 : optimisation par GridSearchCV ──────────────────────────────────────

def optimize_model(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> FitResult:
    """Optimiser un modèle par GridSearchCV et l'évaluer sur le test."""
    logger.info("Optimisation de %s (cv=%d, scoring=%s)", spec.name, cv, scoring)

    search = GridSearchCV(
        estimator=build_pipeline(spec.estimator),
        param_grid=spec.param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)

    best = search.best_estimator_
    proba = best.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    return FitResult(
        name=spec.name,
        best_estimator=best,
        best_params=search.best_params_,
        cv_score=float(search.best_score_),
        f1=f1_score(y_test, preds),
        roc_auc=roc_auc_score(y_test, proba),
        preds=preds,
    )

# ── S7-4 : logging MLflow ─────────────────────────────────────────────────────

def log_run_to_mlflow(
    result: FitResult,
    x_test,
    y_test,
    cv: int,
    scoring: str,
    register_as: str | None = None,
) -> None:
    """Logger un résultat d'optimisation dans un run MLflow imbriqué."""
    with mlflow.start_run(run_name=result.name, nested=True):
        mlflow.set_tag("model_family", result.name)
        mlflow.log_param("cv", cv)
        mlflow.log_param("scoring", scoring)

        # S7-4a : hyperparamètres et métriques
        mlflow.log_params(result.best_params)
        mlflow.log_metrics({
            f"cv_{scoring}": result.cv_score,
            "f1": result.f1,
            "roc_auc": result.roc_auc,
        })

        # Matrice de confusion
        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Rapport de classification
        report_dict = cast(dict, classification_report(y_test, result.preds, output_dict=True))
        mlflow.log_dict(report_dict, "classification_report.json")
        report_text = cast(str, classification_report(y_test, result.preds))
        mlflow.log_text(report_text, "classification_report.txt")

        # S7-4b : SHAP summary plot
        log_shap_summary(result.best_estimator, x_test, result.name)

        # Enregistrement du modèle
        signature = infer_signature(x_test, result.best_estimator.predict(x_test))
        model_info = mlflow.sklearn.log_model(
            result.best_estimator,
            name="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
        )

        # S7-5 bonus : documenter la version dans le Model Registry
        if register_as and model_info.registered_model_version:
            describe_registered_version(
                name=register_as,
                version=int(model_info.registered_model_version),
                result=result,
                cv=cv,
                scoring=scoring,
            )


# ── S7-5 bonus : documentation Model Registry ─────────────────────────────────

def describe_registered_version(
    name: str,
    version: int,
    result: FitResult,
    cv: int,
    scoring: str,
) -> None:
    """Documenter une version enregistrée dans le Model Registry."""
    client = mlflow.MlflowClient()

    description = (
        f"Modèle : {result.name}\n"
        f"Optimisation : GridSearchCV (cv={cv}, scoring={scoring})\n"
        f"Meilleurs hyperparamètres : {result.best_params}\n"
        f"Métriques test → F1: {result.f1:.4f} | ROC AUC: {result.roc_auc:.4f}"
    )
    client.update_model_version(name, str(version), description=description)

    tags = {
        "model_family": result.name,
        "search_method": "GridSearchCV",
        "cv": str(cv),
        "scoring": scoring,
        "f1": f"{result.f1:.4f}",
        "roc_auc": f"{result.roc_auc:.4f}",
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, str(version), key, value)

    logger.info("Version %d du modèle '%s' documentée dans le registry.", version, name)


# ── Orchestration principale ───────────────────────────────────────────────────

def train_all(
    cv: int = 5,
    scoring: str = "roc_auc",
    use_mlflow: bool = True,
) -> list[FitResult]:
    """Entraîner et comparer les trois modèles, sauvegarder le meilleur."""
    df = load_data()
    df_clean = na_handle(df)
    x_train, x_test, y_train, y_test = split(df_clean)

    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info("Suivi MLflow : %s (expérience: %s)", MLFLOW_TRACKING_URI, EXPERIMENT_NAME)

    results = [
        optimize_model(spec, x_train, y_train, x_test, y_test, cv=cv, scoring=scoring)
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.roc_auc, reverse=True)

    best = results[0]
    logger.info("Meilleur modèle : %s (roc_auc=%.4f, f1=%.4f)", best.name, best.roc_auc, best.f1)

    if use_mlflow:
        with mlflow.start_run(run_name="compare-models"):
            mlflow.log_param("cv", cv)
            mlflow.log_param("scoring", scoring)
            mlflow.set_tag("best_model", best.name)
            mlflow.log_metrics({
                "best_roc_auc": best.roc_auc,
                "best_f1": best.f1,
            })
            for result in results:
                register_as = REGISTERED_MODEL if result is best else None
                log_run_to_mlflow(result, x_test, y_test, cv, scoring, register_as=register_as)
        logger.info("Meilleur modèle enregistré dans le registry sous '%s'", REGISTERED_MODEL)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_estimator, MODELS_DIR / "model.joblib")
    logger.info("Modèle sauvegardé dans %s", MODELS_DIR / "model.joblib")

    # Résumé console
    logger.info("\n── Classement final ──")
    for i, r in enumerate(results, 1):
        logger.info("%d. %s | ROC AUC: %.4f | F1: %.4f", i, r.name, r.roc_auc, r.f1)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--scoring", type=str, default="roc_auc")
    parser.add_argument("--no-mlflow", dest="use_mlflow", action="store_false")
    args = parser.parse_args()
    train_all(cv=args.cv, scoring=args.scoring, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()