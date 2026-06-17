"""Optimisation d'hyperparamètres avec Optuna.

Séance 6 - TP Optuna
    Ce module optimise les hyperparamètres de trois familles de modèles
    (Random Forest, XGBoost, LightGBM) avec Optuna (sampler TPE), compare
    leurs performances et persiste le meilleur dans `models/model.joblib`.

Lancement :
    make train-optuna
    PYTHONPATH=. uv run python -m src.train_optuna
    PYTHONPATH=. uv run python -m src.train_optuna --n-trials 50 --cv 3
    PYTHONPATH=. uv run python -m src.train_optuna --no-mlflow
"""
from __future__ import annotations

from src.evaluation import log_shap_summary
import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

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
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# S6-1 : imports Optuna
import optuna
import optuna.samplers
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score

# S6-2 : imports modèles
from sklearn.ensemble import RandomForestClassifier
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

# Silencer les logs Optuna dans la console
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Spécification d'une famille de modèles à optimiser avec Optuna."""
    name: str
    suggest_params: Callable
    build_estimator: Callable[[dict], ClassifierMixin]


@dataclass
class FamilyResult:
    """Résultat d'optimisation d'une famille de modèles."""
    spec: ModelSpec
    study: Any
    best_pipeline: Pipeline
    test_roc_auc: float
    preds: np.ndarray


# ── S6-2 : définition des espaces de recherche ────────────────────────────────

def build_model_specs() -> list[ModelSpec]:
    """Construire la liste des familles de modèles à optimiser."""

    def rf_suggest(trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }

    def rf_build(params: dict) -> ClassifierMixin:
        return cast(ClassifierMixin, RandomForestClassifier(
            **params,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ))

    def xgb_suggest(trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }

    def xgb_build(params: dict) -> ClassifierMixin:
        return cast(ClassifierMixin, XGBClassifier(
            **params,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            scale_pos_weight=358,  # ~36 522 / 102
            n_jobs=-1,
        ))

    def lgbm_suggest(trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
        }

    def lgbm_build(params: dict) -> ClassifierMixin:
        return cast(ClassifierMixin, LGBMClassifier(
            **params,
            random_state=RANDOM_STATE,
            verbose=-1,
            is_unbalance=True,
        ))

    return [
        ModelSpec(name="random_forest", suggest_params=rf_suggest, build_estimator=rf_build),
        ModelSpec(name="xgboost",       suggest_params=xgb_suggest, build_estimator=xgb_build),
        ModelSpec(name="lightgbm",      suggest_params=lgbm_suggest, build_estimator=lgbm_build),
    ]


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("clf", estimator),
    ])


# ── S6-3 : fonction objectif ──────────────────────────────────────────────────

def objective(trial, spec: ModelSpec, x_train, y_train, cv: int) -> float:
    """Fonction objectif Optuna : ROC AUC moyen en validation croisée."""
    params = spec.suggest_params(trial)
    estimator = spec.build_estimator(params)
    pipeline = build_pipeline(estimator)
    scores = cross_val_score(pipeline, x_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
    return float(scores.mean())


# ── S6-4 & S6-5 : création et lancement de l'étude ───────────────────────────

def run_study(spec: ModelSpec, x_train, y_train, n_trials: int, cv: int):
    """Lancer l'étude Optuna pour une famille de modèles."""
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, spec, x_train, y_train, cv),
        n_trials=n_trials,
    )
    return study


# ── Optimisation d'une famille ────────────────────────────────────────────────

def optimize_family(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    n_trials: int,
    cv: int,
) -> FamilyResult:
    logger.info("Optimisation de %s (n_trials=%d, cv=%d)", spec.name, n_trials, cv)
    study = run_study(spec, x_train, y_train, n_trials=n_trials, cv=cv)

    best_pipeline = build_pipeline(spec.build_estimator(study.best_params))
    best_pipeline.fit(x_train, y_train)
    proba = best_pipeline.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    test_roc_auc = float(roc_auc_score(y_test, proba))

    logger.info(
        "%s : cv_roc_auc=%.4f | test_roc_auc=%.4f | params=%s",
        spec.name, study.best_value, test_roc_auc, study.best_params,
    )
    return FamilyResult(
        spec=spec,
        study=study,
        best_pipeline=best_pipeline,
        test_roc_auc=test_roc_auc,
        preds=preds,
    )

# ── S6-6 : logging MLflow ─────────────────────────────────────────────────────

def log_family_to_mlflow(
    result: FamilyResult,
    x_test,
    y_test,
    n_trials: int,
    cv: int,
    register_as: str | None = None,
) -> None:
    with mlflow.start_run(run_name=result.spec.name, nested=True):
        mlflow.set_tag("model_family", result.spec.name)
        mlflow.set_tag("sampler", "TPE")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv", cv)

        # S6-6 : un run imbriqué par trial
        for trial in result.study.trials:
            with mlflow.start_run(
                run_name=f"{result.spec.name}_trial_{trial.number}", nested=True
            ):
                mlflow.log_params(trial.params)
                if trial.value is not None:
                    mlflow.log_metric("cv_roc_auc", trial.value)

        # Meilleurs hyperparamètres et métriques finales
        mlflow.log_params(result.study.best_params)
        mlflow.log_metric("cv_roc_auc", result.study.best_value)
        mlflow.log_metric("test_roc_auc", result.test_roc_auc)

        # Matrice de confusion
        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.spec.name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        # Rapport de classification
        report_dict = cast(dict, classification_report(y_test, result.preds, output_dict=True))
        mlflow.log_dict(report_dict, "classification_report.json")
        report_text = cast(str, classification_report(y_test, result.preds))
        mlflow.log_text(report_text, "classification_report.txt")

        # SHAP
        log_shap_summary(result.best_pipeline, x_test, result.spec.name)

        # Modèle
        signature = infer_signature(x_test, result.best_pipeline.predict(x_test))
        model_info = mlflow.sklearn.log_model(
            result.best_pipeline,
            name="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
        )

        # S6-7 bonus : documenter la version dans le registry
        if register_as and model_info.registered_model_version:
            describe_registered_version(
                name=register_as,
                version=int(model_info.registered_model_version),
                result=result,
                n_trials=n_trials,
                cv=cv,
            )


# ── S6-7 bonus : documentation Model Registry ─────────────────────────────────

def describe_registered_version(
    name: str,
    version: int,
    result: FamilyResult,
    n_trials: int,
    cv: int,
) -> None:
    client = mlflow.MlflowClient()

    description = (
        f"Modèle : {result.spec.name}\n"
        f"Optimisation : Optuna TPE (n_trials={n_trials}, cv={cv})\n"
        f"Meilleurs hyperparamètres : {result.study.best_params}\n"
        f"CV ROC AUC : {result.study.best_value:.4f}\n"
        f"Test ROC AUC : {result.test_roc_auc:.4f}"
    )
    client.update_model_version(name, str(version), description=description)

    tags = {
        "model_family": result.spec.name,
        "search_method": "Optuna_TPE",
        "n_trials": str(n_trials),
        "cv": str(cv),
        "cv_roc_auc": f"{result.study.best_value:.4f}",
        "test_roc_auc": f"{result.test_roc_auc:.4f}",
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, str(version), key, value)

    logger.info("Version %d du modèle '%s' documentée dans le registry.", version, name)


# ── Orchestration principale ───────────────────────────────────────────────────

def optimize(n_trials: int = 30, cv: int = 5, use_mlflow: bool = True) -> list[FamilyResult]:
    df = load_data()
    df_clean = na_handle(df)
    x_train, x_test, y_train, y_test = split(df_clean)

    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info("Suivi MLflow : %s (expérience: %s)", MLFLOW_TRACKING_URI, EXPERIMENT_NAME)

    results = [
        optimize_family(spec, x_train, y_train, x_test, y_test, n_trials=n_trials, cv=cv)
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.test_roc_auc, reverse=True)

    best = results[0]
    logger.info(
        "Meilleure famille : %s (test_roc_auc=%.4f)", best.spec.name, best.test_roc_auc
    )

    if use_mlflow:
        with mlflow.start_run(run_name="optuna-compare"):
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("cv", cv)
            mlflow.set_tag("best_model", best.spec.name)
            mlflow.log_metrics({
                "best_test_roc_auc": best.test_roc_auc,
                "best_cv_roc_auc": best.study.best_value,
            })
            for result in results:
                register_as = REGISTERED_MODEL if result is best else None
                log_family_to_mlflow(result, x_test, y_test, n_trials, cv, register_as=register_as)
        logger.info("Meilleur modèle enregistré dans le registry sous '%s'", REGISTERED_MODEL)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_pipeline, MODELS_DIR / "model.joblib")
    logger.info("Modèle sauvegardé dans %s", MODELS_DIR / "model.joblib")

    logger.info("\n── Classement final ──")
    for i, r in enumerate(results, 1):
        logger.info(
            "%d. %s | test_roc_auc: %.4f | cv_roc_auc: %.4f",
            i, r.spec.name, r.test_roc_auc, r.study.best_value,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--no-mlflow", dest="use_mlflow", action="store_false")
    args = parser.parse_args()
    optimize(n_trials=args.n_trials, cv=args.cv, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()