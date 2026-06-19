"""DAG Airflow - pipeline de ré-entraînement du modèle DPE.

Séance 17 - TP Airflow
    Pipeline complet : préparation → baseline → comparaison modèles → qualité.
    Tous les runs sont loggés dans MLflow sous le même experiment.
    Planifié tous les lundis à 3h du matin.

    Runs MLflow produits à chaque exécution :
        - Baseline_Logistic_Regression
        - compare-models (avec sous-runs RF, XGBoost, LightGBM)

UI Airflow : http://localhost:8080
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.65

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_prepare_data(**context) -> None:
    """Vérifier que les données sont disponibles."""
    import os
    from src.config import CSV_PATH

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Fichier de données introuvable : {CSV_PATH}. "
            "Assurez-vous que le volume data/ est monté."
        )
    logger.info("Données disponibles : %s", CSV_PATH)


def task_train_baseline(**context) -> None:
    """S17-2a : entraîner la régression logistique baseline et pousser f1 dans XCom."""
    import joblib
    import mlflow
    import mlflow.sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import Pipeline

    from src.config import (
        EXPERIMENT_NAME,
        MLFLOW_TRACKING_URI,
        MODELS_DIR,
        REGISTERED_MODEL,
    )
    from src.data import load_data, split
    from src.feature import build_preprocessor, na_handle

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data()
    df_clean = na_handle(df)
    X_train, X_test, y_train, y_test = split(df_clean)

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=2000, random_state=42, class_weight="balanced"
        )),
    ])

    with mlflow.start_run(run_name="Baseline_Logistic_Regression"):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        f1 = float(f1_score(y_test, y_pred))

        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(pipeline, name="model")
        logger.info("Baseline — f1=%.4f", f1)

    # Sauvegarder temporairement (sera écrasé si un meilleur modèle est trouvé)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "model_baseline.joblib")

    context["ti"].xcom_push(key="baseline_f1", value=f1)


def task_train_models(**context) -> None:
    """S17-2b : comparer RF / XGBoost / LightGBM via GridSearchCV et pousser le meilleur f1."""
    from src.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI
    from src.data import load_data, split
    from src.feature import na_handle
    from src.train_models import build_model_specs, optimize_model, log_run_to_mlflow
    from src.config import REGISTERED_MODEL

    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data()
    df_clean = na_handle(df)
    X_train, X_test, y_train, y_test = split(df_clean)

    # GridSearchCV rapide pour le DAG (cv=3 au lieu de 5)
    results = [
        optimize_model(spec, X_train, y_train, X_test, y_test, cv=3, scoring="roc_auc")
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.roc_auc, reverse=True)
    best = results[0]

    logger.info(
        "Meilleur modèle : %s (roc_auc=%.4f, f1=%.4f)",
        best.name, best.roc_auc, best.f1
    )

    with mlflow.start_run(run_name="compare-models"):
        mlflow.log_param("cv", 3)
        mlflow.set_tag("best_model", best.name)
        mlflow.log_metrics({"best_roc_auc": best.roc_auc, "best_f1": best.f1})

        for result in results:
            register_as = REGISTERED_MODEL if result is best else None
            log_run_to_mlflow(result, X_test, y_test, cv=3, scoring="roc_auc",
                              register_as=register_as)

    context["ti"].xcom_push(key="best_model_name", value=best.name)
    context["ti"].xcom_push(key="best_f1", value=best.f1)
    context["ti"].xcom_push(key="best_roc_auc", value=best.roc_auc)


def task_check_quality(**context) -> None:
    """S17-3 : vérifier que le meilleur modèle dépasse le seuil minimal."""
    ti = context["ti"]
    baseline_f1 = ti.xcom_pull(task_ids="train_baseline", key="baseline_f1")
    best_f1 = ti.xcom_pull(task_ids="train_models", key="best_f1")
    best_name = ti.xcom_pull(task_ids="train_models", key="best_model_name")
    best_roc_auc = ti.xcom_pull(task_ids="train_models", key="best_roc_auc")

    logger.info("── Résumé du ré-entraînement ──")
    logger.info("Baseline        : f1=%.4f", baseline_f1)
    logger.info("Meilleur modèle : %s | f1=%.4f | roc_auc=%.4f",
                best_name, best_f1, best_roc_auc)

    if best_f1 < QUALITY_THRESHOLD:
        raise ValueError(
            f"Porte qualité échouée : {best_name} f1={best_f1:.4f} "
            f"< seuil={QUALITY_THRESHOLD}. Le modèle n'est pas promu."
        )

    logger.info("Porte qualité OK ✓ — modèle promu : %s", best_name)


# S17-4 : planifié tous les lundis à 3h
with DAG(
    dag_id="model_retraining",
    description="Baseline + comparaison RF/XGB/LGBM + porte qualité — tous les lundis à 3h",
    schedule="0 3 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["classification", "training", "mlflow"],
) as dag:
    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=task_prepare_data,
    )
    train_baseline = PythonOperator(
        task_id="train_baseline",
        python_callable=task_train_baseline,
    )
    train_models = PythonOperator(
        task_id="train_models",
        python_callable=task_train_models,
    )
    check = PythonOperator(
        task_id="check_quality",
        python_callable=task_check_quality,
    )

    # S17-5 : ordre d'exécution
    prepare >> train_baseline >> train_models >> check