"""Script d'entraînement du modèle baseline (Régression Logistique) avec MLflow."""

from __future__ import annotations

import logging

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import (
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL,
)
from src.data import load_data, split
from src.feature import build_preprocessor, na_handle

# Configuration des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    # ── Connexion et Paramétrage MLflow ───────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Autolog de scikit-learn (désactivé pour la sauvegarde du modèle car fait manuellement)
    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run(run_name="Baseline_Logistic_Regression"):
        logger.info("Chargement des données brutes...")
        df_raw = load_data()

        logger.info("Gestion des valeurs manquantes...")
        df_clean = na_handle(df_raw)

        logger.info("Découpage Train / Test avec stratification...")
        # Utilisation de la fonction split provenant de src/data.py
        X_train, X_test, y_train, y_test = split(df_clean)

        logger.info("Assemblage du Pipeline (Pre-processor + Logistic Regression)...")
        preprocessor = build_preprocessor()

        # On utilise class_weight="balanced" car le recall est ta métrique prioritaire (TP S5)
        model = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

        logger.info("Entraînement de la Régression Logistique baseline...")
        pipeline.fit(X_train, y_train)

        logger.info("Évaluation du modèle sur le jeu de test...")
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Calcul des métriques définies dans ton config.py
        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        # Affichage console
        logger.info("\n--- RÉSULTATS DU MODÈLE BASELINE ---")
        for metric_name, val in metrics.items():
            logger.info(f"{metric_name.upper()}: {val:.4f}")
        print("\n", classification_report(y_test, y_pred))

        # Log des métriques dans MLflow
        mlflow.log_metrics(metrics)

        # ── Enregistrement du modèle dans le Model Registry ───────────────────
        logger.info(f"Enregistrement du modèle sous le nom : {REGISTERED_MODEL}")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL,
        )
        logger.info("Processus de tracking et d'entraînement terminé avec succès.")


if __name__ == "__main__":
    main()
