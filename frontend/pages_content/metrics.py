"""Page Métriques & Outils — Liens MLflow, Airflow, API."""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://mlflow:5000")
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://airflow-webserver:8080")


def _fetch(url: str, timeout: float = 4.0) -> dict | list | None:
    try:
        r = httpx.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _status_badge(ok: bool) -> str:
    if ok:
        return "<span style='background:#1a7a4a;color:white;padding:0.2rem 0.7rem;border-radius:2px;font-size:0.75rem;font-weight:600;'>EN LIGNE</span>"
    return "<span style='background:#b52c2c;color:white;padding:0.2rem 0.7rem;border-radius:2px;font-size:0.75rem;font-weight:600;'>HORS LIGNE</span>"


def render() -> None:
    st.markdown("""
    <div style='padding: 2rem 0 0.5rem 0;'>
        <div class='badge'>Observabilité</div>
        <h1 style='font-size:2.2rem; margin-top:0.8rem;'>Métriques & Outils</h1>
        <div class='gold-bar'></div>
        <p style='color:#4a5a6a; font-size:0.97rem;'>
            Statut des services, métriques du modèle en production et accès aux outils MLOps.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Statut des services ────────────────────────────────────────────────────
    st.markdown("## Statut des services")

    api_health = _fetch(f"{API_URL}/health")
    mlflow_health = _fetch(f"{MLFLOW_URL}/health")
    airflow_health = _fetch(f"{AIRFLOW_URL}/health")
    model_info = _fetch(f"{API_URL}/info")

    sc1, sc2, sc3 = st.columns(3, gap="medium")

    with sc1:
        ok = api_health is not None
        st.markdown(f"""
        <div class='card' style='text-align:center;'>
            <div style='font-size:2rem;'>⚡</div>
            <div style='font-family:Playfair Display,serif;font-weight:700;
                        color:#0d2137;font-size:1.1rem;margin:0.5rem 0;'>API FastAPI</div>
            {_status_badge(ok)}
            <div style='margin-top:0.8rem;'>
                <a href='{API_URL}/docs' target='_blank'
                   style='color:#c9a84c;font-size:0.82rem;font-weight:600;'>
                    Swagger UI →
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sc2:
        ok = mlflow_health is not None
        st.markdown(f"""
        <div class='card' style='text-align:center;'>
            <div style='font-size:2rem;'>📈</div>
            <div style='font-family:Playfair Display,serif;font-weight:700;
                        color:#0d2137;font-size:1.1rem;margin:0.5rem 0;'>MLflow</div>
            {_status_badge(ok)}
            <div style='margin-top:0.8rem;'>
                <a href='{MLFLOW_URL}' target='_blank'
                   style='color:#c9a84c;font-size:0.82rem;font-weight:600;'>
                    Ouvrir MLflow →
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sc3:
        ok = airflow_health is not None
        st.markdown(f"""
        <div class='card' style='text-align:center;'>
            <div style='font-size:2rem;'>🔄</div>
            <div style='font-family:Playfair Display,serif;font-weight:700;
                        color:#0d2137;font-size:1.1rem;margin:0.5rem 0;'>Airflow</div>
            {_status_badge(ok)}
            <div style='margin-top:0.8rem;'>
                <a href='{AIRFLOW_URL}' target='_blank'
                   style='color:#c9a84c;font-size:0.82rem;font-weight:600;'>
                    Ouvrir Airflow →
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='gold-bar'></div>", unsafe_allow_html=True)

    # ── Modèle en production ───────────────────────────────────────────────────
    st.markdown("## Modèle en production")

    if model_info:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Nom du modèle", model_info.get("model_name", "—"))
        with m2:
            st.metric("Version", model_info.get("model_version", "—"))
        with m3:
            st.metric("ROC-AUC", model_info.get("roc_auc", "—"))
        with m4:
            st.metric("F1-score", model_info.get("f1_score", "—"))
    else:
        st.info("ℹ️ Informations du modèle non disponibles — l'API doit exposer un endpoint `/info`.")

    st.markdown("<div class='gold-bar'></div>", unsafe_allow_html=True)

    # ── Historique des prédictions ─────────────────────────────────────────────
    st.markdown("## Historique des prédictions")

    predictions = _fetch(f"{API_URL}/predictions")
    if predictions and isinstance(predictions, list) and len(predictions) > 0:
        import pandas as pd
        df = pd.DataFrame(predictions)
        st.dataframe(df, use_container_width=True, height=300)
        st.caption(f"{len(df)} prédiction(s) enregistrée(s)")
    else:
        st.markdown("""
        <div style='background:white;border:1px solid #ddd8ce;border-left:4px solid #c9a84c;
                    border-radius:4px;padding:1.2rem 1.5rem;color:#6b7c8d;font-size:0.9rem;'>
            Aucune prédiction enregistrée pour l'instant, ou l'endpoint <code>/predictions</code>
            n'est pas encore implémenté dans l'API.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='gold-bar'></div>", unsafe_allow_html=True)

    # ── Liens outils ───────────────────────────────────────────────────────────
    st.markdown("## Liens & ressources")

    links = [
        ("📈", "MLflow UI", "Suivre les expériences, comparer les runs, gérer le Model Registry",
         MLFLOW_URL, "Ouvrir MLflow"),
        ("🔄", "Airflow", "Surveiller les DAGs de réentraînement automatique et leur historique",
         AIRFLOW_URL, "Ouvrir Airflow"),
        ("⚡", "API — Swagger", "Tester les endpoints REST directement depuis le navigateur",
         f"{API_URL}/docs", "Ouvrir Swagger"),
        ("⚡", "API — Redoc", "Documentation lisible de l'API FastAPI",
         f"{API_URL}/redoc", "Ouvrir Redoc"),
        ("💻", "GitHub", "Code source, CI/CD, historique des commits et issues",
         "https://github.com/username/dpe-etiquette-prediction", "Voir le repo"),
    ]

    for icon, title, desc, url, label in links:
        st.markdown(f"""
        <div style='display:flex;align-items:center;justify-content:space-between;
                    background:white;border:1px solid #ddd8ce;border-radius:4px;
                    padding:1rem 1.5rem;margin-bottom:0.6rem;'>
            <div style='display:flex;align-items:center;gap:1rem;'>
                <div style='font-size:1.4rem;'>{icon}</div>
                <div>
                    <div style='font-weight:600;color:#0d2137;'>{title}</div>
                    <div style='font-size:0.8rem;color:#6b7c8d;margin-top:0.2rem;'>{desc}</div>
                </div>
            </div>
            <a href='{url}' target='_blank'
               style='background:#0d2137;color:#c9a84c;padding:0.45rem 1.1rem;
                      border-radius:4px;font-size:0.82rem;font-weight:600;
                      text-decoration:none;border:1px solid #c9a84c;white-space:nowrap;'>
                {label} →
            </a>
        </div>
        """, unsafe_allow_html=True)