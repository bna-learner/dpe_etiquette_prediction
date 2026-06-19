"""Page d'accueil — Contexte du projet DPE."""
from __future__ import annotations

import streamlit as st


def render() -> None:  # noqa: PLR0912
    # ── Hero ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding: 3rem 0 1rem 0;'>
        <div class='badge'>Projet MLOps · DPE Logements Neufs</div>
        <h1 style='font-size:2.8rem; margin-top:1rem; line-height:1.2;'>
            Identifier les passoires<br>thermiques avant qu'il ne soit trop tard.
        </h1>
        <div class='gold-bar'></div>
        <p style='font-size:1.1rem; color:#4a5a6a; max-width:700px; line-height:1.7;'>
            Un outil de prédiction automatisée basé sur les données DPE des logements neufs,
            pour anticiper les étiquettes F et G et guider les décisions énergétiques.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Chiffres clés ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Logements passoires en France", "5,2 M", help="Source ADEME 2023")
    with c2:
        st.metric("Part du parc résidentiel", "17 %", help="Étiquettes F ou G")
    with c3:
        st.metric("Surcoût énergétique moyen", "+€1 200 /an", help="Par rapport à un logement B")
    with c4:
        st.metric("Objectif rénovation 2028", "700 000 /an", help="Loi Climat & Résilience")

    st.markdown("<div class='gold-bar' style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # ── Problématique ──────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("## La problématique")
        st.markdown("""
        <div class='card'>
        <p style='line-height:1.8; font-size:0.97rem;'>
        Le <strong>Diagnostic de Performance Énergétique (DPE)</strong> classe les logements
        de A (très performant) à G (très énergivore). Les étiquettes <strong>F et G</strong>,
        communément appelées <em>passoires thermiques</em>, désignent les biens les plus
        consommateurs d'énergie.
        </p>
        <p style='line-height:1.8; font-size:0.97rem; margin-top:1rem;'>
        Depuis la loi Climat & Résilience de 2021, ces logements sont progressivement
        <strong>interdits à la location</strong> — les G depuis 2025, les F à partir de 2028.
        Anticiper cette classification est devenu un enjeu majeur pour les propriétaires,
        les bailleurs sociaux et les professionnels de l'immobilier.
        </p>
        <p style='line-height:1.8; font-size:0.97rem; margin-top:1rem;'>
        Ce projet propose un modèle de <strong>classification binaire</strong> entraîné sur
        les données officielles DPE de logements neufs, capable de prédire automatiquement
        si un bien risque d'être classé passoire thermique.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("## Impact & Enjeux")
        items = [
            ("🏛️", "Conformité légale", "Anticipez les interdictions de location avant 2028"),
            ("💶", "Valorisation", "Un logement rénové gagne jusqu'à 15 % de valeur"),
            ("🌱", "Environnement", "Le bâtiment = 43 % de la consommation d'énergie en France"),
            ("👪", "Précarité", "12 M de Français en situation de précarité énergétique"),
        ]
        for icon, title, desc in items:
            st.markdown(f"""
            <div style='display:flex; gap:1rem; align-items:flex-start;
                        background:white; border:1px solid #ddd8ce;
                        border-left:4px solid #c9a84c;
                        border-radius:4px; padding:0.9rem 1rem; margin-bottom:0.7rem;'>
                <div style='font-size:1.5rem;'>{icon}</div>
                <div>
                    <div style='font-weight:600; color:#0d2137; font-size:0.9rem;'>{title}</div>
                    <div style='color:#6b7c8d; font-size:0.82rem; margin-top:0.2rem;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='gold-bar'></div>", unsafe_allow_html=True)

    # ── Pipeline MLOps ─────────────────────────────────────────────────────────
    st.markdown("## Architecture MLOps")
    st.markdown("""
    <div style='display:grid; grid-template-columns: repeat(5, 1fr); gap:0.5rem; margin:1.5rem 0;'>
        <div style='text-align:center; background:white; border:1px solid #ddd8ce; border-top:3px solid #c9a84c; border-radius:4px; padding:1rem 0.5rem;'>
            <div style='font-size:1.5rem;'>📦</div>
            <div style='font-size:0.75rem; font-weight:600; color:#0d2137; margin-top:0.4rem;'>Données</div>
            <div style='font-size:0.7rem; color:#6b7c8d;'>ADEME DPE<br>Logements neufs</div>
        </div>
        <div style='text-align:center; background:white; border:1px solid #ddd8ce; border-top:3px solid #c9a84c; border-radius:4px; padding:1rem 0.5rem;'>
            <div style='font-size:1.5rem;'>⚙️</div>
            <div style='font-size:0.75rem; font-weight:600; color:#0d2137; margin-top:0.4rem;'>Entraînement</div>
            <div style='font-size:0.7rem; color:#6b7c8d;'>Scikit-learn<br>+ Optuna</div>
        </div>
        <div style='text-align:center; background:white; border:1px solid #ddd8ce; border-top:3px solid #c9a84c; border-radius:4px; padding:1rem 0.5rem;'>
            <div style='font-size:1.5rem;'>📈</div>
            <div style='font-size:0.75rem; font-weight:600; color:#0d2137; margin-top:0.4rem;'>Tracking</div>
            <div style='font-size:0.7rem; color:#6b7c8d;'>MLflow<br>Model Registry</div>
        </div>
        <div style='text-align:center; background:white; border:1px solid #ddd8ce; border-top:3px solid #c9a84c; border-radius:4px; padding:1rem 0.5rem;'>
            <div style='font-size:1.5rem;'>🚀</div>
            <div style='font-size:0.75rem; font-weight:600; color:#0d2137; margin-top:0.4rem;'>Déploiement</div>
            <div style='font-size:0.7rem; color:#6b7c8d;'>FastAPI<br>Docker · VPS</div>
        </div>
        <div style='text-align:center; background:white; border:1px solid #ddd8ce; border-top:3px solid #c9a84c; border-radius:4px; padding:1rem 0.5rem;'>
            <div style='font-size:1.5rem;'>🔄</div>
            <div style='font-size:0.75rem; font-weight:600; color:#0d2137; margin-top:0.4rem;'>Orchestration</div>
            <div style='font-size:0.7rem; color:#6b7c8d;'>Airflow<br>CI/CD GitHub</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='gold-bar'></div>", unsafe_allow_html=True)

    # ── Auteur ─────────────────────────────────────────────────────────────────
    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.markdown("""
        <div class='card' style='text-align:center;'>
            <div style='font-size:3rem;'>👤</div>
            <div style='font-family: Playfair Display, serif; font-size:1.4rem;
                        font-weight:700; color:#0d2137; margin-top:0.5rem;'> Bienvenu NATCHIA </div>
            <div style='color:#6b7c8d; font-size:0.85rem; margin-top:0.3rem;'>
                Projet MLOps · Promotion 2025
            </div>
            <div style='margin-top:1rem;'>
                <a href='https://github.com/username/dpe-etiquette-prediction'
                   target='_blank'
                   style='background:#0d2137; color:#c9a84c; padding:0.5rem 1.2rem;
                          border-radius:4px; font-size:0.85rem; font-weight:600;
                          text-decoration:none; border:1px solid #c9a84c;'>
                    💻 Voir le repo GitHub
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("## À propos du projet")
        st.markdown("""
        <div style='line-height:1.8; font-size:0.95rem; color:#4a5a6a;'>
        Ce projet a été réalisé dans le cadre d'une formation MLOps. Il couvre l'ensemble
        du cycle de vie d'un modèle de machine learning : collecte et nettoyage des données,
        feature engineering, entraînement et optimisation, évaluation rigoureuse,
        déploiement containerisé et orchestration des pipelines de réentraînement.
        <br><br>
        Les données proviennent de la base officielle ADEME des diagnostics DPE
        pour les <strong>logements neufs</strong>, disponible en open data.
        <br><br>
        Le modèle est réentraîné automatiquement via <strong>Airflow</strong> et tracké
        avec <strong>MLflow</strong>. La porte qualité (ROC-AUC et F1-score) bloque
        tout déploiement d'un modèle dégradé.
        </div>
        """, unsafe_allow_html=True)