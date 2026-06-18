"""Frontend Streamlit — Prédiction des passoires thermiques DPE.

Lancement : streamlit run frontend/app.py
            API_URL=http://localhost:8000 streamlit run frontend/app.py
"""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="DPE — Passoire thermique", layout="wide", page_icon="🏠")
st.title("🏠 Prédiction des passoires thermiques DPE")
st.caption("Prédit si un logement neuf est une passoire thermique (étiquette F ou G).")

api_url = st.text_input("URL de l'API", value=API_URL)

predict_tab, history_tab = st.tabs(["Prédiction", "Historique"])

with predict_tab:
    st.subheader("Tester l'endpoint /predict")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        # ── S14bis-1 : champs numériques ──────────────────────────────────────
        with col1:
            st.markdown("**Caractéristiques du logement**")
            surface_habitable_logement = st.number_input(
                "Surface habitable (m²)", min_value=0.0, value=85.0, step=1.0
            )
            hauteur_sous_plafond = st.number_input(
                "Hauteur sous plafond (m)", min_value=0.0, value=2.5, step=0.1
            )
            ubat_w_par_m2_k = st.number_input(
                "Ubat (W/m²K)", min_value=0.0, value=1.2, step=0.1,
                help="Coefficient de déperdition thermique"
            )
            annee_construction = st.number_input(
                "Année de construction", min_value=1800, max_value=2030, value=1975, step=1
            )
            nombre_niveau_logement = st.number_input(
                "Nombre de niveaux", min_value=0.0, value=2.0, step=1.0
            )

        with col2:
            st.markdown("**Consommations & coûts**")
            conso_chauffage_ep = st.number_input(
                "Conso. chauffage EP (kWh/m²/an)", min_value=0.0, value=250.0, step=10.0
            )
            conso_ecs_ep = st.number_input(
                "Conso. ECS EP (kWh/m²/an)", min_value=0.0, value=45.0, step=5.0
            )
            emission_ges_chauffage = st.number_input(
                "Émissions GES chauffage (kgCO2/m²/an)", min_value=0.0, value=52.0, step=1.0
            )
            cout_chauffage = st.number_input(
                "Coût chauffage (€/an)", min_value=0.0, value=1800.0, step=100.0
            )
            cout_total_5_usages = st.number_input(
                "Coût total 5 usages (€/an)", min_value=0.0, value=2400.0, step=100.0
            )

        with col3:
            st.markdown("**Caractéristiques qualitatives**")
            type_batiment = st.selectbox(
                "Type de bâtiment", ["maison", "appartement", "immeuble"]
            )
            periode_construction = st.selectbox(
                "Période de construction",
                ["avant 1948", "1948-1974", "2006-2012", "2013-2021", "après 2021"],
            )
            type_energie_principale_chauffage = st.selectbox(
                "Énergie principale chauffage",
                ["gaz naturel", "électricité", "fioul domestique", "bois", "réseau de chaleur"],
            )
            qualite_isolation_enveloppe = st.selectbox(
                "Isolation enveloppe", ["insuffisante", "moyenne", "bonne", "très bonne"]
            )
            qualite_isolation_murs = st.selectbox(
                "Isolation murs", ["insuffisante", "moyenne", "bonne", "très bonne"]
            )
            qualite_isolation_menuiseries = st.selectbox(
                "Isolation menuiseries", ["insuffisante", "moyenne", "bonne", "très bonne"]
            )
            zone_climatique = st.selectbox(
                "Zone climatique", ["H1a", "H1b", "H1c", "H2a", "H2b", "H2c", "H2d", "H3"]
            )
            type_ventilation = st.selectbox(
                "Type de ventilation",
                [
                    "ventilation naturelle",
                    "VMC simple flux auto-réglable",
                    "VMC simple flux hygro A",
                    "VMC simple flux hygro B",
                    "VMC double flux",
                ],
            )

        submitted = st.form_submit_button("🔍 Prédire", use_container_width=True)

    if submitted:
        # S14bis-2 : payload avec les mêmes clés que le schéma Features de l'API
        payload = {
            "surface_habitable_logement": surface_habitable_logement,
            "hauteur_sous_plafond": hauteur_sous_plafond,
            "ubat_w_par_m2_k": ubat_w_par_m2_k,
            "annee_construction": int(annee_construction),
            "nombre_niveau_logement": nombre_niveau_logement,
            "conso_chauffage_ep": conso_chauffage_ep,
            "conso_ecs_ep": conso_ecs_ep,
            "emission_ges_chauffage": emission_ges_chauffage,
            "cout_chauffage": cout_chauffage,
            "cout_total_5_usages": cout_total_5_usages,
            "type_batiment": type_batiment,
            "periode_construction": periode_construction,
            "type_energie_principale_chauffage": type_energie_principale_chauffage,
            "qualite_isolation_enveloppe": qualite_isolation_enveloppe,
            "qualite_isolation_murs": qualite_isolation_murs,
            "qualite_isolation_menuiseries": qualite_isolation_menuiseries,
            "zone_climatique": zone_climatique,
            "type_ventilation": type_ventilation,
        }

        try:
            response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Appel à l'API impossible : {exc}")
        else:
            # S14bis-3 : affichage du résultat
            prediction = result["prediction"]
            probability = result["probability"]

            st.divider()
            col_res1, col_res2 = st.columns(2)

            with col_res1:
                if prediction == 1:
                    st.error("🔴 **PASSOIRE THERMIQUE** (étiquette F ou G)")
                else:
                    st.success("🟢 **Non-passoire thermique** (étiquette A à E)")

            with col_res2:
                st.metric(
                    label="Probabilité d'être une passoire",
                    value=f"{probability:.1%}",
                )
                st.progress(probability)

with history_tab:
    st.subheader("Historique des prévisions")
    # S14bis-4 bonus : journal si endpoint /predictions disponible dans l'API
    try:
        resp = httpx.get(f"{api_url}/predictions", timeout=5.0)
        resp.raise_for_status()
        rows = resp.json()
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("Aucune prévision enregistrée pour l'instant.")
    except Exception:
        st.info("Aucun journal de prévisions : ajoutez un endpoint /predictions à l'API (bonus).")