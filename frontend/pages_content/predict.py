"""Page de prédiction — Formulaire + appel API."""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def _get_model_info(api_url: str) -> dict | None:
    """Récupère les infos du modèle depuis /info."""
    try:
        r = httpx.get(f"{api_url}/info", timeout=4.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def render() -> None:
    st.markdown("""
    <div style='padding: 2rem 0 0.5rem 0;'>
        <div class='badge'>Moteur de prédiction</div>
        <h1 style='font-size:2.2rem; margin-top:0.8rem;'>Diagnostic énergétique automatisé</h1>
        <div class='gold-bar'></div>
        <p style='color:#4a5a6a; font-size:0.97rem;'>
            Renseignez les caractéristiques du logement. Le modèle prédit si le bien
            est une <strong>passoire thermique</strong> (étiquette F ou G).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Bandeau modèle ─────────────────────────────────────────────────────────
    col_url, col_info = st.columns([2, 3])
    with col_url:
        api_url = st.text_input("URL de l'API", value=API_URL, label_visibility="collapsed",
                                placeholder="http://localhost:8000")
    with col_info:
        info = _get_model_info(api_url)
        if info:
            st.markdown(f"""
            <div style='background:#0d2137; color:white; border-radius:4px;
                        padding:0.5rem 1.2rem; display:flex; gap:2rem; align-items:center;
                        font-size:0.85rem;'>
                <span>🤖 <strong style='color:#c9a84c;'>Modèle :</strong> {info.get('model_name', '—')}</span>
                <span>🏷️ <strong style='color:#c9a84c;'>Version :</strong> {info.get('model_version', '—')}</span>
                <span>✅ <strong style='color:#c9a84c;'>Statut :</strong> En ligne</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#3a1a1a; color:#ffaaaa; border-radius:4px;
                        padding:0.5rem 1.2rem; font-size:0.85rem;'>
                ⚠️ API inaccessible — vérifiez l'URL et que la stack est démarrée
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Formulaire ─────────────────────────────────────────────────────────────
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown("**🏗️ Caractéristiques du logement**")
            surface_habitable_logement = st.number_input(
                "Surface habitable (m²)", min_value=0.0, value=85.0, step=1.0)
            hauteur_sous_plafond = st.number_input(
                "Hauteur sous plafond (m)", min_value=0.0, value=2.5, step=0.1)
            ubat_w_par_m2_k = st.number_input(
                "Ubat (W/m²K)", min_value=0.0, value=1.2, step=0.1,
                help="Coefficient de déperdition thermique global")
            annee_construction = st.number_input(
                "Année de construction", min_value=1800, max_value=2030, value=1975, step=1)
            nombre_niveau_logement = st.number_input(
                "Nombre de niveaux", min_value=0.0, value=2.0, step=1.0)
            type_batiment = st.selectbox(
                "Type de bâtiment", ["maison", "appartement", "immeuble"])
            periode_construction = st.selectbox(
                "Période de construction",
                ["avant 1948", "1948-1974", "2006-2012", "2013-2021", "après 2021"])

        with col2:
            st.markdown("**⚡ Consommations & coûts**")
            conso_chauffage_ep = st.number_input(
                "Conso. chauffage EP (kWh/m²/an)", min_value=0.0, value=250.0, step=10.0)
            conso_ecs_ep = st.number_input(
                "Conso. ECS EP (kWh/m²/an)", min_value=0.0, value=45.0, step=5.0)
            emission_ges_chauffage = st.number_input(
                "Émissions GES chauffage (kgCO2/m²/an)", min_value=0.0, value=52.0, step=1.0)
            cout_chauffage = st.number_input(
                "Coût chauffage (€/an)", min_value=0.0, value=1800.0, step=100.0)
            cout_total_5_usages = st.number_input(
                "Coût total 5 usages (€/an)", min_value=0.0, value=2400.0, step=100.0)
            type_energie_principale_chauffage = st.selectbox(
                "Énergie principale chauffage",
                ["gaz naturel", "électricité", "fioul domestique", "bois", "réseau de chaleur"])

        with col3:
            st.markdown("**🧱 Isolation & environnement**")
            qualite_isolation_enveloppe = st.selectbox(
                "Isolation enveloppe", ["insuffisante", "moyenne", "bonne", "très bonne"])
            qualite_isolation_murs = st.selectbox(
                "Isolation murs", ["insuffisante", "moyenne", "bonne", "très bonne"])
            qualite_isolation_menuiseries = st.selectbox(
                "Isolation menuiseries", ["insuffisante", "moyenne", "bonne", "très bonne"])
            zone_climatique = st.selectbox(
                "Zone climatique", ["H1a", "H1b", "H1c", "H2a", "H2b", "H2c", "H2d", "H3"])
            type_ventilation = st.selectbox(
                "Type de ventilation",
                ["ventilation naturelle", "VMC simple flux auto-réglable",
                 "VMC simple flux hygro A", "VMC simple flux hygro B", "VMC double flux"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Lancer le diagnostic", use_container_width=True)

    # ── Résultat ───────────────────────────────────────────────────────────────
    if submitted:
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

        with st.spinner("Analyse en cours…"):
            try:
                response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
                response.raise_for_status()
                result = response.json()
            except httpx.HTTPError as exc:
                st.error(f"❌ Erreur API : {exc}")
                return

        prediction = result["prediction"]
        probability = result["probability"]

        st.markdown("<div class='gold-bar'></div>", unsafe_allow_html=True)
        st.markdown("### Résultat du diagnostic")

        res_col1, res_col2, res_col3 = st.columns([2, 1, 1])

        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class='passoire'>
                    <div style='font-size:1.8rem;'>🔴</div>
                    <div style='font-family: Playfair Display, serif; font-size:1.3rem;
                                font-weight:700; color:#b52c2c; margin-top:0.4rem;'>
                        PASSOIRE THERMIQUE
                    </div>
                    <div style='color:#6b3333; font-size:0.9rem; margin-top:0.4rem;'>
                        Ce logement présente les caractéristiques d'une étiquette <strong>F ou G</strong>.
                        Une rénovation énergétique est fortement recommandée.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='non-passoire'>
                    <div style='font-size:1.8rem;'>🟢</div>
                    <div style='font-family: Playfair Display, serif; font-size:1.3rem;
                                font-weight:700; color:#1a7a4a; margin-top:0.4rem;'>
                        NON-PASSOIRE THERMIQUE
                    </div>
                    <div style='color:#1a4a33; font-size:0.9rem; margin-top:0.4rem;'>
                        Ce logement est conforme aux exigences énergétiques en vigueur
                        (étiquette <strong>A à E</strong>).
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.metric("Probabilité passoire", f"{probability:.1%}")

        with res_col3:
            st.metric("Confiance", f"{max(probability, 1 - probability):.1%}")

        # Barre de probabilité stylisée
        color = "#b52c2c" if probability > 0.5 else "#1a7a4a"
        st.markdown(f"""
        <div style='margin-top:1rem;'>
            <div style='display:flex; justify-content:space-between;
                        font-size:0.75rem; color:#6b7c8d; margin-bottom:0.3rem;'>
                <span>Non-passoire</span><span>Passoire</span>
            </div>
            <div style='background:#e8e4de; border-radius:4px; height:12px; position:relative;'>
                <div style='background:{color}; width:{probability*100:.1f}%;
                            height:12px; border-radius:4px; transition:width 0.5s;'></div>
            </div>
            <div style='text-align:center; font-size:0.8rem; color:#6b7c8d; margin-top:0.3rem;'>
                Score : {probability:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)