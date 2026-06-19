"""Frontend Streamlit — Prédiction des passoires thermiques DPE.

Lancement : streamlit run frontend/app.py
            API_URL=http://localhost:8000 streamlit run frontend/app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="DPE Passoire — Louis Laurent",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Charte graphique immobilière ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --navy:   #0d2137;
    --navy2:  #163352;
    --gold:   #c9a84c;
    --gold2:  #e8c97a;
    --cream:  #f7f4ef;
    --white:  #ffffff;
    --text:   #1a2a3a;
    --muted:  #6b7c8d;
    --green:  #1a7a4a;
    --red:    #b52c2c;
    --border: #ddd8ce;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--cream);
    color: var(--text);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--navy) !important;
    border-right: 3px solid var(--gold);
}
section[data-testid="stSidebar"] * {
    color: var(--white) !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}
section[data-testid="stSidebar"] a {
    color: var(--gold2) !important;
    text-decoration: none;
}
section[data-testid="stSidebar"] a:hover {
    color: var(--gold) !important;
    text-decoration: underline;
}

/* Titres */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--navy) !important;
}

/* Boutons */
div.stButton > button {
    background-color: var(--navy) !important;
    color: var(--white) !important;
    border: 2px solid var(--gold) !important;
    border-radius: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    background-color: var(--gold) !important;
    color: var(--navy) !important;
}

/* Cards métriques */
div[data-testid="metric-container"] {
    background: var(--white);
    border: 1px solid var(--border);
    border-top: 3px solid var(--gold);
    border-radius: 4px;
    padding: 1rem 1.2rem;
}

/* Inputs */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select,
div[data-baseweb="select"] {
    border-radius: 4px !important;
    border-color: var(--border) !important;
}

/* Divider */
hr {
    border-color: var(--border) !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}

.gold-bar {
    height: 3px;
    background: linear-gradient(90deg, var(--gold), var(--gold2), var(--gold));
    border-radius: 2px;
    margin: 0.5rem 0 2rem 0;
}

.card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
}

.badge {
    display: inline-block;
    background: var(--navy);
    color: var(--gold) !important;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    padding: 0.2rem 0.8rem;
    border-radius: 2px;
    text-transform: uppercase;
}

.passoire {
    background: #fff0f0;
    border-left: 5px solid var(--red);
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
}
.non-passoire {
    background: #f0fff6;
    border-left: 5px solid var(--green);
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Navigation sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 1rem 0;'>
        <div style='font-family: Playfair Display, serif; font-size:1.4rem; font-weight:700; color:#c9a84c;'>DPE Passoire</div>
        <div style='font-size:0.75rem; letter-spacing:0.12em; color:#aab8c6; margin-top:0.2rem;'>DIAGNOSTIC ÉNERGÉTIQUE</div>
    </div>
    <div style='height:2px; background:linear-gradient(90deg,#c9a84c,#e8c97a,#c9a84c); margin-bottom:1.5rem;'></div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Accueil", "🔍  Prédiction", "📊  Métriques & Outils"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style='position:absolute; bottom:2rem; left:0; right:0; padding:0 1.5rem;'>
        <div style='font-size:0.7rem; color:#6b8099; letter-spacing:0.08em; margin-bottom:0.5rem;'>LIENS RAPIDES</div>
        <a href='http://localhost:5000' target='_blank'>📈 MLflow</a><br><br>
        <a href='http://localhost:8080' target='_blank'>🔄 Airflow</a><br><br>
        <a href='http://localhost:8000/docs' target='_blank'>⚡ API Docs</a><br><br>
        <a href='https://github.com/username/dpe-etiquette-prediction' target='_blank'>💻 GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# ── Routing ────────────────────────────────────────────────────────────────────
if page == "🏠  Accueil":
    from pages_content.home import render
elif page == "🔍  Prédiction":
    from pages_content.predict import render
else:
    from pages_content.metrics import render

render()