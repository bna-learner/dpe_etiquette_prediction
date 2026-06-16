"""Construction du pre-processing."""
from __future__ import annotations
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def na_handle() -> pd.DataFrame:
    data["nombre_niveau_logement"] = data["nombre_niveau_logement"].fillna(
    data["nombre_niveau_logement"].median()
    )
    data = data.dropna()
    return data

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )




"""Construction du pre-processing."""
from __future__ import annotations
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

# On sépare les catégories pour appliquer le bon traitement
# (À adapter ou à importer depuis ton src.config)

ORDINAL_FEATURES = [
    "periode_construction",
    "qualite_isolation_enveloppe",
    "qualite_isolation_murs",
    "qualite_isolation_menuiseries"
]

NOMINAL_FEATURES = [
    "type_batiment", 
    "type_energie_principale_chauffage"
]


def na_handle(data: pd.DataFrame) -> pd.DataFrame:
    """Gère les valeurs manquantes sur le DataFrame."""
    df = data.copy()

    if "nombre_niveau_logement" in df.columns:
        df["nombre_niveau_logement"] = df["nombre_niveau_logement"].fillna(
            df["nombre_niveau_logement"].median()
        )

    df = df.dropna()
    
    return df


def build_preprocessor() -> ColumnTransformer:
    """Construit le pipeline de pré-traitement des données."""
    
    # 1. Ordre pour la période de construction
    ordre_periode = ['avant 1948', '1948-1974', '2006-2012', '2013-2021', 'après 2021']
    
    # 2. Ordre pour les trois variables d'isolation
    ordre_isolation = ['insuffisante', 'moyenne', 'bonne', 'très bonne']
    
    # On définit l'OrdinalEncoder avec les listes d'ordres exactes
    # handle_unknown="use_encoded_value" permet de ne pas crash si une nouvelle catégorie apparaît
    ordinal_transformer = OrdinalEncoder(
        categories=[ordre_periode, ordre_isolation, ordre_isolation, ordre_isolation],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("ord", ordinal_transformer, ORDINAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), NOMINAL_FEATURES),
        ],
        remainder="passthrough"
    )