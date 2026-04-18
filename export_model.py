import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import json

print("1. Chargement des donnees...")
df = pd.read_csv('SossoTrajet_Clean.csv')
FEATURES = [
    'log_distance', 'log_duree', 'vitesse_kmh', 
    'is_pointe', 'is_nuit', 'dispo_ordinal', 
    'heure_num', 'ville_encoded', 'heure_plage_imputed'
]
TARGET = 'log_prix_eco'

df_model = df.dropna(subset=[TARGET] + FEATURES)
X = df_model[FEATURES]
y = df_model[TARGET]

print("2. Entrainement du modele XGBoost...")
# Paramètres optimaux
model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X, y)

print("3. Sauvegarde physique du modele...")
# Sauvegarde native XGBoost en JSON (idéal pour l'API et Streamlit)
model_path = "sosso_trajet_xgboost.json"
model.save_model(model_path)

# Sauvegarde d'un fichier de configuration avec les métadonnées (features attendues et ratios)
config_path = "sosso_trajet_config.json"
config = {
    "features": FEATURES,
    "ratios": {
        "confort": 1.30,
        "confort_plus": 1.727
    }
}
with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

print(f"✅ Modele sauvegarde avec succes sous '{model_path}' ! Configuration dans '{config_path}'.")
