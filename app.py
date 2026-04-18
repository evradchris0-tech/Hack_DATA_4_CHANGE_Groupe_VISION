import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium

# Configuration de la page
st.set_page_config(page_title="SossoTrajet - Simulation", page_icon="🚖", layout="wide")

@st.cache_data
def load_data():
    # Chargement du csv nettoyé
    return pd.read_csv('SossoTrajet_Clean.csv')

@st.cache_resource
def get_model(_df):
    """
    Entraîne le modèle à la volée. 
    L'utilisation de st.cache_resource permet de ne le faire qu'une seule fois.
    """
    FEATURES = [
        'log_distance', 'log_duree', 'vitesse_kmh', 
        'is_pointe', 'is_nuit', 'dispo_ordinal', 
        'heure_num', 'ville_encoded', 'heure_plage_imputed'
    ]
    TARGET = 'log_prix_eco'
    
    df_model = _df.dropna(subset=[TARGET] + FEATURES)
    X = df_model[FEATURES]
    y = df_model[TARGET]
    
    # Paramètres par défaut de votre notebook
    model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    return model

@st.cache_data
def geocode_location(location_name, city):
    """
    Géocodage via OpenStreetMap Nominatim pour récupérer Lat/Lon
    """
    try:
        geolocator = Nominatim(user_agent="sossotrajet_app")
        # Ajout du Cameroun pour améliorer la précision
        loc = geolocator.geocode(f"{location_name}, {city}, Cameroun", timeout=5)
        if loc:
            return loc.latitude, loc.longitude
        return None
    except:
        return None

def main():
    st.title("🚖 SossoTrajet - Estimateur de Prix Yango")
    st.markdown("Bienvenue sur le simulateur participatif **SossoTrajet**. Estimez en toute transparence le coût de votre trajet à **Douala** ou **Yaoundé** avec notre modèle de Machine Learning.")
    
    df = load_data()
    model = get_model(df)
    
    # Ratios de prix (tirés de la médiane calculée dans votre notebook)
    RATIO_CONFORT = 1.30
    RATIO_CONFORT_PLUS = 1.727
    
    # ==========================
    # SIDEBAR: Entrées Utilisateur
    # ==========================
    st.sidebar.header("📍 Itinéraire")
    ville = st.sidebar.selectbox("Ville", ["Douala", "Yaoundé"])
    
    # Filtrer les lieux en fonction de la ville
    df_ville = df[df['ville'] == ville]
    lieux = sorted(list(set(df_ville['depart'].dropna().unique()) | set(df_ville['arrivee'].dropna().unique())))
    
    depart_input = st.sidebar.selectbox("Point de départ", lieux)
    arrivee_input = st.sidebar.selectbox("Point d'arrivée", lieux)
    
    st.sidebar.header("⏰ Conditions de route")
    heure_num = st.sidebar.slider("Heure de départ (0-23h)", 0, 23, 14)
    
    # Disponibilité du trafic
    dispo_map = {"Vert (Fluide)": 0, "Jaune (Normal)": 1, "Orange (Dense)": 2, "Rouge (Bouchons)": 3}
    dispo_str = st.sidebar.selectbox("Trafic / Disponibilité VTC", list(dispo_map.keys()))
    dispo_ordinal = dispo_map[dispo_str]
    
    # ==========================
    # LOGIQUE DISTANCE & DUREE
    # ==========================
    # On vérifie si ce trajet exact existe dans l'historique
    route_stats = df_ville[(df_ville['depart'] == depart_input) & (df_ville['arrivee'] == arrivee_input)]
    
    if len(route_stats) > 0:
        distance_km = route_stats['distance'].median()
        duree_min = route_stats['duree_estimee'].median()
        st.sidebar.success(f"Route existante en base : {distance_km:.1f} km, {duree_min:.0f} min historiquement.")
    else:
        st.sidebar.warning("Trajet inédit en base. Veuillez confirmer ou ajuster la distance et durée.")
        distance_km = st.sidebar.number_input("Distance (km)", min_value=0.1, max_value=100.0, value=5.0, step=0.5)
        duree_min = st.sidebar.number_input("Durée (minutes)", min_value=1.0, max_value=300.0, value=15.0, step=1.0)
    
    # ==========================
    # PREDICTION
    # ==========================
    if st.sidebar.button("🚀 Estimer le Prix", use_container_width=True):
        if depart_input == arrivee_input:
            st.error("Le point de départ et d'arrivée doivent être différents.")
        else:
            # Feature Engineering à la volée
            is_pointe = 1 if heure_num in [7, 8, 9, 17, 18, 19] else 0
            is_nuit = 1 if heure_num >= 21 or heure_num <= 5 else 0
            ville_encoded = 1 if ville == "Yaoundé" else 0
            
            vitesse = distance_km / (duree_min / 60) if duree_min > 0 else 20
            
            input_data = pd.DataFrame([{
                'log_distance': np.log1p(distance_km),
                'log_duree': np.log1p(duree_min),
                'vitesse_kmh': vitesse,
                'is_pointe': is_pointe,
                'is_nuit': is_nuit,
                'dispo_ordinal': dispo_ordinal,
                'heure_num': heure_num,
                'ville_encoded': ville_encoded,
                'heure_plage_imputed': 0
            }])
            
            # Prédiction ML
            log_pred = model.predict(input_data)[0]
            prix_eco = np.expm1(log_pred)
            
            # Formattage à la cinquantaine près pour être réaliste (ex: 1243 -> 1250)
            def round_fcfa(prix):
                return int(round(prix / 50) * 50)
                
            p_eco = round_fcfa(prix_eco)
            p_confort = round_fcfa(prix_eco * RATIO_CONFORT)
            p_confort_plus = round_fcfa(prix_eco * RATIO_CONFORT_PLUS)
            
            # Affichage Résultats
            st.subheader("💡 Estimations Tarifaires")
            col1, col2, col3 = st.columns(3)
            col1.metric("🌱 Yango Éco", f"{p_eco} FCFA", f"{distance_km:.1f} km")
            col2.metric("🛋️ Yango Confort", f"{p_confort} FCFA", f"{duree_min:.0f} min")
            col3.metric("✨ Yango Confort+", f"{p_confort_plus} FCFA", dispo_str.split()[0])
            
            # ==========================
            # CARTE OPENSTREETMAP
            # ==========================
            st.markdown("---")
            st.subheader(f"🗺️ Aperçu du trajet : {depart_input} ➔ {arrivee_input}")
            
            with st.spinner("Recherche des coordonnées OpenStreetMap..."):
                coord_depart = geocode_location(depart_input, ville)
                coord_arrivee = geocode_location(arrivee_input, ville)
            
            # Base map (Centré sur Douala ou Yaoundé)
            base_coords = [4.0511, 9.7679] if ville == "Douala" else [3.8480, 11.5021]
            m = folium.Map(location=base_coords, zoom_start=12, tiles="OpenStreetMap")
            
            bounds = []
            if coord_depart:
                folium.Marker(coord_depart, popup=f"Départ: {depart_input}", icon=folium.Icon(color="green")).add_to(m)
                bounds.append(coord_depart)
            if coord_arrivee:
                folium.Marker(coord_arrivee, popup=f"Arrivée: {arrivee_input}", icon=folium.Icon(color="red")).add_to(m)
                bounds.append(coord_arrivee)
                
            if len(bounds) == 2:
                # Tirer le trait
                folium.PolyLine(bounds, color="blue", weight=2.5, opacity=0.8).add_to(m)
                m.fit_bounds(bounds)
            elif len(bounds) == 0:
                st.info("ℹ️ Les lieux exacts n'ont pas pu être géocodés sur OpenStreetMap. La carte reste centrée sur la ville.")
                
            st_folium(m, width=900, height=450)

if __name__ == "__main__":
    main()
