import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import requests

# === CONFIGURATION GLOBALE ===
st.set_page_config(
    page_title="SossoTrajet | Simulation Tarifaire",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CACHE ET MODELES ===
@st.cache_data
def load_data():
    return pd.read_csv('SossoTrajet_Clean.csv')

@st.cache_resource
def get_model(_df=None):
    model = XGBRegressor()
    model.load_model("sossoTrajet.json")
    return model

@st.cache_data
def geocode_location(location_name, city):
    try:
        geolocator = Nominatim(user_agent="sossotrajet_app")
        loc = geolocator.geocode(f"{location_name}, {city}, Cameroun", timeout=5)
        if loc:
            return loc.latitude, loc.longitude
        return None
    except:
        return None

@st.cache_data
def get_osrm_route(lat1, lon1, lat2, lon2):
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
        res = requests.get(url, timeout=5).json()
        if res.get('code') == 'Ok':
            route = res['routes'][0]
            distance = route['distance'] / 1000.0
            duration = route['duration'] / 60.0
            coords = route['geometry']['coordinates']
            return distance, duration, coords
    except Exception:
        pass
    return None, None, None

def main():
    # === HEADER ===
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>SossoTrajet : Inférence Tarifaire</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6B7280; font-size: 1.1rem;'>Plateforme analytique de prédiction des tarifs VTC basée sur l'algorithme XGBoost et la cartographie OSRM.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    df = load_data()
    model = get_model(df)
    
    RATIO_CONFORT = 1.30
    RATIO_CONFORT_PLUS = 1.727
    
    # === SIDEBAR ===
    st.sidebar.markdown("<h3 style='color: #1E3A8A;'>Paramètres Géospatiaux</h3>", unsafe_allow_html=True)
    ville = st.sidebar.selectbox("Zone métropolitaine", ["Douala", "Yaounde"])
    
    df_ville = df[df['ville'] == ville]
    lieux = sorted(list(set(df_ville['depart'].dropna().unique()) | set(df_ville['arrivee'].dropna().unique())))
    
    depart_input = st.sidebar.selectbox("Origine du trajet", [""] + lieux)
    arrivee_input = st.sidebar.selectbox("Destination du trajet", [""] + lieux)
    
    st.sidebar.markdown("<br><h3 style='color: #1E3A8A;'>Contexte Temporel & Trafic</h3>", unsafe_allow_html=True)
    heure_num = st.sidebar.slider("Heure estimée de départ (0h-23h)", 0, 23, 14)
    
    dispo_map = {"Fluide (Niveau 0)": 0, "Normal (Niveau 1)": 1, "Dense (Niveau 2)": 2, "Bouchons (Niveau 3)": 3}
    dispo_str = st.sidebar.selectbox("Indice de Congestion", list(dispo_map.keys()))
    dispo_ordinal = dispo_map[dispo_str]
    
    # === LOGIQUE PRINCIPALE ===
    if depart_input and arrivee_input and depart_input != arrivee_input:
        
        coord_depart = geocode_location(depart_input, ville)
        coord_arrivee = geocode_location(arrivee_input, ville)
        
        distance_km = None
        duree_min = None
        route_coords = None
        
        if coord_depart and coord_arrivee:
            distance_km, duree_min, route_coords = get_osrm_route(
                coord_depart[0], coord_depart[1], 
                coord_arrivee[0], coord_arrivee[1]
            )
            
        if distance_km is None or duree_min is None:
            # Fallback historique
            route_stats = df_ville[(df_ville['depart'] == depart_input) & (df_ville['arrivee'] == arrivee_input)]
            if len(route_stats) > 0:
                distance_km = route_stats['distance'].median()
                duree_min = route_stats['duree_estimee'].median()
                st.info("Données GPS inaccessibles. Utilisation des médianes historiques du dataset.")
            else:
                distance_km = 5.0
                duree_min = 15.0
                st.warning("Aucune donnée topologique ou historique disponible. Valeurs par défaut appliquées.")

        st.sidebar.markdown("---")
        st.sidebar.markdown(f"<span style='color:#4B5563'><b>Distance réseau :</b> {distance_km:.2f} km</span>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<span style='color:#4B5563'><b>Durée estimée :</b> {duree_min:.0f} min</span>", unsafe_allow_html=True)
        st.sidebar.markdown("<br>", unsafe_allow_html=True)

        if st.sidebar.button("Exécuter la prédiction globale", use_container_width=True):
            
            # Feature Engineering
            is_pointe = 1 if heure_num in [7, 8, 9, 17, 18, 19] else 0
            is_nuit = 1 if heure_num >= 21 or heure_num <= 5 else 0
            ville_encoded = 1 if ville == "Yaounde" else 0
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
            
            # Inférence
            log_pred = model.predict(input_data)[0]
            prix_eco = np.expm1(log_pred)
            
            def round_fcfa(prix):
                return int(round(prix / 50) * 50)
                
            p_eco = round_fcfa(prix_eco)
            p_confort = round_fcfa(prix_eco * RATIO_CONFORT)
            p_confort_plus = round_fcfa(prix_eco * RATIO_CONFORT_PLUS)
            
            # --- AFFICHAGE RESULTATS ---
            st.markdown("<h3 style='color: #111827; margin-bottom: 20px;'>Résultats de l'Inférence (XGBoost)</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Catégorie Éco", value=f"{p_eco} FCFA", delta="Formule de base", delta_color="off")
            with col2:
                st.metric(label="Catégorie Confort", value=f"{p_confort} FCFA", delta=f"+{int((RATIO_CONFORT-1)*100)}%", delta_color="normal")
            with col3:
                st.metric(label="Catégorie Confort+", value=f"{p_confort_plus} FCFA", delta=f"+{int((RATIO_CONFORT_PLUS-1)*100)}%", delta_color="normal")
            
            st.markdown("---")
            
            # --- CARTE TOPOLOGIQUE ---
            st.markdown(f"<h3 style='color: #111827;'>Analyse Géospatiale OSRM</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #6B7280;'>Visualisation du trajet optimal entre <b>{depart_input}</b> et <b>{arrivee_input}</b>.</p>", unsafe_allow_html=True)
            
            base_coords = [4.0511, 9.7679] if ville == "Douala" else [3.8480, 11.5021]
            m = folium.Map(location=base_coords, zoom_start=13, tiles="CartoDB positron")
            
            bounds = []
            if coord_depart:
                # Marqueur Point A (Dark blue)
                folium.CircleMarker(
                    location=coord_depart,
                    radius=7,
                    popup=f"Origine : {depart_input}",
                    color="#1E3A8A",
                    fill=True,
                    fill_color="#1E3A8A"
                ).add_to(m)
                bounds.append(coord_depart)
                
            if coord_arrivee:
                # Marqueur Point B (Red)
                folium.CircleMarker(
                    location=coord_arrivee,
                    radius=7,
                    popup=f"Destination : {arrivee_input}",
                    color="#DC2626",
                    fill=True,
                    fill_color="#DC2626"
                ).add_to(m)
                bounds.append(coord_arrivee)
            
            if route_coords:
                folium_coords = [[lat, lon] for lon, lat in route_coords]
                folium.PolyLine(folium_coords, color="#4F46E5", weight=4, opacity=0.8).add_to(m)
                m.fit_bounds(bounds)
            elif len(bounds) == 2:
                folium.PolyLine(bounds, color="#9CA3AF", weight=2, dash_array="5, 5").add_to(m)
                m.fit_bounds(bounds)
                
            st_folium(m, width=1000, height=450, returned_objects=[])
            
    elif depart_input == arrivee_input and depart_input != "":
        st.error("Règle métier ignorée : L'origine et la destination ne peuvent être identiques.")
    else:
        st.markdown("<br><p style='color: #6B7280; font-style: italic;'>Veuillez paramétrer l'origine et la destination dans le panneau interactif à gauche pour initialiser l'algorithme.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
