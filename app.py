import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import requests

# Configuration de la page
st.set_page_config(page_title="SossoTrajet - Simulation", page_icon="🚖", layout="wide")

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
    """Calcule la distance, duree et la route GEOJSON via OSRM"""
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
        res = requests.get(url, timeout=5).json()
        if res.get('code') == 'Ok':
            route = res['routes'][0]
            distance = route['distance'] / 1000.0
            duration = route['duration'] / 60.0
            coords = route['geometry']['coordinates'] # [lon, lat] list
            return distance, duration, coords
    except Exception as e:
        pass
    return None, None, None

def main():
    st.title("🚖 SossoTrajet - Estimateur de Prix Yango")
    st.markdown("Bienvenue sur le simulateur **SossoTrajet**. Indiquez vos points de départ et d'arrivée, nous calculons **automatiquement** l'itinéraire et estimons le prix.")
    
    df = load_data()
    model = get_model(df)
    
    # Ratios de prix
    RATIO_CONFORT = 1.30
    RATIO_CONFORT_PLUS = 1.727
    
    st.sidebar.header("📍 Itinéraire")
    ville = st.sidebar.selectbox("Ville", ["Douala", "Yaounde"])
    
    df_ville = df[df['ville'] == ville]
    lieux = sorted(list(set(df_ville['depart'].dropna().unique()) | set(df_ville['arrivee'].dropna().unique())))
    
    depart_input = st.sidebar.selectbox("Point de départ", [""] + lieux)
    arrivee_input = st.sidebar.selectbox("Point d'arrivée", [""] + lieux)
    
    st.sidebar.header("⏰ Conditions de route")
    heure_num = st.sidebar.slider("Heure de départ (0-23h)", 0, 23, 14)
    
    dispo_map = {"Vert (Fluide)": 0, "Jaune (Normal)": 1, "Orange (Dense)": 2, "Rouge (Bouchons)": 3}
    dispo_str = st.sidebar.selectbox("Trafic / Disponibilité VTC", list(dispo_map.keys()))
    dispo_ordinal = dispo_map[dispo_str]
    
    if depart_input and arrivee_input and depart_input != arrivee_input:
        
        # Geocodage et OSRM
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
            
        # Si OSRM ou geocodage échoue, on cherche dans l'historique de notre Dataset !
        if distance_km is None or duree_min is None:
            st.warning("⚠️ Calcul OpenStreetMap indisponible pour ces lieux exacts. Utilisation des moyennes de notre base historique.")
            route_stats = df_ville[(df_ville['depart'] == depart_input) & (df_ville['arrivee'] == arrivee_input)]
            if len(route_stats) > 0:
                distance_km = route_stats['distance'].median()
                duree_min = route_stats['duree_estimee'].median()
            else:
                # Fallback absolut
                distance_km = 5.0
                duree_min = 15.0
                st.error("Aucune donnée historique trouvée. Affichage par défaut (5km).")

        # Affichage temps / distance calculés auto
        st.sidebar.markdown(f"**Distance calculée :** {distance_km:.1f} km")
        st.sidebar.markdown(f"**Durée estimée :** {duree_min:.0f} min")

        if st.sidebar.button("🚀 Estimer le Prix", use_container_width=True):
            # --- FEATURE ENGINEERING ---
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
            
            # --- ML PREDICTION ---
            log_pred = model.predict(input_data)[0]
            prix_eco = np.expm1(log_pred)
            
            def round_fcfa(prix):
                return int(round(prix / 50) * 50)
                
            p_eco = round_fcfa(prix_eco)
            p_confort = round_fcfa(prix_eco * RATIO_CONFORT)
            p_confort_plus = round_fcfa(prix_eco * RATIO_CONFORT_PLUS)
            
            # --- RESULTATS ---
            st.subheader("💡 Estimations Tarifaires")
            col1, col2, col3 = st.columns(3)
            col1.metric("🌱 Yango Éco", f"{p_eco} FCFA", f"{distance_km:.1f} km")
            col2.metric("🛋️ Yango Confort", f"{p_confort} FCFA", f"{duree_min:.0f} min")
            col3.metric("✨ Yango Confort+", f"{p_confort_plus} FCFA", dispo_str.split()[0])
            
            # --- MAP FOLIUM ---
            st.markdown("---")
            st.subheader(f"🗺️ Navigation OSRM : {depart_input} ➔ {arrivee_input}")
            
            base_coords = [4.0511, 9.7679] if ville == "Douala" else [3.8480, 11.5021]
            m = folium.Map(location=base_coords, zoom_start=13, tiles="Cartodb Positron")
            
            bounds = []
            if coord_depart:
                folium.Marker(coord_depart, popup=f"Départ: {depart_input}", icon=folium.Icon(color="green")).add_to(m)
                bounds.append(coord_depart)
            if coord_arrivee:
                folium.Marker(coord_arrivee, popup=f"Arrivée: {arrivee_input}", icon=folium.Icon(color="red")).add_to(m)
                bounds.append(coord_arrivee)
            
            # Draw real road polyline if OSRM gave it
            if route_coords:
                # route_coords is [lon, lat], folium needs [lat, lon]
                folium_coords = [[lat, lon] for lon, lat in route_coords]
                folium.PolyLine(folium_coords, color="blue", weight=4, opacity=0.8).add_to(m)
                m.fit_bounds(bounds)
            elif len(bounds) == 2:
                # Fallback straight line
                folium.PolyLine(bounds, color="gray", weight=2, dash_array="5, 5").add_to(m)
                m.fit_bounds(bounds)
                
            st_folium(m, width=900, height=450)
            
    elif depart_input == arrivee_input and depart_input != "":
        st.error("Le point de départ et d'arrivée doivent être différents.")
    else:
        st.info("👈 Veuillez d'abord choisir un point de départ et un point d'arrivée valides à gauche.")

if __name__ == "__main__":
    main()
