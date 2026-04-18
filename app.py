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
    page_title="SossoTrajet | Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === INJECTION CSS CUSTOM (UI/UX) ===
st.markdown("""
<style>
    /* Import police professionnelle (Inter) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Style des Cartes "KPI / Metrics" */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 5% 5% 5% 5%;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="metric-container"] label {
        color: #6B7280 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #1E3A8A !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
    }

    /* Style du Bouton Principal */
    .stButton>button {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: #ffffff !important;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.4);
        width: 100%;
        margin-top: 15px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.5);
    }

    /* Style des headers Custom */
    .dashboard-title {
        color: #111827;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: -15px;
        margin-top: -30px;
    }
    .dashboard-subtitle {
        color: #6B7280;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 30px;
    }
    .section-title {
        color: #1E3A8A;
        font-weight: 700;
        font-size: 1.5rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    
    /* Conteneur esthétique des inputs sidebar */
    .css-1544g2n {
        padding: 3rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# === CACHE ET MODELES ===
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv('SossoTrajet_Clean.csv')

@st.cache_resource(show_spinner=False)
def get_model(_df=None):
    model = XGBRegressor()
    model.load_model("sossoTrajet.json")
    return model

@st.cache_data(show_spinner=False)
def geocode_location(location_name, city):
    try:
        geolocator = Nominatim(user_agent="sossotrajet_app")
        loc = geolocator.geocode(f"{location_name}, {city}, Cameroun", timeout=5)
        if loc:
            return loc.latitude, loc.longitude
        return None
    except:
        return None

@st.cache_data(show_spinner=False)
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
    # === HEADER PRINCIPAL ===
    st.markdown("<div class='dashboard-title'>SossoTrajet Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='dashboard-subtitle'>Moteur de tarification VTC optimisé par XGBoost & Routing Géospatial Automatique</div>", unsafe_allow_html=True)
    
    df = load_data()
    model = get_model(df)
    
    RATIO_CONFORT = 1.30
    RATIO_CONFORT_PLUS = 1.727
    
    # === SIDEBAR (PANNEAU DE CONTROLE) ===
    with st.sidebar:
        st.markdown("<div style='text-align:center;'><h2 style='color:#111827;'>Panneau de Contrôle</h2><p style='color:#6B7280; font-size:14px;'>Configuration de l'Inférence</p></div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin-top:0px; margin-bottom:20px;'>", unsafe_allow_html=True)
        
        ville = st.selectbox("Zone Métropolitaine", ["Douala", "Yaounde"])
        
        df_ville = df[df['ville'] == ville]
        lieux = sorted(list(set(df_ville['depart'].dropna().unique()) | set(df_ville['arrivee'].dropna().unique())))
        
        depart_input = st.selectbox("Origine du trajet", [""] + lieux)
        arrivee_input = st.selectbox("Destination du trajet", [""] + lieux)
        
        st.markdown("<hr style='margin-top:20px; margin-bottom:20px;'>", unsafe_allow_html=True)
        
        heure_num = st.slider("Heure de Modélisation (0h-23h)", 0, 23, 14, help="Saisissez l'heure prévue du départ pour prendre en compte le trafic historique.")
        
        dispo_map = {"Fluide (Niv. 0)": 0, "Normal (Niv. 1)": 1, "Dense (Niv. 2)": 2, "Critique (Niv. 3)": 3}
        dispo_str = st.selectbox("Indice de Congestion", list(dispo_map.keys()))
        dispo_ordinal = dispo_map[dispo_str]
        
        # Placeholder pour contenir le bouton qui sera activé au clic
        execute_btn = False

    # === LOGIQUE & AFFICHAGE CENTRAL ===
    if depart_input and arrivee_input and depart_input != arrivee_input:
        
        with st.spinner("Calcul topologique de l'itinéraire en cours..."):
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
                route_stats = df_ville[(df_ville['depart'] == depart_input) & (df_ville['arrivee'] == arrivee_input)]
                if len(route_stats) > 0:
                    distance_km = route_stats['distance'].median()
                    duree_min = route_stats['duree_estimee'].median()
                else:
                    distance_km = 5.0
                    duree_min = 15.0

        # Affichage Infos Techniques dans la barre sous les inputs
        st.sidebar.markdown(f"<div style='background-color:#F3F4F6; padding:15px; border-radius:8px; margin-top:20px;'>"
                            f"<p style='margin:0; color:#4B5563; font-size:14px;'>Distance Routière : <b>{distance_km:.2f} km</b></p>"
                            f"<p style='margin:0; color:#4B5563; font-size:14px; margin-top:5px;'>Durée Estimée : <b>{duree_min:.0f} min</b></p></div>", unsafe_allow_html=True)

        # --- PREDICTION ML ---
        if st.sidebar.button("Exécuter la Modélisation ML"):
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
            
            # --- LAYOUT DASHBOARD: RESULTATS ---
            st.markdown("<div class='section-title'>Outputs Financiers (Prédictions XGBoost)</div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Gamme Éco", value=f"{p_eco} XAF", delta="Standard")
            with col2:
                st.metric(label="Gamme Confort", value=f"{p_confort} XAF", delta=f"+{int((RATIO_CONFORT-1)*100)}%")
            with col3:
                st.metric(label="Gamme Confort+", value=f"{p_confort_plus} XAF", delta=f"+{int((RATIO_CONFORT_PLUS-1)*100)}%")
        
        # --- CARTE (TOUJOURS VISIBLE) ---
        st.markdown("<div class='section-title'>Analyse Topologique & Trajet</div>", unsafe_allow_html=True)
        
        base_coords = [4.0511, 9.7679] if ville == "Douala" else [3.8480, 11.5021]
        m = folium.Map(location=base_coords, zoom_start=13, tiles="CartoDB Positron")
        
        bounds = []
        if coord_depart:
            folium.Marker(
                location=coord_depart,
                popup=f"Origine : {depart_input}",
                icon=folium.Icon(color="green", icon="play")
            ).add_to(m)
            bounds.append(coord_depart)
            
        if coord_arrivee:
            folium.Marker(
                location=coord_arrivee,
                popup=f"Destination : {arrivee_input}",
                icon=folium.Icon(color="red", icon="stop")
            ).add_to(m)
            bounds.append(coord_arrivee)
        
        if route_coords:
            folium_coords = [[lat, lon] for lon, lat in route_coords]
            from folium.plugins import AntPath
            AntPath(locations=folium_coords, color="#3B82F6", weight=5, opacity=0.8, delay=1000).add_to(m)
            m.fit_bounds(bounds)
        elif len(bounds) == 2:
            folium.PolyLine(bounds, color="#9CA3AF", weight=2, dash_array="5, 5").add_to(m)
            m.fit_bounds(bounds)
            
        st_folium(m, width=1100, height=500, key="sosso_map")

        # --- SECTION METHODOLOGIE & TRANSPARENCE ---
        st.markdown("<div class='section-title'>Méthodologie & Architecture du Modèle</div>", unsafe_allow_html=True)
        
        with st.expander("Consulter les détails techniques de l'algorithme SossoTrajet"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("""
                ### 🧠 Notre Cerveau Algorithmique
                Pour ce projet, nous avons implémenté **XGBoost (Extreme Gradient Boosting)**. 
                
                **Pourquoi ce choix ?**
                - **Données Tabulaires** : C'est le roi incontesté pour les données de transport structurées.
                - **Non-Linéarité** : Il capture les interactions complexes entre l'heure de pointe et la congestion que les modèles simples ignorent.
                - **Vitesse** : L'inférence se fait en moins de 10ms, idéal pour une application mobile.
                """)
                
            with col_b:
                st.markdown("""
                ### 📊 Ingénierie des Données
                - **Transformation Logarithmique** : Nous prédisons le `log(prix)`. Cela permet de minimiser l'**erreur relative**. (Une erreur de 200F est grave sur une course à 500F, mais négligeable sur 5000F).
                - **Features Clés** : Le modèle analyse la vélocité moyenne théorique, l'indice de congestion (0 à 3), la binarité Jour/Nuit et les spécificités propres à Douala et Yaoundé.
                """)
                
            st.info("💡 **Stratégie métier :** En raison de la volatilité des données réelles, nous prédisons la base 'Eco' de manière robuste, puis appliquons les ratios médians historiques pour dériver les tarifs 'Confort' (x1.3) et 'Confort+' (x1.73).")

    elif depart_input == arrivee_input and depart_input != "":
        st.error("Erreur d'intégrité : L'origine et la destination doivent impérativement différer pour l'inférence.")
    else:
        # Message par defaut pour inciter l'utilisateur
        st.markdown("<div style='text-align: center; margin-top: 10vh;'>"
                    "<img src='https://cdn-icons-png.flaticon.com/512/854/854878.png' height='80' style='opacity: 0.5;'/>"
                    "<h3 style='color: #6B7280; font-weight: 500;'>Sélectionnez des points géospatiaux dans le panneau <br>latéral pour débuter la prédiction tarifaire.</h3>"
                    "</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
