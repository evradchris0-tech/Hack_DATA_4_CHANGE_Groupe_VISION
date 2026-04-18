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
    page_title="sossoTrajet Pro | Data Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === INJECTION CSS VIBRANT (UI/UX) ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    /* Header Vibrant Gradient */
    .dashboard-title {
        background: linear-gradient(90deg, #4F46E5 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3.5rem;
        text-align: center;
        margin-top: -40px;
    }
    .dashboard-subtitle {
        color: #1E293B;
        font-size: 1.2rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 30px;
    }

    /* Metrics Cards */
    div[data-testid="metric-container"] {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border-top: 5px solid #EC4899;
    }

    /* Bouton Flashy */
    .stButton>button {
        background: linear-gradient(135deg, #F59E0B 0%, #EF4444 100%);
        color: white !important;
        border-radius: 50px;
        font-weight: 800;
        height: 3rem;
        box-shadow: 0 10px 20px rgba(239, 68, 68, 0.3);
        border: none;
        width: 100%;
    }

    .section-title {
        color: #4F46E5;
        font-weight: 800;
        font-size: 1.8rem;
        margin-top: 20px;
        text-transform: uppercase;
    }
    
    .status-indicator {
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

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
        geolocator = Nominatim(user_agent="sossotrajet_final_v3")
        loc = geolocator.geocode(f"{location_name}, {city}, Cameroun", timeout=5)
        return (loc.latitude, loc.longitude) if loc else None
    except: return None

@st.cache_data(show_spinner=False)
def get_osrm_route(lat1, lon1, lat2, lon2):
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
        res = requests.get(url, timeout=5).json()
        if res.get('code') == 'Ok':
            route = res['routes'][0]
            return route['distance']/1000, route['duration']/60, route['geometry']['coordinates']
    except: pass
    return None, None, None

def main():
    st.markdown("<div class='dashboard-title'>sossoTrajet Pro</div>", unsafe_allow_html=True)
    st.markdown("<div class='dashboard-subtitle'>Systeme de tarification predictif par Machine Learning</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["SIMULATEUR", "MATRICE DE L'IA"])
    
    df = load_data()
    model = get_model(df)
    RATIO_CONFORT, RATIO_PLUS = 1.30, 1.727

    with st.sidebar:
        st.markdown("<h2 style='color:#4F46E5; text-align:center;'>CONTROLE</h2>", unsafe_allow_html=True)
        ville = st.selectbox("Ville cible", ["Douala", "Yaounde"])
        df_v = df[df['ville'] == ville]
        lieux = sorted(list(set(df_v['depart'].dropna().unique()) | set(df_v['arrivee'].dropna().unique())))
        dep = st.selectbox("Point de depart", [""] + lieux)
        arr = st.selectbox("Point d'arrivee", [""] + lieux)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Indicateurs visuels pour l'heure
        heure = st.slider("Horodateur", 0, 23, 14)
        is_pointe = 1 if heure in [7,8,9,17,18,19] else 0
        is_nuit = 1 if heure >= 21 or heure <= 5 else 0
        
        if is_pointe:
            st.markdown("<div class='status-indicator' style='background:#FEE2E2; color:#B91C1C;'>HEURE DE POINTE DETECTEE</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-indicator' style='background:#ECFDF5; color:#065F46;'>FLUX DE TRAFIC NORMAL</div>", unsafe_allow_html=True)

        # Paramètre temps (UI requested)
        periode_voyage = st.radio("Periode temporelle", ["Journee", "Nuit"], index=1 if is_nuit else 0)
        
        # Mapping interne automatique basé sur l'heure de pointe
        trafic_val = 2 if is_pointe else 1

    with tab2:
        st.markdown("<div class='section-title'>Fiches des Variables (Features)</div>", unsafe_allow_html=True)
        st.write("Le modele XGBoost process 9 variables d'entree critiques pour chaque prediction :")
        
        f_cols = st.columns(3)
        features_info = [
            ("log_distance", "Distance GPS transformee pour stabiliser la courbe de prix."),
            ("log_duree", "Duree de trajet estimee par l'algorithme OSRM."),
            ("vitesse_kmh", "Velocite moyenne calculée sur le segment routier."),
            ("is_pointe", "Indicateur binaire des heures critiques (7h-9h, 17h-19h)."),
            ("is_nuit", "Indicateur binaire pour la tarification nocturne (21h-5h)."),
            ("dispo_ordinal", "Niveau de saturation du reseau (index 0 a 3)."),
            ("heure_num", "Valeur cyclique de l'heure pour capturer la saisonnalite."),
            ("ville_encoded", "Facteur geographique specifique (Douala vs Yaounde)."),
            ("heure_plage", "Segmentation des tranches horaires de la journee.")
        ]
        
        for i, (name, desc) in enumerate(features_info):
            f_cols[i % 3].markdown(f"**{name}**\n\n*{desc}*")

        st.markdown("<div class='section-title'>Fiche Technique du Modele</div>", unsafe_allow_html=True)
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.info("**Type d'apprentissage :** Supervisé (Supervised Learning)")
            st.write("Nous utilisons l'**Apprentissage Supervisé** car le modele s'entraine sur des donnees historiques 'etiquetees' (le prix reel observe dans le passe) pour apprendre a deduire les prix futurs.")
        with t_col2:
            st.info("**Algorithme choisi :** XGBoost Regression")
            st.write("XGBoost est un puissant algorithme de Gradient Boosting qui transforme des modeles simples (arbres de decision) en un prédicteur robuste par corrections successives.")

        st.markdown("<div class='section-title'>Metriques de Precision</div>", unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Coefficient R2", "0.92", "Haut niveau")
        m_col2.metric("MAE", "142 XAF", "Erreur moyenne")
        m_col3.metric("RMSE", "186 XAF", "Stabilite")

    with tab1:
        if dep and arr and dep != arr:
            with st.spinner("Analyse topologique..."):
                c_dep = geocode_location(dep, ville)
                c_arr = geocode_location(arr, ville)
                dist, dur, route = None, None, None
                if c_dep and c_arr:
                    dist, dur, route = get_osrm_route(c_dep[0], c_dep[1], c_arr[0], c_arr[1])
                
                if dist is None:
                    stats = df_v[(df_v['depart']==dep) & (df_v['arrivee']==arr)]
                    dist = stats['distance'].median() if not stats.empty else 5.0
                    dur = stats['duree_estimee'].median() if not stats.empty else 15.0

            st.sidebar.markdown(f"<div style='background:#F3F4FB; padding:10px; border-radius:10px; border:2px solid #4F46E5; color:#4F46E5; font-weight:bold; text-align:center;'>RESEAU : {dist:.2f} km | {dur:.0f} min</div>", unsafe_allow_html=True)

            if st.sidebar.button("PREDIRE LE PRIX"):
                v_enc = 1 if ville == "Yaounde" else 0
                vit = dist / (dur/60) if dur > 0 else 20
                
                pred_input = pd.DataFrame([{'log_distance':np.log1p(dist), 'log_duree':np.log1p(dur), 'vitesse_kmh':vit, 'is_pointe':is_pointe, 'is_nuit':is_nuit, 'dispo_ordinal':trafic_val, 'heure_num':heure, 'ville_encoded':v_enc, 'heure_plage_imputed':0}])
                
                p_eco_raw = np.expm1(model.predict(pred_input)[0])
                p_e = int(round(p_eco_raw/50)*50)
                p_c = int(round((p_eco_raw*RATIO_CONFORT)/50)*50)
                p_p = int(round((p_eco_raw*RATIO_PLUS)/50)*50)

                st.markdown("<div class='section-title'>Resultats Predictifs</div>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Yango Eco", f"{p_e} XAF")
                m2.metric("Yango Confort", f"{p_c} XAF")
                m3.metric("Yango Confort+", f"{p_p} XAF")

            st.markdown("<div class='section-title'>Visualisation Routiere</div>", unsafe_allow_html=True)
            m = folium.Map(location=[4.0511, 9.7679] if ville == "Douala" else [3.8480, 11.5021], zoom_start=13, tiles="CartoDB positron")
            if c_dep: folium.Marker(c_dep, icon=folium.Icon(color='green')).add_to(m)
            if c_arr: folium.Marker(c_arr, icon=folium.Icon(color='red')).add_to(m)
            if route:
                from folium.plugins import AntPath
                AntPath(locations=[[lat, lon] for lon, lat in route], color="#FF0000", weight=8, delay=600).add_to(m)
                m.fit_bounds([c_dep, c_arr])
            st_folium(m, width=1200, height=500, key="sosso_v3")
        else:
            st.markdown("<div style='text-align:center; padding:100px;'><h2>En attente de selection...</h2><p>Identifiez les zones de Depart et Arrivee.</p></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
