import json

ipynb_path = r'd:\ING 5\ML\venv\ML\hackDATAchange\SossoTrajet_Modelling.ipynb'

with open(ipynb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%pip install xgboost streamlit folium streamlit-folium geopy pandas numpy"
    ]
}

# Insert at the top
nb['cells'].insert(0, new_cell)

with open(ipynb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Cell added successfully!")
