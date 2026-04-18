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
        "import joblib\n",
        "# Sauvegarde du modele XGBoost au format Pikl (.pkl)\n",
        "nom_projet = 'sossoTrajet'\n",
        "joblib.dump(best_model, f'{nom_projet}.pkl')\n",
        "print(f\"Le modèle a bien été sauvegardé sous le nom : {nom_projet}.pkl\")"
    ]
}

# Insert at the bottom
nb['cells'].append(new_cell)

with open(ipynb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Cellule pkl ajoutee avec succes.")
