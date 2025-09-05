import logging
import azure.functions as func
import pandas as pd
import numpy as np
import json
import os
import pickle
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
from azure.storage.blob import BlobServiceClient
from utils import get_file_path

# --- Chargement metadata depuis LFS / file ---
file_path = get_file_path("articles_metadata.csv")
metadata = pd.read_csv(file_path)

# --- Chargement des fichiers clicks depuis Blob Storage ---
def load_clicks_from_blob():
    """
    Charge tous les fichiers CSV du dossier 'clicks/' accessible via get_file_path()
    et les concatène en un seul DataFrame.
    Fonctionne aussi bien en local qu'avec le montage /file sur Azure.
    """
    # Chemin du dossier clicks
    clicks_folder = get_file_path("clicks")  # /file/clicks sur Azure ou ./file/clicks en local
    print(f"🔍 Vérification du dossier clicks : {clicks_folder}")

    if not os.path.exists(clicks_folder) or not os.path.isdir(clicks_folder):
        raise FileNotFoundError(f"Le dossier {clicks_folder} est introuvable sur Azure ou en local.")

    all_clicks = []
    for filename in os.listdir(clicks_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(clicks_folder, filename)
            print(f"Lecture du fichier : {file_path}")
            df = pd.read_csv(file_path)
            all_clicks.append(df)

    if not all_clicks:
        raise RuntimeError("Aucun fichier de clic trouvé dans le dossier clicks/.")

    concatenated = pd.concat(all_clicks, ignore_index=True)
    print(f"Nombre total de clics chargés : {len(concatenated)}")
    return concatenated

clicks = load_clicks_from_blob()

# --- Construction matrice utilisateur-article ---
user_ids = clicks["user_id"].astype("category")
article_ids = clicks["click_article_id"].astype("category")
user_map = dict(enumerate(user_ids.cat.categories))
article_map = dict(enumerate(article_ids.cat.categories))

matrix = coo_matrix(
    (np.ones(len(clicks)), (user_ids.cat.codes, article_ids.cat.codes))
).tocsr()

# --- Charger modèle implicit pré-entraîné depuis /file ---
with open(get_file_path("implicit_model.pkl"), "rb") as f:
    model: AlternatingLeastSquares = pickle.load(f)

# --- Recommandations ---
def recommend_for_user(user_id, top_k=5):
    if user_id not in user_ids.cat.categories:
        return []
    user_index = list(user_ids.cat.categories).index(user_id)
    rec_indices, scores = model.recommend(user_index, matrix[user_index], N=top_k)
    return [{"article_id": int(article_map[i]), "score": float(s)} for i, s in zip(rec_indices, scores)]

# --- Point d'entrée Azure ---
def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        user_id = int(req.route_params.get("userId", -1))
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Invalid userId"}),
            status_code=400,
            mimetype="application/json"
        )

    recos = recommend_for_user(user_id)
    return func.HttpResponse(
        json.dumps({"method": "implicit", "user_id": user_id, "recommendations": recos}),
        mimetype="application/json"
    )
