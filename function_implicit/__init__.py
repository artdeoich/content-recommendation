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
    connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    container_name = "file"
    clicks_prefix = "clicks/"  # sous-dossier dans le container

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    all_clicks = []
    for blob in container_client.list_blobs(name_starts_with=clicks_prefix):
        if blob.name.endswith(".csv"):
            print(f"➡️ Lecture du blob : {blob.name}")
            blob_client = container_client.get_blob_client(blob)
            data = blob_client.download_blob().readall()
            df = pd.read_csv(io.BytesIO(data))
            all_clicks.append(df)

    if not all_clicks:
        raise RuntimeError(f"Aucun fichier CSV trouvé dans '{clicks_prefix}' du container '{container_name}'.")

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
