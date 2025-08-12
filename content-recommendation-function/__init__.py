import logging
import azure.functions as func
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
import json

# --- Chargement des données ---
def load_embeddings():
    with open("articles_embeddings.pickle", "rb") as f:
        return pickle.load(f)

def load_metadata():
    return pd.read_csv("articles_metadata.csv")

def load_clicks():
    clicks_folder = "clicks"
    all_clicks = []
    i = 0
    while True:
        filename = f"clicks_hour_{i:03d}.csv"
        file_path = os.path.join(clicks_folder, filename)
        if not os.path.exists(file_path):
            break
        df = pd.read_csv(file_path)
        all_clicks.append(df)
        i += 1

    if not all_clicks:
        raise RuntimeError("Aucun fichier de clic trouvé dans le dossier.")

    return pd.concat(all_clicks, ignore_index=True)

# Chargement au démarrage
embeddings = load_embeddings()
metadata = load_metadata()
clicks = load_clicks()
metadata["embedding"] = list(embeddings)

# --- Recommandations ---
def recommend_for_user(user_id, top_k=5):
    user_clicks = clicks[clicks["user_id"] == user_id]
    if user_clicks.empty:
        return []

    clicked_articles = user_clicks["click_article_id"].unique()
    clicked_embeddings = metadata[metadata["article_id"].isin(clicked_articles)]["embedding"].tolist()

    if not clicked_embeddings:
        return []

    user_profile = np.mean(clicked_embeddings, axis=0).reshape(1, -1)
    all_embeddings = np.vstack(metadata["embedding"].values)
    similarities = cosine_similarity(user_profile, all_embeddings)[0]

    metadata["similarity"] = similarities
    recommendations = metadata[~metadata["article_id"].isin(clicked_articles)]
    top_recos = recommendations.sort_values(by="similarity", ascending=False).head(top_k)

    return top_recos[["article_id", "similarity"]].to_dict(orient="records")

# --- Point d'entrée Azure ---
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Récupération du userId depuis la route
    user_id = req.route_params.get('userId')

    if not user_id:
        try:
            req_body = req.get_json()
            user_id = req_body.get('user_id')
        except ValueError:
            pass

    if not user_id:
        return func.HttpResponse(
            json.dumps({"error": "Missing user_id"}),
            status_code=400,
            mimetype="application/json"
        )
    
    try:
        user_id = int(user_id)
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "user_id must be an integer"}),
            status_code=400,
            mimetype="application/json"
        )

    recos = recommend_for_user(user_id)
    return func.HttpResponse(
        json.dumps({"user_id": user_id, "recommendations": recos}),
        status_code=200,
        mimetype="application/json"
    )