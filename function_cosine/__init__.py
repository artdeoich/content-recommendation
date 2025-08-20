import logging
import azure.functions as func
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import pickle

# --- Chargement des données ---
BASE_DIR = os.path.dirname(__file__)+'\\..\\'

with open(os.path.join(BASE_DIR, "file", "articles_embeddings.pickle"), "rb") as f:
    embeddings = pickle.load(f)

metadata = pd.read_csv(os.path.join(BASE_DIR, "articles_metadata.csv"))

clicks_dir = os.path.join(BASE_DIR, "clicks")
clicks = pd.concat(
    [pd.read_csv(os.path.join(clicks_dir, f)) for f in os.listdir(clicks_dir)],
    ignore_index=True
)

metadata["embedding"] = list(embeddings)


def recommend_for_user(user_id, top_k=5):
    user_clicks = clicks[clicks["user_id"] == user_id]
    if user_clicks.empty:
        return []

    clicked_articles = user_clicks["click_article_id"].unique()
    clicked_embeddings = metadata[metadata["article_id"].isin(clicked_articles)]["embedding"].tolist()
    if not clicked_embeddings:
        return []

    user_profile = np.mean(clicked_embeddings, axis=0).reshape(1, -1)
    similarities = cosine_similarity(user_profile, np.vstack(metadata["embedding"].values))[0]
    metadata["similarity"] = similarities

    recos = metadata[~metadata["article_id"].isin(clicked_articles)]
    return recos.sort_values("similarity", ascending=False).head(top_k)[["article_id", "similarity"]].to_dict(orient="records")


def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        user_id = int(req.route_params.get('userId', -1))
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Invalid userId"}),
            status_code=400,
            mimetype="application/json"
        )

    return func.HttpResponse(
        json.dumps({"method": "cosine", "recommendations": recommend_for_user(user_id)}),
        mimetype="application/json"
    )
