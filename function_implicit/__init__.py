import logging
import azure.functions as func
import pandas as pd
import numpy as np
import json
import os
import pickle
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix

metadata = pd.read_csv("articles_metadata.csv")
clicks = pd.concat(
    [pd.read_csv(os.path.join("clicks", f)) for f in os.listdir("clicks")],
    ignore_index=True
)

# Construction matrice utilisateur-article
user_ids = clicks["user_id"].astype("category")
article_ids = clicks["click_article_id"].astype("category")
user_map = dict(enumerate(user_ids.cat.categories))
article_map = dict(enumerate(article_ids.cat.categories))

matrix = coo_matrix(
    (np.ones(len(clicks)), (user_ids.cat.codes, article_ids.cat.codes))
).tocsr()

BASE_DIR = os.path.dirname(__file__)+'\\..\\'

# Charger modèle implicit pré-entraîné
with  open(os.path.join(BASE_DIR, "file", "implicit_model.pkl"), "rb") as f:
    model: AlternatingLeastSquares = pickle.load(f)

def recommend_for_user(user_id, top_k=5):
    if user_id not in user_ids.cat.categories:
        return []
    user_index = list(user_ids.cat.categories).index(user_id)
    rec_indices, scores = model.recommend(user_index, matrix[user_index], N=top_k)
    return [{"article_id": int(article_map[i]), "score": float(s)} for i, s in zip(rec_indices, scores)]

def main(req: func.HttpRequest) -> func.HttpResponse:
    user_id = int(req.route_params.get('userId', -1))
    return func.HttpResponse(
        json.dumps({"method": "implicit", "recommendations": recommend_for_user(user_id)}),
        mimetype="application/json"
    )
