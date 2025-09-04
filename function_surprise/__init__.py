import logging
import azure.functions as func
import pandas as pd
import pickle
import json
import os
from utils import get_file_path

# --- Charger le modèle Surprise depuis /file ---
with open(get_file_path("recommendation_model_surprise.pkl"), "rb") as f:
    model = pickle.load(f)

# --- Charger les données ---
base_dir = os.path.dirname(os.path.dirname(__file__))  # <- remonte d'un dossier
file_path =  os.path.join(base_dir, "articles_metadata.csv")
metadata = pd.read_csv(file_path)

clicks = pd.concat(
    [pd.read_csv(os.path.join("clicks", f)) for f in os.listdir("clicks") if f.endswith(".csv")],
    ignore_index=True
)

# --- Génération de recommandations ---
def recommend_for_user(user_id, top_k=5):
    seen = set(clicks[clicks["user_id"] == user_id]["click_article_id"])
    candidates = set(metadata["article_id"]) - seen

    predictions = [(aid, model.predict(user_id, aid).est) for aid in candidates]
    predictions.sort(key=lambda x: x[1], reverse=True)

    return [{"article_id": aid, "score": score} for aid, score in predictions[:top_k]]

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
        json.dumps({"method": "surprise", "user_id": user_id, "recommendations": recos}),
        mimetype="application/json"
    )
