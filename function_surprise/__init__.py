import logging
import azure.functions as func
import pandas as pd
import pickle
import json
import os

BASE_DIR = os.path.dirname(__file__)+'\\..\\'

# Charger le modèle Surprise
with open(os.path.join(BASE_DIR, "file", "recommendation_model_surprise.pkl"), "rb") as f:
    model = pickle.load(f)

metadata = pd.read_csv("articles_metadata.csv")
clicks = pd.concat(
    [pd.read_csv(os.path.join("clicks", f)) for f in os.listdir("clicks")],
    ignore_index=True
)

def recommend_for_user(user_id, top_k=5):
    seen = set(clicks[clicks["user_id"] == user_id]["click_article_id"])
    candidates = set(metadata["article_id"]) - seen
    predictions = [(aid, model.predict(user_id, aid).est) for aid in candidates]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [{"article_id": aid, "score": score} for aid, score in predictions[:top_k]]

def main(req: func.HttpRequest) -> func.HttpResponse:
    user_id = int(req.route_params.get('userId', -1))
    return func.HttpResponse(
        json.dumps({"method": "surprise", "recommendations": recommend_for_user(user_id)}),
        mimetype="application/json"
    )
