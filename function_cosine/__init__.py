import logging
import azure.functions as func
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import pickle
from utils import get_file_path
import traceback

# Variables globales initialisées à None
embeddings = None
metadata = None
clicks = None
cold_start_done = False


# --- Chargement des fichiers avec vérification ---
def load_embeddings():
    file_path = get_file_path("articles_embeddings.pkl")
    print(f"Chemin embeddings : {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable sur Azure.")
    with open(file_path, "rb") as f:
        embeddings = pickle.load(f)
    print(f"Nombre d'embeddings chargés : {len(embeddings)}")
    return embeddings


def load_metadata():
    file_path = get_file_path("articles_metadata.csv")
    print(f"Chemin metadata : {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable sur Azure.")
    df = pd.read_csv(file_path)
    print(f"Nombre de lignes metadata : {len(df)}")
    return df


def load_clicks():
    clicks_folder = "clicks"
    print(f"Vérification du dossier clicks : {clicks_folder}")
    if not os.path.exists(clicks_folder) or not os.path.isdir(clicks_folder):
        raise FileNotFoundError(f"Le dossier {clicks_folder} est introuvable sur Azure.")
    all_clicks = []
    for filename in os.listdir(clicks_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(clicks_folder, filename)
            df = pd.read_csv(file_path)
            all_clicks.append(df)
    if not all_clicks:
        raise RuntimeError("Aucun fichier de clic trouvé dans le dossier clicks/.")
    concatenated = pd.concat(all_clicks, ignore_index=True)
    print(f"Nombre total de clics : {len(concatenated)}")
    return concatenated


# --- Initialisation des données (appelée dans main) ---
def init_data():
    global embeddings, metadata, clicks, cold_start_done
    if cold_start_done:
        return

    print("=== Initialisation des données ===")
    try:
        embeddings = load_embeddings()
        metadata = load_metadata()
        clicks = load_clicks()

        if len(metadata) != len(embeddings):
            raise ValueError(
                f"Mismatch entre metadata ({len(metadata)} lignes) "
                f"et embeddings ({len(embeddings)} éléments)."
            )

        metadata["embedding"] = list(embeddings)
        print("Chargement initial terminé")
        cold_start_done = True
    except Exception as e:
        print("Erreur pendant l'initialisation :", e)
        traceback.print_exc()
        raise


# --- Fonction de recommandation ---
def recommend_for_user(user_id, top_k=5):
    try:
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
        top_recos = recos.sort_values("similarity", ascending=False).head(top_k)
        return top_recos[["article_id", "similarity"]].to_dict(orient="records")
    except Exception as e:
        print(f"Erreur dans recommend_for_user : {e}")
        traceback.print_exc()
        raise


# --- Entrée HTTP Azure ---
def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Initialisation retardée (cold start simulé à la première requête)
        init_data()

        user_id = req.route_params.get("userId")
        if not user_id:
            req_body = req.get_json()
            user_id = req_body.get("user_id")

        if not user_id:
            return func.HttpResponse(
                json.dumps({"error": "Missing user_id"}),
                status_code=400,
                mimetype="application/json"
            )

        user_id = int(user_id)
        recos = recommend_for_user(user_id)

        return func.HttpResponse(
            json.dumps({"user_id": user_id, "recommendations": recos}),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        print("Erreur lors du traitement de la requête : ", e)
        traceback.print_exc()
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
