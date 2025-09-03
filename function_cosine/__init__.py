import logging
import azure.functions as func
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import pickle
from utils import get_file_path

# --- Chargement des données ---
def load_embeddings():
    try:
        file_path = get_file_path("articles_embeddings.pkl")
        logging.info(f"Chargement des embeddings depuis : {file_path}")
        with open(file_path, "rb") as f:
            embeddings = pickle.load(f)
        logging.info(f"{len(embeddings)} embeddings chargés")
        return embeddings
    except Exception as e:
        logging.error(f"Erreur lors du chargement des embeddings : {e}", exc_info=True)
        raise

def load_metadata():
    try:
        logging.info("Chargement du fichier articles_metadata.csv")
        df = pd.read_csv("articles_metadata.csv")
        logging.info(f"{len(df)} lignes de metadata chargées")
        return df
    except Exception as e:
        logging.error(f"Erreur lors du chargement des metadata : {e}", exc_info=True)
        raise

def load_clicks():
    try:
        clicks_folder = "clicks"
        logging.info(f"Lecture des fichiers de clics depuis {clicks_folder}/")
        all_clicks = []
        for filename in os.listdir(clicks_folder):
            if filename.endswith(".csv"):
                file_path = os.path.join(clicks_folder, filename)
                logging.info(f"Lecture du fichier {file_path}")
                df = pd.read_csv(file_path)
                all_clicks.append(df)
        if not all_clicks:
            raise RuntimeError("Aucun fichier de clic trouvé dans le dossier.")
        concatenated = pd.concat(all_clicks, ignore_index=True)
        logging.info(f"Total clics chargés : {len(concatenated)}")
        return concatenated
    except Exception as e:
        logging.error(f"Erreur lors du chargement des clics : {e}", exc_info=True)
        raise

# Chargement au démarrage (cold start)
try:
    embeddings = load_embeddings()
    metadata = load_metadata()
    clicks = load_clicks()
    metadata["embedding"] = list(embeddings)
    logging.info("Chargement initial terminé")
except Exception as e:
    logging.error("Erreur pendant le cold start", exc_info=True)
    raise

# --- Recommandations ---
def recommend_for_user(user_id, top_k=5):
    try:
        logging.info(f"Génération des recommandations pour user_id={user_id}")
        user_clicks = clicks[clicks["user_id"] == user_id]
        logging.info(f"Nombre de clics de l'utilisateur : {len(user_clicks)}")
        if user_clicks.empty:
            logging.warning(f"Aucun clic trouvé pour user_id={user_id}")
            return []

        clicked_articles = user_clicks["click_article_id"].unique()
        clicked_embeddings = metadata[metadata["article_id"].isin(clicked_articles)]["embedding"].tolist()
        if not clicked_embeddings:
            logging.warning(f"Aucun embedding trouvé pour les articles cliqués par user_id={user_id}")
            return []

        user_profile = np.mean(clicked_embeddings, axis=0).reshape(1, -1)
        similarities = cosine_similarity(user_profile, np.vstack(metadata["embedding"].values))[0]

        metadata["similarity"] = similarities
        recos = metadata[~metadata["article_id"].isin(clicked_articles)]
        top_recos = recos.sort_values("similarity", ascending=False).head(top_k)
        logging.info(f"{len(top_recos)} recommandations générées pour user_id={user_id}")
        return top_recos[["article_id", "similarity"]].to_dict(orient="records")
    except Exception as e:
        logging.error(f"Erreur dans recommend_for_user pour user_id={user_id} : {e}", exc_info=True)
        raise

# --- Point d'entrée Azure ---
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function démarrée")
    user_id = req.route_params.get("userId")

    if not user_id:
        try:
            req_body = req.get_json()
            user_id = req_body.get("user_id")
        except ValueError as e:
            logging.warning(f"Impossible de parser le body JSON : {e}")
    
    if not user_id:
        logging.error("user_id manquant dans la requête")
        return func.HttpResponse(
            json.dumps({"error": "Missing user_id"}),
            status_code=400,
            mimetype="application/json"
        )

    try:
        user_id = int(user_id)
    except ValueError:
        logging.error(f"user_id invalide : {user_id}")
        return func.HttpResponse(
            json.dumps({"error": "user_id must be an integer"}),
            status_code=400,
            mimetype="application/json"
        )

    try:
        recos = recommend_for_user(user_id)
        logging.info(f"Recommandations renvoyées pour user_id={user_id}")
        return func.HttpResponse(
            json.dumps({"user_id": user_id, "recommendations": recos}),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(f"Erreur lors du traitement de la requête pour user_id={user_id} : {e}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": "Internal server error"}),
            status_code=500,
            mimetype="application/json"
        )
