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
from azure.storage.blob import BlobServiceClient

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
    file_path =  get_file_path("articles_metadata.csv")
    print(f"Chemin metadata : {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable sur Azure.")
    df = pd.read_csv(file_path)
    
    # Affiche les 5 premières lignes
    print("Aperçu des 5 premières lignes :")
    print(df.head())

    print(f"Nombre de lignes metadata : {len(df)}")
    return df


def load_clicks():
    """
    Charge tous les fichiers CSV du dossier clicks/ dans le container file du Blob Storage
    et les concatène en un seul DataFrame.
    """
    # --- Paramètres Blob Storage ---
    connection_string = "<TA_CONNECTION_STRING>"  # mettre ta connexion Azure Storage
    container_name = "file"  # le container où tu as mis clicks/
    clicks_prefix = "clicks/"  # sous-dossier dans le container

    print(f"🔍 Connexion au container '{container_name}' pour récupérer '{clicks_prefix}'")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    all_clicks = []
    # Lister tous les blobs commençant par clicks/
    for blob in container_client.list_blobs(name_starts_with=clicks_prefix):
        if blob.name.endswith(".csv"):
            print(f"➡️ Lecture du fichier blob : {blob.name}")
            blob_client = container_client.get_blob_client(blob)
            data = blob_client.download_blob().readall()
            df = pd.read_csv(io.BytesIO(data))
            all_clicks.append(df)

    if not all_clicks:
        raise RuntimeError(f"Aucun fichier CSV trouvé dans '{clicks_prefix}' du container '{container_name}'.")

    concatenated = pd.concat(all_clicks, ignore_index=True)
    print(f"Nombre total de clics chargés : {len(concatenated)}")
    return concatenated

# --- Initialisation des données (appelée dans main) ---
def init_data():
    global embeddings, metadata, clicks, cold_start_done
    if cold_start_done:
        return

    print("📂 Contenu du dossier clicks/ sur Azure :")
    if os.path.exists("clicks"):
        print(os.listdir("clicks"))
    else:
        print("Le dossier clicks n'existe pas")

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
