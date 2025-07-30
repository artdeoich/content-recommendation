from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# === Chargement des données ===

def load_embeddings():
    with open("articles_embeddings.pickle", "rb") as f:
        return pickle.load(f)

def load_metadata():
    df = pd.read_csv("articles_metadata.csv")
    return df

#def load_clicks():
#    base = pd.read_csv("clicks_sample.csv")
#    clicks_folder = "clicks"
#    if os.path.isdir(clicks_folder):
#        for file in os.listdir(clicks_folder):
#            if file.endswith(".csv"):
#                df = pd.read_csv(os.path.join(clicks_folder, file))
#                base = pd.concat([base, df])
#    return base
    
def load_clicks():
    clicks_folder = "clicks"
    all_clicks = []

    i = 0
    while True:
        filename = f"clicks_hour_{i:03d}.csv"
        file_path = os.path.join(clicks_folder, filename)

        if not os.path.exists(file_path):
            break  # stop when the file doesn't exist

        df = pd.read_csv(file_path)
        all_clicks.append(df)

        i += 1

    if not all_clicks:
        raise RuntimeError("Aucun fichier de clic trouvé dans le dossier.")

    clicks_df = pd.concat(all_clicks, ignore_index=True)
    return clicks_df

# Chargement (au démarrage)
embeddings = load_embeddings()
metadata = load_metadata()
clicks = load_clicks()

metadata["embedding"] = list(embeddings)

# === Recommandation ===

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

    return top_recos[["article_id", "article_id", "similarity"]].to_dict(orient="records")

# === Route API ===

@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user_id", type=int)
    if user_id is None:
        return jsonify({"error": "Missing or invalid user_id"}), 400

    recos = recommend_for_user(user_id)
    return jsonify({"user_id": user_id, "recommendations": recos})

# === Lancement ===

if __name__ == "__main__":
    app.run(debug=True)
