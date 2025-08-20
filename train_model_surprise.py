import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD

# --- Charger les clics ---
clicks_folder = "clicks"
all_clicks = []
i = 0
while True:
    filename = f"clicks_hour_{i:03d}.csv"
    try:
        df = pd.read_csv(f"{clicks_folder}/{filename}")
    except FileNotFoundError:
        break
    all_clicks.append(df)
    i += 1

clicks = pd.concat(all_clicks, ignore_index=True)
clicks["rating"] = 1.0  # rating implicite

# --- Entraînement modèle Surprise ---
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(clicks[["user_id", "click_article_id", "rating"]], reader)
trainset = data.build_full_trainset()

model = SVD()
model.fit(trainset)

# Sauvegarde du modèle
with open("recommendation_model_surprise.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modèle entraîné et sauvegardé en recommendation_model_surprise.pkl")
