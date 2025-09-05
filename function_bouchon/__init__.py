import logging
import azure.functions as func
import json
import traceback
import pandas as pd
import os
# --- Entrée HTTP Azure ---
def main(req: func.HttpRequest) -> func.HttpResponse:
    try:

        # -- clicks_folder = "./file/clicks"  # chemin local vers tes CSV
        # -- all_clicks = pd.concat(
        # --     [pd.read_csv(os.path.join(clicks_folder, f)) for f in os.listdir(clicks_folder) if f.endswith(".csv")],
        # --     ignore_index=True
        # -- )
        # -- all_clicks.to_parquet("./file/clicks.parquet", index=False)

        user_id = req.route_params.get("userId")

        if not user_id:
            return func.HttpResponse(
                json.dumps({"error": "Missing user_id"}),
                status_code=400,
                mimetype="application/json"
            )

        user_id = int(user_id)

        return func.HttpResponse(
            json.dumps({"user_id": user_id}),
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
