import logging
import azure.functions as func
import json
import traceback

# --- Entrée HTTP Azure ---
def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
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
