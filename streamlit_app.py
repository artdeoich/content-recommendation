import streamlit as st
import requests
import pandas as pd

USER_IDS = [1, 2, 3, 4, 5, 123]  
BASE_URL = "https://content-recommendation-fngjfxc6bsfdh4cr.canadacentral-01.azurewebsites.net/api"
#BASE_URL = "http://localhost:7071/api"


st.title("Recommandation d'articles - 3 méthodes")

user_id = st.selectbox("Choisir un utilisateur", USER_IDS)

if st.button("Afficher recommandations"):
    methods = {
        "Cosine Similarity": f"{BASE_URL}/recommendations/cosine/{user_id}",
        "Surprise SVD": f"{BASE_URL}/recommendations/surprise/{user_id}",
        "Implicit ALS": f"{BASE_URL}/recommendations/implicit/{user_id}"
    }

    for method_name, url in methods.items():
        st.subheader(method_name)
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                recos = resp.json().get("recommendations", [])
                if recos:
                    st.table(pd.DataFrame(recos))
                else:
                    st.warning("Aucune recommandation trouvée.")
            else:
                st.error(f"Erreur API ({resp.status_code})")
        except Exception as e:
            st.error(f"Erreur : {e}")
