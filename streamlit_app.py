import streamlit as st
import requests
import pandas as pd


USER_IDS = [1, 2, 3, 4, 5, 123]  
st.title("Recommandation d'articles")

# Sélection utilisateur via liste déroulante
user_id = st.selectbox("Choisir un utilisateur", USER_IDS)

if st.button("Afficher recommandations"):
    # Appel à l'API Flask
    url = f"http://localhost:5000/recommend?user_id={user_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        recos = data.get("recommendations", [])
        
        if not recos:
            st.warning("Aucune recommandation trouvée pour cet utilisateur.")
        else:
            # Affichage sous forme de tableau
            df = pd.DataFrame(recos)
            st.table(df)
    else:
        st.error(f"Erreur API: {response.status_code}")
