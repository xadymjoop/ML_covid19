import streamlit as st
import joblib
import pandas as pd
import gdown
import os

# Titre de l'application
st.title("Prédiction d'admission en soins intensifs (COVID-19)")

# Télécharger et charger le modèle
@st.cache_resource
def load_model():
    url = 'https://drive.google.com/uc?id=1X_aSkREb2TRXOHLmqzr8_neAy9OJEuTb'
    output = 'covid_icu_model.pkl'
    
    # Télécharger le modèle s'il n'existe pas déjà
    if not os.path.exists(output):
        gdown.download(url, output, quiet=True)
    
    # Vérifier que le fichier a bien été téléchargé
    if not os.path.exists(output):
        raise FileNotFoundError(f"Le fichier du modèle {output} n'a pas pu être téléchargé.")
    
    return joblib.load(output)

# Charger le modèle avec vérification
try:
    model = load_model()
    st.success("Modèle chargé avec succès !")
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()  # Arrête l'exécution de l'application si le modèle ne peut pas être chargé

# Formulaire pour saisir les données
st.sidebar.header("Saisissez les informations du patient")
age = st.sidebar.number_input('Âge', min_value=0, max_value=120, value=50)
diabetes = st.sidebar.selectbox('Diabète', [0, 1], help="0 = Non, 1 = Oui")
hypertension = st.sidebar.selectbox('Hypertension', [0, 1], help="0 = Non, 1 = Oui")
obesity = st.sidebar.selectbox('Obésité', [0, 1], help="0 = Non, 1 = Oui")
tobacco = st.sidebar.selectbox('Tabagisme', [0, 1], help="0 = Non, 1 = Oui")

# Bouton pour faire une prédiction
if st.sidebar.button('Prédire'):
    input_data = pd.DataFrame({
        'AGE': [age],
        'DIABETES': [diabetes],
        'HIPERTENSION': [hypertension],
        'OBESITY': [obesity],
        'TOBACCO': [tobacco]
    })
    
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error("Le patient est susceptible d'être admis en soins intensifs.")
        else:
            st.success("Le patient n'est pas susceptible d'être admis en soins intensifs.")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
