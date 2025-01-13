import streamlit as st
import joblib
import pandas as pd

# Titre de l'application
st.title('Application de détection de risque de décès COVID-19')

# Charger le modèle sauvegardé
model = joblib.load('covid_risk_model.pkl')

# Entrées utilisateur
st.sidebar.header("Entrez les informations du patient")

age = st.sidebar.slider('Âge', 0, 100, 50)
sex = st.sidebar.selectbox('Sexe', ['Homme', 'Femme'])
pneumonia = st.sidebar.selectbox('Pneumonie', ['Non', 'Oui'])
diabetes = st.sidebar.selectbox('Diabète', ['Non', 'Oui'])
obesity = st.sidebar.selectbox('Obésité', ['Non', 'Oui'])

# Convertir les entrées en valeurs numériques
sex = 1 if sex == 'Femme' else 0
pneumonia = 1 if pneumonia == 'Oui' else 0
diabetes = 1 if diabetes == 'Oui' else 0
obesity = 1 if obesity == 'Oui' else 0

# Créer un DataFrame avec les entrées
input_data = pd.DataFrame({
    'AGE': [age],
    'SEX': [sex],
    'PNEUMONIA': [pneumonia],
    'DIABETES': [diabetes],
    'OBESITY': [obesity]
})

# Bouton pour lancer la prédiction
if st.sidebar.button('Prédire le risque de décès'):
    # Faire la prédiction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Afficher le résultat
    if prediction[0] == 1:
        st.error('Risque de décès élevé')
    else:
        st.success('Risque de décès faible')

    # Afficher la probabilité
    st.write(f'Probabilité de décès : {prediction_proba[0][1]:.2f}')