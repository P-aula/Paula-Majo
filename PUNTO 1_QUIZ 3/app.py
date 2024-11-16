import streamlit as st
import pickle
from transformers import BertTokenizer, BertModel
import torch

# Cargar los modelos
with open('lda_model_tfidf.pkl', 'wb') as f:
    lda_model_tfidf = pickle.load(f)
with open('kmeans_tfidf.pkl', 'wb') as f:
    kmeans_tfidf = pickle.load(f)
with open('kmeans_bert.pkl', 'wb') as f:
    kmeans_bert = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Funciones de procesamiento y clasificación
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    return tokens  # Devuelve tokens como lista

    # Función de preprocesamiento como en el paso anterior
    return text.lower()

def classify_lda(text, use_tfidf=False):
    processed_text = preprocess(text)
    bow_vector = lda_model.id2word.doc2bow(processed_text)
    if use_tfidf:
        topic_probs = lda_model_tfidf.get_document_topics(bow_vector)
    else:
        topic_probs = lda_model.get_document_topics(bow_vector)
    top_topic = max(topic_probs, key=lambda x: x[1])[0]
    return top_topic

def classify_tfidf_kmeans(text):
    processed_text = preprocess(text)
    tfidf_vector = tfidf.transform([processed_text]).toarray()
    cluster = kmeans_tfidf.predict(tfidf_vector)[0]
    return cluster

# Streamlit app
st.title("Clasificación de Noticias por Tema")
st.write("Selecciona un modelo y proporciona un texto de noticia para clasificar su tema.")

# Menú de selección de modelo
model_choice = st.selectbox("Selecciona el modelo", ["LDA", "LDA con TF-IDF", "TF-IDF + KMeans", "BERT + KMeans"])

# Input de texto
text_input = st.text_area("Ingresa el texto de la noticia aquí")

# Clasificar y mostrar resultado
if st.button("Classify"):
    if model_choice == "LDA":
        result = classify_lda(text_input)
    elif model_choice == "LDA con TF-IDF":
        result = classify_lda(text_input, use_tfidf=True)
    elif model_choice == "TF-IDF + KMeans":
        result = classify_tfidf_kmeans(text_input)
    st.write(f"El tema de la noticia es: {result}")
