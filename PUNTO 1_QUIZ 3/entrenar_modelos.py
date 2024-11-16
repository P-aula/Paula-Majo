import pandas as pd
import pickle
import nltk
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Función para preprocesar el texto
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    return ' '.join(tokens)

# Cargar los datos
data = pd.read_csv('test_news.csv')
data['processed_text'] = data['headline_text'].apply(preprocess)

# Entrenar modelo LDA
dictionary = corpora.Dictionary(data['processed_text'].str.split())
corpus = [dictionary.doc2bow(text.split()) for text in data['processed_text']]
lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=5)

# Guardar modelo LDA
with open('lda_model.pkl', 'wb') as f:
    pickle.dump(lda_model, f)

# Entrenar y guardar LDA con TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf.fit_transform(data['processed_text'])
lda_model_tfidf = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=5)

with open('lda_model_tfidf.pkl', 'wb') as f:
    pickle.dump(lda_model_tfidf, f)

# Generar embeddings con TF-IDF y aplicar KMeans
tfidf_matrix_dense = tfidf_matrix.toarray()  # Convertir TF-IDF a matriz densa
kmeans_tfidf = KMeans(n_clusters=5, random_state=0).fit(tfidf_matrix_dense)

with open('kmeans_tfidf.pkl', 'wb') as f:
    pickle.dump(kmeans_tfidf, f)

# Entrenar BERT + KMeans
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
embeddings = []

for text in data['processed_text']:
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    embeddings.append(embedding)

kmeans_bert = KMeans(n_clusters=5, random_state=0).fit(embeddings)

with open('kmeans_bert.pkl', 'wb') as f:
    pickle.dump(kmeans_bert, f)
