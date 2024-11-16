import pandas as pd
import numpy as np
import gensim
from gensim import models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def load_data():
    data = pd.read_pickle("training_set_rel3.pkl")
    features = pd.read_pickle("training_features_NLP_Encuestas.pkl")
    doc_embedding = pd.read_csv("Doc_Embedding_300_NLP_Encuestas.csv")
    return data, doc_embedding, features

def extract_topics(data, topic_indices):
    topics = []
    for y in range(data.shape[0]):
        if len(topic_indices[y]) > 0:
            valid_sublist = [sublist for sublist in topic_indices[y] if len(sublist) > 1]
            if len(valid_sublist) > 0:
                max_index = np.argmax([sublist[1] for sublist in valid_sublist])
                topics.append(valid_sublist[max_index][0])
            else:
                topics.append(None)
        else:
            topics.append(None)
    return topics

def create_bow_corpus(data):
    processed_docs = data["processed_docs"]
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=500)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    return dictionary, bow_corpus

def run_lda_model(bow_corpus, dictionary):
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=8, id2word=dictionary, passes=2, workers=2)
    lda_indices = lda_model[bow_corpus]
    return lda_indices

def run_tfidf_model(bow_corpus, dictionary):
    tfidf_model = models.TfidfModel(bow_corpus)
    tfidf_corpus = tfidf_model[bow_corpus]
    lda_model_tfidf = gensim.models.LdaMulticore(tfidf_corpus, num_topics=8, id2word=dictionary, passes=2, workers=4)
    tfidf_indices = lda_model_tfidf[bow_corpus]
    return tfidf_indices

def run_fasttext_model(doc_embedding):
    pca = PCA(n_components=30, random_state=0)
    pcs = pca.fit_transform(doc_embedding.values)
    kmeans = KMeans(n_clusters=8, random_state=0)
    kmeans.fit(pcs)
    return kmeans.labels_

def generate_results(data, embeddings, doc_embedding):
    dictionary, bow_corpus = create_bow_corpus(data)

    lda_indices = run_lda_model(bow_corpus, dictionary)
    tfidf_indices = run_tfidf_model(bow_corpus, dictionary)

    kmeans = KMeans(n_clusters=8, random_state=0).fit(embeddings)
    fast_model = run_fasttext_model(doc_embedding)

    results = data[["essay_id", "essay_set"]]
    results["topic_lda"] = extract_topics(results, lda_indices)
    results["topic_tfidf"] = extract_topics(results, tfidf_indices)
    results['bert'] = kmeans.labels_
    results["fasttext"] = fast_model
    return results

def process_results(data):
    data["topic_lda"] = data["topic_lda"] + 1
    data["topic_tfidf"] = data["topic_tfidf"] + 1
    data["bert"] = data["bert"] + 1
    data["fasttext"] = data["fasttext"] + 1

    matrices = [
        pd.crosstab(data["essay_set"], data["topic_lda"]),
        pd.crosstab(data["essay_set"], data["topic_tfidf"]),
        pd.crosstab(data["essay_set"], data["bert"]),
        pd.crosstab(data["essay_set"], data["fasttext"])
    ]

    results = []
    for matrix in matrices:
        used_categories = set()
        assignment = {}

        for essay_set, row in matrix.iterrows():
            sorted_categories = row.sort_values(ascending=False).index
            for category in sorted_categories:
                if category not in used_categories:
                    assignment[essay_set] = category
                    used_categories.add(category)
                    break
        result = pd.DataFrame(list(assignment.items()), columns=['essay_set', 'Category'])
        results.append(result)

    final_result = pd.merge(results[0], results[1], on="essay_set", suffixes=('_lda', '_tfidf'))
    final_result = pd.merge(final_result, results[2], on="essay_set")
    final_result = pd.merge(final_result, results[3], on="essay_set", suffixes=('_bert', '_fasttext'))

    return final_result

def calculate_roc_values(data):
    processed_data = process_results(data)
    models = ["topic_lda", "topic_tfidf", "bert", "fasttext"]
    unique_topics = processed_data["essay_set"].unique()
    y_true, y_pred = [], []

    for topic in unique_topics:
        processed_data.loc[processed_data["essay_set"] == topic, "essay_set"] = 1
        processed_data.loc[processed_data["essay_set"] != topic, "essay_set"] = 0

        y_true.append(list(processed_data["essay_set"]))
        y_est = []
        for model in models:
            processed_data.loc[processed_data[model] == topic, model] = 1
            processed_data.loc[processed_data[model] != topic, model] = 0
            y_est.append(list(processed_data[model]))
        y_pred.append(y_est)

    roc_values = pd.DataFrame({"Topics": [f"Topic {i+1}" for i in range(8)]})
    for model_index in range(len(y_pred[0])):
        roc_scores = [roc_auc_score(y_true[i], y_pred[i][model_index]).round(2) for i in range(len(y_pred))]
        roc_values[f"Model {model_index}"] = roc_scores

    return roc_values

def generate_roc_table():
    data, doc_embedding, features = load_data()
    embeddings = features[["feature_1", "feature_2"]].values  # Ajustar según las características del dataset.
    roc_values = calculate_roc_values(data)
    roc_values = roc_values.rename(columns={
        'Model 0': 'No TF-IDF', 'Model 1': 'With TF-IDF',
        'Model 2': 'BERT Model', 'Model 3': 'FastText Model'
    })
    roc_values.to_csv("roc_curve_results.csv", index=False)
    return roc_values
