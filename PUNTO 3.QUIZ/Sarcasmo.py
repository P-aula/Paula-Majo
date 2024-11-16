import pandas as pd
import spacy
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
import gensim.downloader as api  # Usa Gensim si no puedes instalar FastText
import csv

# Cargar los datos
df_sarcasmo_train = pd.read_csv('Sarcasmo_train.csv', sep=';')
df_sarcasmo_test = pd.read_csv('Sarcasmo_test.csv', sep=';')

# Cargar el modelo Spacy para español
nlp = spacy.load('es_core_news_sm')

# Preprocesar los datos (eliminación de stopwords, lematización, y filtrado de puntuación)
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
    return ' '.join(tokens)

df_sarcasmo_train['processed_Locución'] = df_sarcasmo_train['Locución'].apply(preprocess_text)
df_sarcasmo_test['processed_Locución'] = df_sarcasmo_test['Locución'].apply(preprocess_text)

# Convertir las etiquetas en numéricas
le = LabelEncoder()
df_sarcasmo_train['Sarcasmo'] = le.fit_transform(df_sarcasmo_train['Sarcasmo'])
df_sarcasmo_test['Sarcasmo'] = le.transform(df_sarcasmo_test['Sarcasmo'])

X_train = df_sarcasmo_train['processed_Locución']
y_train = df_sarcasmo_train['Sarcasmo']
X_test = df_sarcasmo_test['processed_Locución']
y_test = df_sarcasmo_test['Sarcasmo']

# Tokenización y preparación de datos para BERT
tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')

train_encodings = tokenizer(X_train.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=256)
test_encodings = tokenizer(X_test.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=256)

train_labels = torch.tensor(y_train.values, dtype=torch.long)
test_labels = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Cargar el modelo BERT
model = BertForSequenceClassification.from_pretrained('dccuchile/bert-base-spanish-wwm-cased', num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimización y entrenamiento de BERT
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Entrenamiento de BERT
model.train()
for epoch in range(3):  # Puedes cambiar el número de épocas
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluación de BERT
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, _ = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())

roc_auc_bert = roc_auc_score(y_test, predictions)
print(f"BERT AUC: {roc_auc_bert}")

# Función para mostrar la matriz de confusión y la curva ROC
def predicciones_train(ytrue, yhat):
    # Matriz de confusión
    matriz_confusion = pd.DataFrame(confusion_matrix(ytrue, yhat))
    plt.figure(figsize=(8,6))
    sns.heatmap(matriz_confusion, linewidth=0.01, annot=True, cmap='Blues', fmt='.0f')
    plt.title('Matriz de confusión para clasificación')
    plt.xlabel('Valores predichos'); plt.ylabel('Valores observados')
    plt.show()

    # Curva ROC
    print(f'El área bajo la curva ROC es: {roc_auc_score(ytrue, yhat):.5f}')
    fpr, tpr, _ = roc_curve(ytrue, yhat)
    roc_auc = auc(fpr, tpr)

    # Graficar la curva ROC
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

# Resultados de BERT
print("Resultados para BERT:")
predicciones_train(y_test, predictions)

# Gensim (o FastText) en lugar de FastText
word_vectors = api.load("glove-wiki-gigaword-100")

def gensim_predict(texts):
    predictions = []
    for text in texts:
        tokens = text.split()
        embeddings = [word_vectors[token] for token in tokens if token in word_vectors]

        if len(embeddings) > 0:
            embedding_avg = sum(embeddings) / len(embeddings)
            predictions.append(1 if sum(embedding_avg) > 0 else 0)
        else:
            predictions.append(0)  # Si no hay embeddings, predecimos 0
    return predictions

# Evaluación de Gensim
gensim_preds = gensim_predict(X_test)
roc_auc_gensim = roc_auc_score(y_test, gensim_preds)
print(f"Gensim AUC: {roc_auc_gensim}")

# Resultados de Gensim
print("Resultados para Gensim:")
predicciones_train(y_test, gensim_preds)

# Guardar métricas en CSV
with open('sarcasmo_metrics.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Modelo', 'AUC'])
    writer.writerow(['BERT', roc_auc_bert])
    writer.writerow(['Gensim', roc_auc_gensim])
