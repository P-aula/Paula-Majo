import unittest
import os
import pandas as pd
from Sarcasmo import train_loader, test_loader, model, predicciones_train, gensim_predict, y_test, predictions, roc_auc_score, X_test

class TestSarcasmoModels(unittest.TestCase):
    
    # Prueba para asegurarse de que BERT genera un AUC mayor a 0.5
    def test_bert_auc(self):
        roc_auc_bert = roc_auc_score(y_test, predictions)
        self.assertGreater(roc_auc_bert, 0.5)
    
    # Prueba para asegurarse de que Gensim genera un AUC mayor a 0.5
    def test_gensim_auc(self):
        gensim_preds = gensim_predict(X_test)
        roc_auc_gensim = roc_auc_score(y_test, gensim_preds)
        self.assertGreater(roc_auc_gensim, 0.5)
    
    # Prueba para verificar si el archivo CSV se ha creado y contiene las métricas correctas
    def test_csv_output(self):
        file_exists = os.path.exists('sarcasmo_metrics.csv')
        self.assertTrue(file_exists)

        if file_exists:
            df = pd.read_csv('sarcasmo_metrics.csv')
            self.assertIn('Modelo', df.columns)
            self.assertIn('AUC', df.columns)
            self.assertEqual(len(df), 2)  # Debe haber 2 modelos: BERT y Gensim

if __name__ == '__main__':
    unittest.main()

