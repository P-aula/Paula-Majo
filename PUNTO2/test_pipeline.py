import unittest
import pandas as pd
from PIPELINE import calculate_roc_values, generate_results
from sklearn.metrics import roc_auc_score

class TestPipeline(unittest.TestCase):

    def test_calculate_roc_values(self):
        # Cargar datos de ejemplo
        data, doc_embedding, features = generate_results()  # Ajusta según sea necesario
        roc_values = calculate_roc_values(data)

        # Verificar que las columnas de los modelos están presentes
        self.assertIn('Model 0', roc_values.columns)
        self.assertIn('Model 1', roc_values.columns)
        self.assertIn('Model 2', roc_values.columns)
        self.assertIn('Model 3', roc_values.columns)

    def test_best_model_selection(self):
        # Cargar datos y calcular curvas ROC
        data, doc_embedding, features = generate_results()  # Ajusta según sea necesario
        roc_values = calculate_roc_values(data)

        # Obtener el mejor modelo por el área bajo la curva ROC
        best_model = roc_values.drop("Topics", axis=1).max(axis=1).idxmax()

        # Verificar que el mejor modelo esté bien seleccionado
        self.assertEqual(best_model, 'Model 2')  # Asumiendo que el mejor modelo es el 'BERT Model'

if __name__ == '__main__':
    unittest.main()

