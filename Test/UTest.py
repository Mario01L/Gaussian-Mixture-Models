import unittest
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode
from main import load_mnist_data, reduce_dimensions, train_gmm, map_labels, evaluate_gmm, train_knn, evaluate_knn

class TestModelFunctions(unittest.TestCase):

    def setUp(self):
        # Ładowanie danych
        X, y = load_mnist_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train_reduced = reduce_dimensions(self.X_train)
        self.X_test_reduced = reduce_dimensions(self.X_test)

    def test_load_mnist_data(self):
        X, y = load_mnist_data()
        self.assertEqual(X.shape[0], 70000)  # czy mamy 70,000 próbek
        self.assertEqual(y.shape[0], 70000)  # czy etykiety mają 70,000 elementów

    def test_reduce_dimensions(self):
        X_reduced = reduce_dimensions(self.X_train)
        self.assertEqual(X_reduced.shape[1], 50)  # po PCA liczba cech powinna wynosić 50

    def test_train_gmm(self):
        gmm = train_gmm(self.X_train_reduced)
        self.assertIsInstance(gmm, GaussianMixture)  # czy zwrócony obiekt to GaussianMixture

    def test_map_labels(self):
        gmm_labels = np.zeros_like(self.y_train)
        mapped_labels = map_labels(self.y_train, gmm_labels)
        self.assertEqual(mapped_labels.shape, self.y_train.shape)  # czy mapowane etykiety mają odpowiedni rozmiar

    def test_evaluate_gmm(self):
        accuracy, conf_matrix = evaluate_gmm(self.X_train_reduced, self.y_train)
        self.assertGreater(accuracy, 0)  # czy dokładność jest większa niż 0
        self.assertEqual(conf_matrix.shape, (10, 10))  # czy macierz pomyłek ma odpowiedni rozmiar (10x10)

    def test_train_knn(self):
        knn = train_knn(self.X_train_reduced, self.y_train)
        self.assertIsInstance(knn, KNeighborsClassifier)  # czy zwrócony obiekt to KNeighborsClassifier

    def test_evaluate_knn(self):
        knn = train_knn(self.X_train_reduced, self.y_train)
        accuracy_knn, conf_matrix_knn = evaluate_knn(self.X_test_reduced, self.y_test, knn)
        self.assertGreater(accuracy_knn, 0)  # czy dokładność jest większa niż 0
        self.assertEqual(conf_matrix_knn.shape, (10, 10))  # czy macierz pomyłek ma odpowiedni rozmiar (10x10)

if __name__ == "__main__":
    unittest.main()
