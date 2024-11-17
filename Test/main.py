import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
matplotlib.use('Agg')

def load_mnist_data():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist['data'], mnist['target']
    y = y.astype(int)  # Konwersja etykiety do typu całkowitego
    return X, y

def reduce_dimensions(X):
    pca = PCA(n_components=50, whiten=True, random_state=42)
    return pca.fit_transform(X)

def train_gmm(X_reduced, n_components=10):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_reduced)
    return gmm

def map_labels(y_true, y_pred):
    labels = np.zeros_like(y_pred)
    for i in range(10):
        mask = (y_pred == i)
        labels[mask] = mode(y_true[mask])[0]
    return labels

def evaluate_gmm(X_reduced, y):
    gmm = train_gmm(X_reduced)
    gmm_labels = gmm.predict(X_reduced)
    gmm_mapped_labels = map_labels(y, gmm_labels)
    accuracy = accuracy_score(y, gmm_mapped_labels)
    conf_matrix = confusion_matrix(y, gmm_mapped_labels)
    return accuracy, conf_matrix

def train_knn(X_train, y_train, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def evaluate_knn(X_test, y_test, knn):
    y_pred_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
    return accuracy_knn, conf_matrix_knn


def plot_confusion_matrix(conf_matrix, filename="GMM", title="Macierz pomyłek"):
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, cmap='viridis', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Etykiety przewidywane")
    plt.ylabel("Etykiety prawdziwe")

    # Dodanie etykiet do komórek macierzy pomyłek
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, f'{conf_matrix[i, j]}', ha='center', va='center', color='white', fontsize=12)
    plt.savefig(filename)
def main():
    # Ładowanie danych
    X, y = load_mnist_data()

    # Redukcja wymiarowości
    X_reduced = reduce_dimensions(X)

    # Trenowanie GMM
    gmm = train_gmm(X_reduced)
    gmm_labels = gmm.predict(X_reduced)

    # Mapowanie etykiet GMM
    gmm_mapped_labels = map_labels(y, gmm_labels)

    # Obliczanie dokładności i macierzy pomyłek
    accuracy = accuracy_score(y, gmm_mapped_labels)
    print(f"Dokładność klasteryzacji: {accuracy * 100:.2f}%")

    conf_matrix = confusion_matrix(y, gmm_mapped_labels)
    plot_confusion_matrix(conf_matrix, 'confusion_matrix_gmm.png')

    # Trenowanie k-NN
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
    knn = train_knn(X_train, y_train)

    # Ocena k-NN
    accuracy_knn, conf_matrix_knn = evaluate_knn(X_test, y_test, knn)
    print(f"Dokładność klasyfikacji (k-NN): {accuracy_knn * 100:.2f}%")

    plot_confusion_matrix(conf_matrix_knn, 'confusion_matrix_knn.png', title="Macierz pomyłek (k-NN)")


if __name__ == "__main__":
    main()