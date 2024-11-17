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

#Dane MNIST
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']
y = y.astype(int)  # Konwersja etykiety do typu całkowitego

#Redukcja wymiarowości
#PCA w celu przyspieszenia obliczeń
pca = PCA(n_components=50, whiten=True, random_state=42)
X_reduced = pca.fit_transform(X)

#podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

#Trenowanie modelu Gaussian Mixture Model
gmm = GaussianMixture(n_components=10, random_state=42)
gmm.fit(X_reduced)

# Predykcja etykiet klastrów dla każdego przypadku w zbiorze danych
gmm_labels = gmm.predict(X_reduced)

#Ocena wydajności klasteryzacji
def map_labels(y_true, y_pred):
    labels = np.zeros_like(y_pred)
    for i in range(10):
        mask = (y_pred == i)
        labels[mask] = mode(y_true[mask])[0]
    return labels

# Mapowanie etykiet GMM na rzeczywiste etykiety cyfr
gmm_mapped_labels = map_labels(y, gmm_labels)

# Obliczanie dokładności
accuracy = accuracy_score(y, gmm_mapped_labels)
print(f"Dokładność klasteryzacji: {accuracy * 100:.2f}%")

# Wyświetlanie macierzy pomyłek
conf_matrix = confusion_matrix(y, gmm_mapped_labels)
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, cmap='viridis')
plt.title("Macierz pomyłek")
plt.colorbar()
plt.xlabel("Etykiety przewidywane")
plt.ylabel("Etykiety prawdziwe")
plt.savefig('confusion_matrix_gmm.png')

# Trenowanie klasyfikatora k-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred_knn = knn.predict(X_test)

# Ewaluacja klasyfikatora k-NN
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Dokładność klasyfikacji (k-NN): {accuracy_knn * 100:.2f}%")

# Wyświetlanie macierzy pomyłek dla k-NN
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix_knn, cmap='plasma')
plt.title("Macierz pomyłek (k-NN)")
plt.colorbar()
plt.xlabel("Etykiety przewidywane")
plt.ylabel("Prawdziwe etykiety")
plt.savefig('confusion_matrix_knn.png')