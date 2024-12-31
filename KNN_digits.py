import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report


digits = load_digits()
X = digits.data
y_true = digits.target  


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)


optimal_k = 10
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
clusters = kmeans.fit_predict(X_scaled)


from scipy.stats import mode

labels = np.zeros_like(clusters)
for i in range(optimal_k):
    mask = (clusters == i)
    labels[mask] = mode(y_true[mask])[0]


conf_matrix = confusion_matrix(y_true, labels)
print("Confusion Matrix:\n", conf_matrix)


class_report = classification_report(y_true, labels)
print("\nClassification Report:\n", class_report)
