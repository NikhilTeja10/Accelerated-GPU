import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)


df = df[['petal length (cm)', 'petal width (cm)']]


scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)


inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 6))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Plot to Determine Optimal k')
plt.show()


optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
df['Cluster'] = kmeans.fit_predict(df_scaled)


labels = np.zeros_like(df['Cluster'])
for i in range(optimal_k):
    mask = (df['Cluster'] == i)
    labels[mask] = mode(iris.target[mask])[0]


accuracy = accuracy_score(iris.target, labels)
print(f'Clustering Accuracy for k={optimal_k}: {accuracy:.4f}')


plt.figure(figsize=(8, 6))
plt.scatter(df['petal length (cm)'], df['petal width (cm)'], c=df['Cluster'], cmap='viridis', marker='o', edgecolor='k', s=50)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title(f'K-Means Clustering (k={optimal_k}) on Iris Dataset')
plt.colorbar(label='Cluster')
plt.show()
