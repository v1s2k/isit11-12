import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


dataset=pd.read_csv('train.csv')
X = dataset.iloc[:, [28, 528]].values



kmeansModel=KMeans(n_clusters=2,init='k-means++')
kmeansModel.fit(X)
plt.scatter(dataset.iloc[:, 28], dataset.iloc[:, 528], c=kmeansModel.labels_, cmap='coolwarm')
plt.show()

wcss = []
for i in range(2, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(2, 10), wcss)
plt.title('The Elbow Method')
plt.xlabel('Число кластеров')
plt.ylabel('Инерция(среднеквадрат расстояние)')
plt.show()

Silhouette_measure = []
K = range(2, 10)
for num_clusters in K:
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    Silhouette_measure.append(silhouette_score(X, kmeans.labels_))
plt.plot(K, Silhouette_measure, 'bx-',color='green')
plt.xlabel('Число кластеров')
plt.ylabel('Silhouette measure')
plt.title('Оптимальное число кластеров')
plt.show()

