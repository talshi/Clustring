
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

# vectors = np.array([
#     [1,2],
#     [2,3],
#     [3,4],
#     [10,15],
#     [8,6],
#     [1,2],
#     [2,3],
#     [3,4],
#     [10,15],
#     [8,6],
#     [1,2],
#     [2,3],
#     [3,4],
#     [10,15],
#     [8,6]
# ])

vectors = []
for i in range(0, 20):
    dummy = np.random.random_sample((2,))
    vectors.append(dummy)
vectors = np.array(vectors)
print vectors

kmeans = KMeans(n_clusters=10, random_state=0).fit(vectors)
labels = kmeans.labels_
# print kmeans.predict([[0, 0], [4, 4]])
# print kmeans.cluster_centers_
print metrics.silhouette_score(vectors, labels, metric='euclidean')

plt.plot(vectors, 'ro', kmeans.cluster_centers_, 'go')
plt.show()