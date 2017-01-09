import random
import numpy as np
import similarity


class KMeans(object):

    def __init__(self, k, vectors, metric):
        assert len(vectors) >= k
        self.centers = random.sample(vectors, k)
        self.clusters = [[] for c in self.centers]
        self.vectors = vectors
        self.metric = metric
        self.init_similarity_function()

    def init_similarity_function(self):
        if self.metric == 'euclidean':
            self.similarity_function = similarity.euclidean_similarity
        elif self.metric == 'jaccard':
            pass
        elif self.metric == 'cosine':
            self.similarity_function = similarity.cosine_similarity
        else:
            self.similarity_function = similarity.euclidean_similarity

    def update_clusters(self):
        def closest_center_index(vector):
            similarity_to_vector = lambda center: self.similarity_function(center,vector)
            center = np.array(max(self.centers, key=similarity_to_vector))
            deltas = []
            for c in self.centers:
                if len(c) > 0:
                    deltas.append(np.subtract(c, center))
            return np.where(deltas == np.array([0, 0]))[0][0] or np.where(np.min(deltas))[0][0]

        self.clusters = [[] for c in self.centers]
        for vector in self.vectors:
            index = closest_center_index(vector)
            self.clusters[index].append(vector)

    def update_centers(self):

        def average(sequence):
            return sum(sequence) / len(sequence)

        new_centers = []
        for cluster in self.clusters:
            center = [average(ci) for ci in zip(*cluster)]
            new_centers.append(center)

        if np.array_equal(new_centers, self.centers):
            return False

        self.centers = new_centers
        return True

    def main_loop(self):
        self.update_clusters()
        while self.update_centers():
            self.update_clusters()

    def get_clusers(self):
        return self.clusters

    def get_centers(self):
        return self.centers


