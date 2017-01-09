import copy
import math
import random
import decimal
import similarity

class FCmeans(object):

    def __init__(self, data, metric, cluster_number):
        self.prevU = []
        self.centers = []
        self.metric = metric
        self.data = data
        self.epsilon = 0.00000001
        self.max = 10000.0
        self.cluster_number = cluster_number
        self.U = self.init_U()
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

    def init_U(self):
        U = []
        for i in range(0, len(self.data)):
            current = []
            rand_sum = 0.0
            for j in range(0, self.cluster_number):
                m = random.randint(1, self.max)
                current.append(m)
                rand_sum += m
            for j in range(0, self.cluster_number):
                current[j] = current[j] / rand_sum
            U.append(current)
        return U

    def end_conditon(self):
        for i in range(0, len(self.U)):
            for j in range(0, len(self.U[0])):
                if abs(self.U[i][j] - self.prevU[i][j]) > self.epsilon:
                    return False
        return True

    def fuzzy_cmeans(self, m=2):
        while (True):
            self.prevU = copy.deepcopy(self.U)
            C = []
            for j in range(0, self.cluster_number):
                current_cluster_center = []
                for i in range(0, len(self.data[0])):
                    dummy_sum_num = 0.0
                    dummy_sum_dum = 0.0
                    for k in range(0, len(self.data)):
                        dummy_sum_num += (self.U[k][j] ** m) * self.data[k][i]
                        dummy_sum_dum += (self.U[k][j] ** m)
                    current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
                C.append(current_cluster_center)

            distance_matrix = []
            for i in range(0, len(self.data)):
                current = []
                for j in range(0, self.cluster_number):
                    current.append(self.similarity_function(self.data[i], C[j]))
                distance_matrix.append(current)

            for j in range(0, self.cluster_number):
                for i in range(0, len(self.data)):
                    dummy = 0.0
                    for k in range(0, self.cluster_number):
                        dummy += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2 / (m - 1))
                    self.U[i][j] = 1 / dummy

            if self.end_conditon():
                print "clustering has finished."
                break

    def get_centers(self):
        return self.centers

    def get_U(self):
        return self.U

