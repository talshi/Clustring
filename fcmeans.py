import copy
import math
import random
import decimal

class FCmeans(object):

    def __init__(self, data):
        self.U = init_U()
        self.centers = []
        self.data = data

    def init_U(self):
        for i in range(0, len(data)):
            current = []
            rand_sum = 0.0
            for j in range(0, cluster_number):
                dummy = random.randint(1, int(MAX))
                current.append(dummy)
                rand_sum += dummy
            for j in range(0, cluster_number):
                current[j] = current[j] / rand_sum
            self.U.append(current)

    def normalise_U(self):
        """
        This de-fuzzifies the U, at the end of the clustering. It would assume that the point is a member of the cluster whoes membership is maximum.
        """
        for i in range(0, len(self.U)):
            maximum = max(self.U[i])
            for j in range(0, len(self.U[0])):
                if self.U[i][j] != maximum:
                    self.U[i][j] = 0
                else:
                    self.U[i][j] = 1

    def end_conditon(U, U_old):
        """
        This is the end conditions, it happens when the U matrix stops chaning too much with successive iterations.
        """
        global Epsilon
        for i in range(0, len(U)):
            for j in range(0, len(U[0])):
                if abs(U[i][j] - U_old[i][j]) > Epsilon:
                    return False

        return True

    def fuzzy_cmeans(self, cluster_num, m=2):
        while (True):
            # create a copy of it, to check the end conditions
            U_old = copy.deepcopy(self.U)
            # cluster center vector
            C = []
            for j in range(0, cluster_num):
                current_cluster_center = []
                for i in range(0, len(self.data[0])):  # this is the number of dimensions
                    dummy_sum_num = 0.0
                    dummy_sum_dum = 0.0
                    for k in range(0, len(self.data)):
                        dummy_sum_num += (self.U[k][j] ** m) * self.data[k][i]
                        dummy_sum_dum += (self.U[k][j] ** m)
                    current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
                C.append(current_cluster_center)

            # creating a distance vector, useful in calculating the U matrix.

            distance_matrix = []
            for i in range(0, len(self.data)):
                current = []
                for j in range(0, cluster_number):
                    current.append(distance(self.data[i], C[j]))
                distance_matrix.append(current)

            # update U vector
            for j in range(0, cluster_number):
                for i in range(0, len(self.data)):
                    dummy = 0.0
                    for k in range(0, cluster_number):
                        dummy += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2 / (m - 1))
                    self.U[i][j] = 1 / dummy

            if end_conditon(self.U, U_old):
                print "finished clustering"
                break

        self.U = normalise_U(self.U)
        print "normalised U"

        def get_centers(self):
            return self.centers

        def get_U(self):
            return self.U

