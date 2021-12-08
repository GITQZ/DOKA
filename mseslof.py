import random

from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

class LOF:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.N = self.data.shape[0]
        self.knn = []
        self.Non_NaN = []
        self.edge = []
        self.NaN = []

    def get_dist(self):

        return cdist(self.data, self.data)

    def get_knn(self, dis_mat):
        for i in range(self.N):
            # print(i)
            Selected = np.array([True]*self.N)
            Selected[i] = False

            e = 0

            min_val = np.min(dis_mat[i, Selected])

            min_index = np.where(dis_mat[i, Selected] == min_val)[0]
            Selected[min_index+1] = False

            e += min_val*len(min_index)

            while np.sum(Selected == False) < self.k + 1:

                s = dis_mat[~Selected][:, Selected]

                min_val = np.min(s)

                index = np.unique(np.where(s == min_val)[1])
                i = np.where(Selected == True)[0][index]
                Selected[i] = False

                e += min_val*(len(i))
                # print(e)
            self.edge.append(e)
            Selected[i] = True
            self.knn.append(np.where(Selected == False)[0])
            # print(self.knn)

    def get_Non_NaN(self):
        for i in range(self.N):
            non_nan = []
            nan = []
            for j in self.knn[i]:
                if i not in self.knn[j]:
                    non_nan.append(j)
                else:
                    nan.append(j)
            self.Non_NaN.append(non_nan)
            self.NaN.append(nan)
        return self.Non_NaN, self.NaN

    def run(self):

        lrd = np.array([0.0]*self.N)

        lof = np.array([0.0]*self.N)
        dis_mat = np.round(self.get_dist(), 8)
        self.get_knn(dis_mat)

        Non_NaN, NaN = self.get_Non_NaN()
        for i in range(self.N):
            lrd[i] = (len(self.knn[i]))/self.edge[i]
        for i in range(self.N):
            if lof[i] == 0:
                if len(Non_NaN[i]) != 0:
                    lof[i] = (np.sum(lrd[Non_NaN[i]])/lrd[i])/len(Non_NaN[i])
                else:
                    lof[i] = 1
            # lof[NaN[i]] = lof[i] + np.random.normal(0.0001, 0.0001)

        return lof


