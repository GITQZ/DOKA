import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

data = pd.read_csv('wine.csv')  # 0-9
data = data.values
data = data[:, 1:14]
o = 10

N = data.shape[0]

def k_choose(dis_mat, N):
    m = 0
    krnn_1 = np.zeros(N)
    for k in range(1, N):
        krnn_0 = np.zeros(N)
        for i in range(N):
            Selected = np.array([True] * N)
            Selected[i] = False

            min_val = np.min(dis_mat[i, Selected])

            min_index = np.where(dis_mat[i, Selected] == min_val)[0]
            Selected[min_index + 1] = False
            while np.sum(Selected == False) < k + 1:

                s = dis_mat[~Selected][:, Selected]
                min_val = np.min(s)

                index = np.unique(np.where(s == min_val)[1])
                i = np.where(Selected == True)[0][index]
                Selected[i] = False
            Selected[i] = True
            krnn_0[np.where(Selected == False)[0]] += 1

        if (len(np.where(krnn_0 == 0)[0]) == len(np.where(krnn_1 == 0)[0])):
            m += 1
        else:
            m = 0
            krnn_1 = krnn_0.copy()
        if m == 3:
            return np.max(krnn_0)

dis_mat = cdist(data, data)

if __name__ == '__main__':
    krnn = k_choose(dis_mat, N)
    print(np.max(krnn))