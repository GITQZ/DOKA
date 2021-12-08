from DOKA import k_choose, mseslof
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from DOKA.tool import cal_auc


data = pd.read_csv('wine.csv')  # 0-9
data = data.values
data = data[:, 1:14]
o = 10

N = data.shape[0]
# 计算距离矩阵
dis_mat = cdist(data, data)
# 计算K
K = k_choose.k_choose(dis_mat, N)#72

lof = mseslof.LOF(data, K)
score = lof.run()
auc = cal_auc(o, score)
print(auc)





