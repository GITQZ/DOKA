import numpy as np

def cal_auc(o, score):
    y = []
    x = []
    for item in score:
        out_ind = np.where(np.array(score) > item)[0]

        l1 = len(out_ind)
        l2 = len(np.where(out_ind < o)[0])
        y.append(l2 / o)
        x.append((l1 - l2) / (len(score) - o))
    auc = 0
    x_sort = np.sort(x)
    x_arg = np.argsort(x)
    y_sort = np.array(y)[x_arg]

    for j in range(len(y) - 1):
        auc += (x_sort[j + 1] - x_sort[j]) * (y_sort[j] + y_sort[j + 1])

    return 0.5 * auc + (1-np.max(x))













