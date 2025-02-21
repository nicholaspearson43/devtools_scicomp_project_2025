from pyclassify import utils
import numpy as np
import line_profiler
from .module import distance_numba

LINE_PROFILE = 1

class kNN():
    def __init__(self, k, backhand = "plain"):
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        if backhand not in ["plain", "numpy", "numba"]:
            raise ValueError("backhand must either be numpy or plain")
        self.k = k
        self.backhand = backhand
        if self.backhand == "plain":
            self.distance = utils.distance
        elif self.backhand == "numpy":
            self.distance = utils.distance_numpy
        elif self.backhand == "numba":
            self.distance = utils.distance_numba

    @line_profiler.profile
    def _get_k_nearest_neighbors(self, X, y, x):
        dist_list = [(idx, self.distance(p, x)) for idx, p in enumerate(X)]
        #Order list of touples based on disatnces
        dist_list.sort(key=lambda x: x[1])
        idx_list = [t[0] for t in dist_list[:self.k]]
        final_list = [y[idx] for idx in idx_list]
        return final_list
    
    @line_profiler.profile
    def __call__(self, data, new_points):
        X = data[0]
        y = data[1]
        y_pred = []
        if self.backhand == "numpy":
            X = np.array(X)
            #y = np.array(y)
            new_points = np.array(new_points)
        for x in new_points:
            tmp_labels = self._get_k_nearest_neighbors(X, y, x)
            max_label = utils.majority_vote(tmp_labels)
            y_pred.append(max_label)
        return y_pred