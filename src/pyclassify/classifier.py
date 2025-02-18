from pyclassify import utils

class kNN():
    def __init__(self, k):
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        self.k = k

    def _get_k_nearest_neighbors(self, X, y, x):
        dist_list = [(idx, utils.distance(p, x)) for idx, p in enumerate(X)]
        #Order list of touples based on disatnces
        dist_list.sort(key=lambda x: x[1])
        idx_list = [t[0] for t in dist_list[:self.k]]
        final_list = [y[idx] for idx in idx_list]
        return final_list
    
    def __call__(self, data, new_points):
        X = data[0]
        y = data[1]
        y_pred = []
        for x in new_points:
            tmp_labels = self._get_k_nearest_neighbors(X, y, x)
            max_label = utils.majority_vote(tmp_labels)
            y_pred.append(max_label)
        return y_pred