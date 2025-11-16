# util.py  (исправленный, без лишних сдвигов)
import numpy as np
from scipy.sparse import csr_matrix, diags
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*divide by zero encountered.*")

# ---------- sparse-incidence helpers ----------------------------------------
def data_masks(all_sessions, n_node):
    indptr, indices, data = [0], [], []

    for seq in all_sessions:
        uniq = np.unique(seq)
        indices.extend(uniq)
        data.extend([1] * len(uniq))
        indptr.append(indptr[-1] + len(uniq))

    return csr_matrix((data, indices, indptr),
                      shape=(len(all_sessions), n_node))

# ---------- validation split -------------------------------------------------
def split_validation(train_set, valid_portion):
    x, y = train_set
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    n_train = int(len(x) * (1 - valid_portion))
    return (x[idx[:n_train]], y[idx[:n_train]]), \
           (x[idx[n_train:]], y[idx[n_train:]])

# ---------- Data container ---------------------------------------------------
class Data:
    def __init__(self, data, shuffle=False, n_node=None):
        self.raw     = np.array(data[0], dtype=object)
        self.targets = np.asarray(data[1])

        H_T = data_masks(self.raw, n_node)

        # row-normalised H_T
        row_sum = np.asarray(H_T.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1
        BH_T = H_T.multiply(1.0 / row_sum.reshape(-1, 1))

        # col-normalised H
        H = H_T.T
        col_sum = np.asarray(H.sum(axis=1)).ravel()
        col_sum[col_sum == 0] = 1
        DH = H.multiply(1.0 / col_sum.reshape(-1, 1))

        self.adjacency = (DH @ BH_T).tocoo()
        self.n_node    = n_node
        self.length    = len(self.raw)
        self.shuffle   = shuffle

    # -------------------------------------------------------------------------
    def generate_batch(self, batch_size):
        if self.shuffle:
            idx = np.arange(self.length)
            np.random.shuffle(idx)
            self.raw     = self.raw[idx]
            self.targets = self.targets[idx]

        n_batch = (self.length + batch_size - 1) // batch_size
        slices  = np.arange(n_batch * batch_size).reshape(n_batch, batch_size)
        slices[-1] = np.arange(max(0, self.length - batch_size), self.length)
        return slices

    # -------------------------------------------------------------------------
    def get_slice(self, index):
        items, lens, mask = [], [], []
        sessions = self.raw[index]

        max_len = max(len(s) for s in sessions)

        for s in sessions:
            l = len(s)
            items.append(list(s) + [0] * (max_len - l))
            lens.append([l])
            mask.append([1] * l + [0] * (max_len - l))

        return self.targets[index], lens, items, mask