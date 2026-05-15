import numpy as np
from scipy.sparse import csr_matrix
import warnings

# Ignore divide-by-zero warnings for empty sessions
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero encountered.*")

# -------------------- 1. Build binary item-session incidence matrix --------------------
def data_masks(all_sessions, n_node):
    """
    Build binary incidence matrix H_T (sessions × items).
    Each row corresponds to a session, each column to an item.
    Entry = 1 if item occurs in session, else 0.
    """
    indptr, indices, data = [0], [], []

    for seq in all_sessions:
        uniq = np.unique(seq)  # remove duplicates
        indices.extend(uniq)
        data.extend([1] * len(uniq))
        indptr.append(indptr[-1] + len(uniq))

    return csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node), dtype=np.float32)

# -------------------- 2. Train/Validation split --------------------
def split_validation(train_set, valid_portion):
    """
    Randomly split dataset into training and validation sets.
    train_set: tuple(X, y)
    valid_portion: fraction allocated to validation
    Returns: (train_x, train_y), (valid_x, valid_y)
    """
    x, y = train_set
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    n_train = int(len(x) * (1 - valid_portion))
    return (x[idx[:n_train]], y[idx[:n_train]]), (x[idx[n_train:]], y[idx[n_train:]])

# -------------------- 3. Data container --------------------
class Data:
    """
    Holds one split (train / valid / test) for DHCN.
    Precomputes adjacency DH·BH^T (not used in CPU-only model).
    Provides dynamic mini-batch generator.
    """
    def __init__(self, data, shuffle=False, n_node=None):
        self.raw = np.array(data[0], dtype=object)      # list of sessions (ragged)
        self.targets = np.asarray(data[1], dtype=np.int64)  # next-item labels
        self.n_node = n_node
        self.length = len(self.raw)
        self.shuffle = shuffle

        # Build adjacency matrix for DHCN (sparse item-item) – not used in stripped variant
        H_T = data_masks(self.raw, n_node)  # sessions × items
        row_sum = np.asarray(H_T.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1
        BH_T = H_T.multiply(1.0 / row_sum.reshape(-1, 1))

        H = H_T.T
        col_sum = np.asarray(H.sum(axis=1)).ravel()
        col_sum[col_sum == 0] = 1
        DH = H.multiply(1.0 / col_sum.reshape(-1, 1))

        self.adjacency = (DH @ BH_T).tocoo()

    # -------------------- Mini-batch generator --------------------
    def generate_batch(self, batch_size):
        """Return list of session indices for each batch."""
        if self.shuffle:
            idx = np.arange(self.length)
            np.random.shuffle(idx)
            self.raw = self.raw[idx]
            self.targets = self.targets[idx]

        n_batch = (self.length + batch_size - 1) // batch_size
        slices = np.arange(n_batch * batch_size).reshape(n_batch, batch_size)
        slices[-1] = np.arange(max(0, self.length - batch_size), self.length)
        return slices

    # -------------------- Get batch slice --------------------
    def get_slice(self, index):
        """Return padded session items, mask, and target labels for batch."""
        items, lens, mask = [], [], []
        sessions = self.raw[index]

        max_len = max(len(s) for s in sessions) if sessions else 1

        for s in sessions:
            l = len(s)
            items.append(list(s) + [0] * (max_len - l))  # pad with 0
            lens.append([l])
            mask.append([1] * l + [0] * (max_len - l))

        return self.targets[index], lens, items, mask
