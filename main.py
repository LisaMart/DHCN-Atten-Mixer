import argparse
import pickle
import warnings
import numpy as np
from util import Data, split_validation
from model import DHCN, train_test
import os

# Ignore warnings when dividing by zero in sparse normalisation
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero encountered.*")

# -------------------- 1. Load sessions --------------------
def load_sessions(path: str, dataset: str):
    """
    Load sessions from file and return dense indices.
    Returns:
        sessions : list of lists – session item indices
        labels   : list – next-item labels
        item2idx : dict – mapping original item ids -> dense 1-based indices (0 = padding)
    """
    sessions, labels = [], []

    if dataset == 'diginetica':
        sess_raw, label_raw = pickle.load(open(path, 'rb'))
        raw_sessions, raw_labels = sess_raw, label_raw
    else:
        raw_sessions, raw_labels = [], []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                items = list(map(int, line.strip().split()))
                if len(items) < 2:  # at least 1 event + label
                    continue
                raw_sessions.append(items[:-1])
                raw_labels.append(items[-1])

    # Build dense vocabulary
    all_items = set(raw_labels)
    for seq in raw_sessions:
        all_items.update(seq)
    item2idx = {item: idx + 1 for idx, item in enumerate(sorted(all_items))}

    sessions = [[item2idx[i] for i in seq] for seq in raw_sessions]
    labels = [item2idx[i] for i in raw_labels]

    return sessions, labels, item2idx

# -------------------- 2. CLI arguments --------------------
parser = argparse.ArgumentParser(description='DHCN without GNN propagation')
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset folder name')
parser.add_argument('--epoch', type=int, default=2, help='number of training epochs')
parser.add_argument('--batchSize', type=int, default=100, help='mini-batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='L2 weight decay')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
opt = parser.parse_args()
print(opt)

# -------------------- 3. Main pipeline --------------------
def main():
    import os
    # Universal dataset path relative to script
    base_path = os.path.join(os.path.dirname(__file__), 'datasets')
    train_path = os.path.join(base_path, opt.dataset, 'train.txt')
    test_path  = os.path.join(base_path, opt.dataset, 'test.txt')

    # Load train and test sessions
    tr_sess, tr_lab, tr_dict = load_sessions(train_path, opt.dataset)
    te_sess, te_lab, _       = load_sessions(test_path,  opt.dataset)

    # Enlarge dictionary with test-only items
    for seq in te_sess:
        for i in seq:
            if i not in tr_dict:
                tr_dict[i] = len(tr_dict) + 1
    for i in te_lab:
        if i not in tr_dict:
            tr_dict[i] = len(tr_dict) + 1

    n_node = len(tr_dict)
    print(f'Dense vocab size (n_node): {n_node}')

    # Wrap data into Data objects (adjacency built but ignored)
    train_data = Data((tr_sess, tr_lab), shuffle=True, n_node=n_node)
    test_data  = Data((te_sess, te_lab), shuffle=True, n_node=n_node)

    # Instantiate DHCN model
    model = DHCN(n_node=n_node, lr=opt.lr, l2=opt.l2, emb_size=opt.embSize, batch_size=opt.batchSize)

    # Train and evaluate
    top_K = [5, 10, 20]
    best_results = {f'epoch{K}': [0, 0] for K in top_K}
    for K in top_K:
        best_results[f'precision{K}'] = [0.0, 0.0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch:', epoch)
        metrics, total_loss = train_test(model, train_data, test_data)

        # Update best metrics
        for K in top_K:
            p = np.mean(metrics[f'precision{K}']) * 100
            m = np.mean(metrics[f'mrr{K}']) * 100

            if best_results[f'precision{K}'][0] < p:
                best_results[f'precision{K}'][0] = p
                best_results[f'epoch{K}'][0] = epoch
            if best_results[f'precision{K}'][1] < m:
                best_results[f'precision{K}'][1] = m
                best_results[f'epoch{K}'][1] = epoch

        # Print results
        for K in top_K:
            print(f'train_loss:{total_loss:.4f}\t'
                  f'Precision@{K}:{best_results[f"precision{K}"][0]:.4f}\t'
                  f'MRR{K}:{best_results[f"precision{K}"][1]:.4f}\t'
                  f'Epoch:{best_results[f"epoch{K}"][0]},{best_results[f"epoch{K}"][1]}')

if __name__ == '__main__':
    main()
    
