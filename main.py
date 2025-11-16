# main_no_gnn.py  (без GNN-пропагации, читает yoochoose1_64, плотные id)
import argparse
import pickle
import warnings
import numpy as np
from util import Data, split_validation
from model import DHCN, train_test
import os

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*divide by zero encountered.*")

# ---------- чтение данных ------------------------------------------------------
def load_sessions(path, dataset):
    sessions, labels = [], []

    # 1. читаем «сырые» сессии
    if dataset == 'diginetica':
        sess_raw, label_raw = pickle.load(open(path, 'rb'))
        raw_sessions, raw_labels = sess_raw, label_raw
    else:                       # plain txt
        raw_sessions, raw_labels = [], []
        with open(path, 'r') as f:
            for line in f:
                items = list(map(int, line.strip().split()))
                if len(items) < 2:
                    continue
                raw_sessions.append(items[:-1])
                raw_labels.append(items[-1])

    # 2. строим общий словарь (0 оставляем под padding)
    all_items = set(raw_labels)
    for seq in raw_sessions:
        all_items.update(seq)

    item2idx = {item: idx + 1 for idx, item in enumerate(sorted(all_items))}

    # 3. перекодируем
    sessions = [[item2idx[i] for i in seq] for seq in raw_sessions]
    labels   = [item2idx[i] for i in raw_labels]
    return sessions, labels, item2idx


# ---------- CLI --------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64',
                    help='dataset name: diginetica/yoochoose1_64/sample')
parser.add_argument('--epoch', type=int, default=2)
parser.add_argument('--batchSize', type=int, default=100)
parser.add_argument('--embSize', type=int, default=100)
parser.add_argument('--l2', type=float, default=1e-5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--layer', type=int, default=3)
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--filter', type=bool, default=False)
opt = parser.parse_args()
print(opt)


# ---------- main -------------------------------------------------------------
def main():
    train_path = f'./datasets/{opt.dataset}/train.txt'
    test_path  = f'./datasets/{opt.dataset}/test.txt'

    # 1. читаем train и test
    tr_sess, tr_lab, tr_dict = load_sessions(train_path, opt.dataset)
    te_sess, te_lab, _       = load_sessions(test_path,  opt.dataset)

    # 2. добавляем в словарь айтемы из test, которых не было в train
    for seq in te_sess:
        for i in seq:
            if i not in tr_dict:
                tr_dict[i] = len(tr_dict) + 1
    for i in te_lab:
        if i not in tr_dict:
            tr_dict[i] = len(tr_dict) + 1

    # 3. перекодируем test тем же словарём
    te_sess = [[tr_dict[i] for i in s] for s in te_sess]
    te_lab  = [tr_dict[i] for i in te_lab]

    n_node = len(tr_dict)
    print(f'Dense n_node: {n_node}')

    # 4. формируем объекты Data
    train_data = Data((tr_sess, tr_lab), shuffle=True, n_node=n_node)
    test_data  = Data((te_sess, te_lab), shuffle=True, n_node=n_node)

    # 5. модель без графовой матрицы
    model = DHCN(n_node=n_node,
                 lr=opt.lr,
                 l2=opt.l2,
                 emb_size=opt.embSize,
                 batch_size=opt.batchSize)

    # 6. обучение
    top_K = [5, 10, 20]
    best_results = {f'epoch{K}': [0, 0] for K in top_K}
    for K in top_K:
        best_results[f'precision{K}'] = [0.0, 0.0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch:', epoch)
        metrics, total_loss = train_test(model, train_data, test_data)

        for K in top_K:
            p = np.mean(metrics[f'precision{K}']) * 100
            m = np.mean(metrics[f'mrr{K}']) * 100
            if best_results[f'precision{K}'][0] < p:
                best_results[f'precision{K}'][0] = p
                best_results[f'epoch{K}'][0] = epoch
            if best_results[f'precision{K}'][1] < m:
                best_results[f'precision{K}'][1] = m
                best_results[f'epoch{K}'][1] = epoch

        for K in top_K:
            print(f'train_loss:{total_loss:.4f}\t'
                  f'Precision@{K}:{best_results[f"precision{K}"][0]:.4f}\t'
                  f'MRR{K}:{best_results[f"precision{K}"][1]:.4f}\t'
                  f'Epoch:{best_results[f"epoch{K}"][0]},{best_results[f"epoch{K}"][1]}')

if __name__ == '__main__':
    main()