import datetime
import numpy as np
import torch
from torch import nn
import torch.sparse
from numba import jit
import heapq
import sys

# Убираем трансфер на GPU
def trans_to_cuda(variable):
    return variable  # всегда CPU

def trans_to_cpu(variable):
    return variable  # всегда CPU

class AttenMixer(nn.Module):
    """
    3-уровневый attention mixer (K=3, h=2 heads)
    """
    def __init__(self, emb_size, max_len=200, n_heads=2, levels=3):
        super().__init__()
        self.levels = levels
        self.emb_size = emb_size
        self.scale = emb_size ** -0.5

        # позиционные эмбеддинги для 3 уровней
        self.pos_embed = nn.Parameter(torch.randn(levels, max_len, emb_size))

        # multi-head attention
        self.to_qkv = nn.Linear(emb_size, emb_size * 3, bias=False)
        self.out_proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        """
        x: (B, L, d) – последовательность items в сессии
        mask: (B, L) – 1 реальный токен, 0 паддинг
        """
        B, L, d = x.size()
        x = x.unsqueeze(1).repeat(1, self.levels, 1, 1)  # (B, K, L, d)
        x = x + self.pos_embed[:, :L, :]                 # добавляем позиции

        # flatten levels & seq_len
        x = x.view(B * self.levels, L, d)
        mask = mask.unsqueeze(1).repeat(1, self.levels, 1).view(B * self.levels, L)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B * self.levels, -1, 2, d // 2).transpose(1, 2),
                      qkv)  # (B*K, h, L, d/h)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = mask.unsqueeze(1).unsqueeze(2)  # (B*K, 1, 1, L)
        scores.masked_fill_(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B*K, h, L, d/h)
        out = out.transpose(1, 2).contiguous().view(B * self.levels, L, d)
        out = self.out_proj(out)
        out = out.view(B, self.levels, L, d)

        # просто усредняем уровни (можно заменить на learnable mixing)
        session_emb = out.mean(dim=2).mean(dim=1)   # (B, d)
        return session_emb

class DHCN(nn.Module):
    def __init__(self,
                 n_node: int,
                 lr: float,
                 l2: float,
                 emb_size: int = 100,
                 batch_size: int = 100):
        super().__init__()
        self.emb_size   = emb_size
        self.batch_size = batch_size
        self.n_node     = n_node
        self.L2         = l2
        self.lr         = lr

        # эмбеддинги items
        self.embedding = nn.Embedding(self.n_node, self.emb_size)

        # attention-микер
        self.atten_mixer = AttenMixer(emb_size, max_len=200, n_heads=2, levels=3)

        # loss + оптимизатор
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer     = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

        self._init_weights()

    # инициализация
    def _init_weights(self):
        stdv = 1.0 / (self.emb_size ** 0.5)
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    # единственный forward
    def forward(self, session_item, mask):
        """
        session_item: (B, L)  – идентификаторы items в сессии
        mask        : (B, L)  – 1 = реальный токен, 0 = PAD
        return      : (B, n_node) – логиты по всем товарам
        """
        item_emb = self.embedding(session_item)          # (B, L, d)
        sess_emb = self.atten_mixer(item_emb, mask)      # (B, d)
        scores   = torch.matmul(sess_emb,
                                self.embedding.weight.T) # (B, n_node)
        return scores

@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    heapq.heapify(n_candidates)
    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))
    n_candidates.sort(key=lambda d: d[0], reverse=True)
    ids = [item[1] for item in n_candidates]
    return ids

def _progress_bar(current, total, width=40, prefix='Progress', suffix=''):
    ratio = current / max(total, 1)
    filled = int(width * ratio)
    bar   = '█' * filled + '-' * (width - filled)
    percent = f'{100*ratio:5.1f}%'
    sys.stdout.write(f'\r{prefix} |{bar}| {percent} {suffix}')
    sys.stdout.flush()

def train_test(model, train_data, test_data):
    @jit(nopython=True)
    def find_k_largest(K, candidates):
        n_candidates = []
        for iid, score in enumerate(candidates[:K]):
            n_candidates.append((score, iid))
        heapq.heapify(n_candidates)
        for iid, score in enumerate(candidates[K:]):
            if score > n_candidates[0][0]:
                heapq.heapreplace(n_candidates, (score, iid + K))
        n_candidates.sort(key=lambda d: d[0], reverse=True)
        return [item[1] for item in n_candidates]

    def _progress_bar(current, total, width=40, prefix='Progress', suffix=''):
        ratio = current / max(total, 1)
        filled = int(width * ratio)
        bar = '█' * filled + '-' * (width - filled)
        percent = f'{100*ratio:5.1f}%'
        sys.stdout.write(f'\r{prefix} |{bar}| {percent} {suffix}')
        sys.stdout.flush()

    # ---------- TRAIN ----------
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for idx, i in enumerate(slices, 1):
        model.zero_grad()
        targets, session_len, session_item, mask = train_data.get_slice(i)
        session_item = torch.tensor(session_item, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)

        scores = model(session_item, mask)
        loss = model.loss_function(scores, targets)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
        _progress_bar(idx, len(slices), prefix='Train')
    print('\n\tLoss:\t%.3f' % total_loss)

    # ---------- TEST ----------
    top_K = [5, 10, 20]
    metrics = {f'precision{k}': [] for k in top_K}
    metrics.update({f'mrr{k}': [] for k in top_K})
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    with torch.no_grad():
        for idx, i in enumerate(slices, 1):
            tar, session_len, session_item, mask = test_data.get_slice(i)
            session_item = torch.tensor(session_item, dtype=torch.long)
            mask = torch.tensor(mask, dtype=torch.long)
            tar = torch.tensor(tar, dtype=torch.long)

            scores = model(session_item, mask).detach().numpy()
            index = [find_k_largest(20, score) for score in scores]
            index = np.array(index)
            tar = tar.numpy()

            for K in top_K:
                for pred, target in zip(index[:, :K], tar):
                    hit = np.isin(target, pred)
                    metrics[f'precision{K}'].append(float(hit))
                    idxs = np.where(pred == target)[0]
                    metrics[f'mrr{K}'].append(0 if len(idxs) == 0 else 1 / (idxs[0] + 1))
            _progress_bar(idx, len(slices), prefix='Test ')
    print()
    return metrics, total_loss