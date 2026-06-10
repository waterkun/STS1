#!/usr/bin/env python3
"""
PyTorch Transformer 排序模型 - 杀戮尖塔决策助手 V3 核心

使用 PyTorch 实现 Transformer 编码器，支持 CUDA 加速训练。
推理时兼容 CPU，接口保持 numpy 输入/输出。

架构:
  Input (N, F) → Linear(F, d_model) → N x TransformerBlock → Linear(d_model, 1) → scores (N,)

每个 TransformerBlock:
  Pre-norm: LayerNorm → MultiHeadAttention → Residual
            LayerNorm → FeedForward        → Residual
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Transformer Block (Pre-Norm)
# ============================================================

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer 编码块:
      x → LN → MHA → Dropout → x+residual → LN → FFN → Dropout → x+residual
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True,
                                          dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # x: (B, N, d_model), key_padding_mask: (B, N) True=ignore
        h1 = self.ln1(x)
        attn_out, _ = self.attn(h1, h1, h1, key_padding_mask=key_padding_mask)
        x2 = x + self.drop1(attn_out)

        h2 = self.ln2(x2)
        ffn_out = self.ffn(h2)
        return x2 + self.drop2(ffn_out)


# ============================================================
# Full Model: STSTransformerRanker
# ============================================================

class STSTransformerRanker(nn.Module):
    """
    杀戮尖塔 Transformer 排序模型。

    输入: N 个候选选项的特征向量 (N, F)
    输出: 每个选项的评分 (N,)，用于排序推荐

    与 V1/V2 模型的核心区别: 通过自注意力机制，
    每个选项的评分受其他选项影响（捕捉相对价值）。
    """

    def __init__(self, input_dim, d_model=64, n_heads=4, d_ff=128, n_layers=2,
                 dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.output_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 输出头用小权重
        nn.init.normal_(self.output_head.weight, std=0.01)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, x, key_padding_mask=None):
        """
        x: (B, N, input_dim) 或 (N, input_dim) → scores (B, N) 或 (N,)
        key_padding_mask: (B, N) bool tensor, True = padding position to ignore
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, N, F)
            squeeze = True

        h = self.input_drop(self.input_proj(x))  # (B, N, d_model)
        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask)
        scores = self.output_head(h).squeeze(-1)  # (B, N)

        if squeeze:
            scores = scores.squeeze(0)  # (N,)
        return scores

    @torch.no_grad()
    def predict(self, x):
        """推理: 接受 numpy 数组，返回归一化概率 numpy 数组 (N,)。"""
        self.eval()
        device = next(self.parameters()).device
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        scores = self.forward(x_t)   # (N,)
        probs = torch.softmax(scores, dim=-1)
        return probs.cpu().numpy()


# ============================================================
# 数据准备工具
# ============================================================

def decisions_from_ranking_data(X, y, groups):
    """
    将 V2 排序数据 (X, y, groups) 转换为逐决策列表。

    返回:
      decisions_X: list of (N_i, F) arrays
      decisions_y: list of (N_i,) label arrays
    """
    decisions_X, decisions_y = [], []
    start = 0
    for g in groups:
        g = int(g)
        decisions_X.append(X[start: start + g].astype(np.float32))
        decisions_y.append(y[start: start + g].astype(np.float32))
        start += g
    return decisions_X, decisions_y


# ============================================================
# 训练函数
# ============================================================

def _collate_decisions(batch_X, batch_y, device):
    """
    将一批变长决策 pad 成统一 tensor。

    返回:
      x_pad:  (B, max_N, F)  float32
      y_pad:  (B, max_N)     float32  (padding 位置为 -inf)
      mask:   (B, max_N)     bool     (True = padding，用于 attention mask)
    """
    lengths = [x.shape[0] for x in batch_X]
    max_n = max(lengths)
    feat_dim = batch_X[0].shape[1]
    B = len(batch_X)

    x_pad = np.zeros((B, max_n, feat_dim), dtype=np.float32)
    y_pad = np.full((B, max_n), -1e9, dtype=np.float32)  # padding 用 -inf
    mask = np.ones((B, max_n), dtype=bool)  # True = padding

    for i, (x, y, n) in enumerate(zip(batch_X, batch_y, lengths)):
        x_pad[i, :n] = x
        y_pad[i, :n] = y
        mask[i, :n] = False

    return (torch.from_numpy(x_pad).to(device),
            torch.from_numpy(y_pad).to(device),
            torch.from_numpy(mask).to(device))


def train_transformer(decisions_X, decisions_y,
                      d_model=128, n_heads=4, d_ff=256, n_layers=2,
                      n_epochs=60, lr=5e-4, batch_size=128,
                      name="model"):
    """
    训练一个 STSTransformerRanker。

    decisions_X: list of (N_i, F) feature arrays (每个决策一个)
    decisions_y: list of (N_i,) label arrays (2/1/0 相关性标签)
    返回: 训练好的 STSTransformerRanker (在 CPU 上)
    """
    if not decisions_X:
        print(f"  {name}: 无训练数据，跳过")
        return None

    # 过滤掉只有一个选项的决策
    pairs = [(x, y) for x, y in zip(decisions_X, decisions_y) if len(x) >= 2]
    if not pairs:
        print(f"  {name}: 所有决策只有一个选项，跳过")
        return None

    decisions_X, decisions_y = zip(*pairs)
    decisions_X = list(decisions_X)
    decisions_y = list(decisions_y)

    # 清洗数据：替换 nan/inf
    for i in range(len(decisions_X)):
        decisions_X[i] = np.nan_to_num(decisions_X[i], nan=0.0, posinf=0.0, neginf=0.0)
        decisions_y[i] = np.nan_to_num(decisions_y[i], nan=0.0, posinf=0.0, neginf=0.0)

    input_dim = decisions_X[0].shape[1]

    # 设备选择
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  使用设备: {torch.cuda.get_device_name(0)} (CUDA)")
    else:
        device = torch.device("cpu")
        print(f"  使用设备: CPU")

    print(f"  Transformer {name}: {len(decisions_X)} 个决策, 特征维度={input_dim}, "
          f"d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")

    # 按选项数分组排序，减少 padding 浪费
    sorted_indices = sorted(range(len(decisions_X)),
                            key=lambda i: decisions_X[i].shape[0])

    model = STSTransformerRanker(input_dim, d_model, n_heads, d_ff, n_layers)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Warmup + Cosine Decay: 前 5 epochs 线性 warmup (0.2→1.0)，之后 cosine decay 到 1%
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.2 + 0.8 * epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(n_epochs - warmup_epochs, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(n_epochs):
        # 每个 epoch 对排序后的索引做局部打乱（在相近长度内打乱）
        perm = sorted_indices.copy()
        # 在 batch 级别打乱 batch 顺序
        n_batches = (len(perm) + batch_size - 1) // batch_size
        batch_order = np.random.permutation(n_batches)

        total_loss = 0.0
        n_processed = 0

        for bi in batch_order:
            start = bi * batch_size
            end = min(start + batch_size, len(perm))
            batch_idx = perm[start:end]

            bX = [decisions_X[i] for i in batch_idx]
            bY = [decisions_y[i] for i in batch_idx]
            lengths = [x.shape[0] for x in bX]

            x_pad, y_pad, mask = _collate_decisions(bX, bY, device)
            # x_pad: (B, max_N, F), y_pad: (B, max_N), mask: (B, max_N)

            scores = model(x_pad, key_padding_mask=mask)  # (B, max_N)

            # Pairwise Margin Loss: 对 label[i] > label[j] 的对，
            # 计算 softplus(score_j - score_i)
            # 向量化实现（广播），O(n^2) 但 n 通常只有 3-5
            valid = ~mask  # (B, max_N), True = 有效位置
            # 构造 pairwise 差异
            s_i = scores.unsqueeze(2)  # (B, N, 1)
            s_j = scores.unsqueeze(1)  # (B, 1, N)
            y_i = y_pad.unsqueeze(2)   # (B, N, 1)
            y_j = y_pad.unsqueeze(1)   # (B, 1, N)
            # 有效对: 两个位置都非 padding 且 label_i > label_j
            valid_mask = valid.unsqueeze(2) & valid.unsqueeze(1)  # (B, N, N)
            pair_mask = (y_i > y_j) & valid_mask  # (B, N, N)
            # softplus(score_j - score_i) 对应 label_i > label_j 的对
            pair_loss = F.softplus(s_j - s_i)  # (B, N, N)
            pair_loss = pair_loss * pair_mask.float()
            n_pairs = pair_mask.float().sum()
            loss = pair_loss.sum() / n_pairs.clamp(min=1.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_idx)
            n_processed += len(batch_idx)

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg = total_loss / max(n_processed, 1)
            current_lr = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch+1:>2d}/{n_epochs}  loss={avg:.4f}  lr={current_lr:.6f}")

    print(f"  Transformer {name} 训练完成")

    # 训练完成后移回 CPU 以便序列化和跨设备推理
    model.cpu()
    model.eval()
    return model
