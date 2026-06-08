#!/usr/bin/env python3
"""
纯 NumPy Transformer 排序模型 - 杀戮尖塔决策助手 V3 核心

实现轻量级 Transformer 编码器，允许候选选项之间互相注意，
捕捉选项间的相对优劣关系（协同效应、机会成本等）。

架构:
  Input (N, F) → Linear(F, d_model) → N x TransformerBlock → Linear(d_model, 1) → scores (N,)

每个 TransformerBlock:
  Pre-norm: LayerNorm → MultiHeadAttention → Residual
            LayerNorm → FeedForward        → Residual
"""

import numpy as np
import pickle
from pathlib import Path


# ============================================================
# 基础数学工具
# ============================================================

def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _softmax_backward_rows(probs, dout):
    """
    对 softmax 的行向量批量反向传播。
    probs, dout: (B, N)
    返回: dL/d_input, shape (B, N)
    公式: ds_i = p_i * (do_i - sum_j(do_j * p_j))
    """
    dots = (dout * probs).sum(axis=-1, keepdims=True)  # (B, 1)
    return probs * (dout - dots)


def _relu(x):
    return np.maximum(0.0, x)


def _relu_grad(x):
    return (x > 0).astype(np.float32)


# ============================================================
# LayerNorm
# ============================================================

class LayerNorm:
    """沿最后一维做层归一化，支持批量 (N, d)。"""

    def __init__(self, d, eps=1e-5):
        self.gamma = np.ones(d, dtype=np.float32)
        self.beta = np.zeros(d, dtype=np.float32)
        self.eps = eps
        self._cache = None

    def forward(self, x):
        # x: (N, d)
        mean = x.mean(axis=-1, keepdims=True)          # (N, 1)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)  # (N, 1)
        std_inv = 1.0 / np.sqrt(var + self.eps)         # (N, 1)
        x_norm = (x - mean) * std_inv                   # (N, d)
        self._cache = (x_norm, std_inv)
        return self.gamma * x_norm + self.beta           # (N, d)

    def backward(self, dout):
        """
        返回 (dx, dgamma, dbeta)
        标准层归一化梯度（对最后一维）。
        """
        x_norm, std_inv = self._cache
        N, d = x_norm.shape

        dgamma = (dout * x_norm).sum(axis=0)             # (d,)
        dbeta = dout.sum(axis=0)                          # (d,)

        dx_norm = dout * self.gamma                       # (N, d)
        # dx = std_inv/d * (d*dx_norm - sum(dx_norm) - x_norm*sum(dx_norm*x_norm))
        dx = std_inv / d * (
            d * dx_norm
            - dx_norm.sum(axis=-1, keepdims=True)
            - x_norm * (dx_norm * x_norm).sum(axis=-1, keepdims=True)
        )
        return dx, dgamma, dbeta


# ============================================================
# Multi-Head Self-Attention
# ============================================================

class MultiHeadAttention:
    """缩放点积多头自注意力（Q=K=V=输入）。"""

    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        s = np.sqrt(2.0 / d_model)
        self.Wq = (np.random.randn(d_model, d_model) * s).astype(np.float32)
        self.Wk = (np.random.randn(d_model, d_model) * s).astype(np.float32)
        self.Wv = (np.random.randn(d_model, d_model) * s).astype(np.float32)
        self.Wo = (np.random.randn(d_model, d_model) * s).astype(np.float32)
        self.bo = np.zeros(d_model, dtype=np.float32)

        self._cache = None

    def forward(self, x):
        # x: (N, d_model)
        N, H, d_h = len(x), self.n_heads, self.d_head
        scale = 1.0 / np.sqrt(d_h)

        Q = (x @ self.Wq).reshape(N, H, d_h).transpose(1, 0, 2)  # (H, N, d_h)
        K = (x @ self.Wk).reshape(N, H, d_h).transpose(1, 0, 2)
        V = (x @ self.Wv).reshape(N, H, d_h).transpose(1, 0, 2)

        S = np.matmul(Q, K.transpose(0, 2, 1)) * scale            # (H, N, N)
        A = _softmax(S, axis=-1)                                    # (H, N, N)

        out_h = np.matmul(A, V)                                    # (H, N, d_h)
        out_flat = out_h.transpose(1, 0, 2).reshape(N, -1)        # (N, d_model)

        result = out_flat @ self.Wo + self.bo                      # (N, d_model)
        self._cache = (x, Q, K, V, S, A, out_flat)
        return result

    def backward(self, dout):
        """
        返回 (dx, param_grads_list)
        param_grads_list: [(param_array, grad_array), ...]
        """
        x, Q, K, V, S, A, out_flat = self._cache
        N, H, d_h = len(x), self.n_heads, self.d_head
        scale = 1.0 / np.sqrt(d_h)

        # 输出投影层
        dWo = out_flat.T @ dout                                    # (d_model, d_model)
        dbo = dout.sum(axis=0)                                     # (d_model,)
        d_out_flat = dout @ self.Wo.T                              # (N, d_model)

        # 转回多头格式 (H, N, d_h)
        d_out_h = d_out_flat.reshape(N, H, d_h).transpose(1, 0, 2)

        # matmul(A, V) 反向
        dA = np.matmul(d_out_h, V.transpose(0, 2, 1))             # (H, N, N)
        dV_h = np.matmul(A.transpose(0, 2, 1), d_out_h)           # (H, N, d_h)

        # Softmax 反向 (对每个 head 和 query 位置)
        dS = _softmax_backward_rows(
            A.reshape(H * N, N), dA.reshape(H * N, N)
        ).reshape(H, N, N) * scale                                 # (H, N, N)

        # Q @ K^T 反向
        dQ_h = np.matmul(dS, K)                                    # (H, N, d_h)
        dK_h = np.matmul(dS.transpose(0, 2, 1), Q)                # (H, N, d_h)

        # 转回 (N, d_model)
        dQ = dQ_h.transpose(1, 0, 2).reshape(N, -1)
        dK = dK_h.transpose(1, 0, 2).reshape(N, -1)
        dV = dV_h.transpose(1, 0, 2).reshape(N, -1)

        dWq = x.T @ dQ
        dWk = x.T @ dK
        dWv = x.T @ dV

        dx = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T

        param_grads = [
            (self.Wq, dWq), (self.Wk, dWk), (self.Wv, dWv),
            (self.Wo, dWo), (self.bo, dbo),
        ]
        return dx, param_grads


# ============================================================
# Feed-Forward Network
# ============================================================

class FeedForward:
    """逐位置前馈网络: Linear → ReLU → Linear。"""

    def __init__(self, d_model, d_ff):
        s1 = np.sqrt(2.0 / d_model)
        s2 = np.sqrt(2.0 / d_ff)
        self.W1 = (np.random.randn(d_model, d_ff) * s1).astype(np.float32)
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.W2 = (np.random.randn(d_ff, d_model) * s2).astype(np.float32)
        self.b2 = np.zeros(d_model, dtype=np.float32)
        self._cache = None

    def forward(self, x):
        h = x @ self.W1 + self.b1          # (N, d_ff)
        h_relu = _relu(h)
        out = h_relu @ self.W2 + self.b2   # (N, d_model)
        self._cache = (x, h, h_relu)
        return out

    def backward(self, dout):
        x, h, h_relu = self._cache

        dW2 = h_relu.T @ dout
        db2 = dout.sum(axis=0)
        d_h_relu = dout @ self.W2.T

        d_h = d_h_relu * _relu_grad(h)

        dW1 = x.T @ d_h
        db1 = d_h.sum(axis=0)
        dx = d_h @ self.W1.T

        param_grads = [(self.W1, dW1), (self.b1, db1), (self.W2, dW2), (self.b2, db2)]
        return dx, param_grads


# ============================================================
# Transformer Block (Pre-Norm)
# ============================================================

class TransformerBlock:
    """
    Pre-norm Transformer 编码块:
      x → LN → MHA → x+residual → LN → FFN → x+residual
    """

    def __init__(self, d_model, n_heads, d_ff):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self._cache = None

    def forward(self, x):
        # Attention sub-layer
        h1 = self.ln1.forward(x)
        attn_out = self.attn.forward(h1)
        x2 = x + attn_out                  # residual

        # FFN sub-layer
        h2 = self.ln2.forward(x2)
        ffn_out = self.ffn.forward(h2)
        out = x2 + ffn_out                 # residual

        self._cache = (x, h1, x2, h2)
        return out

    def backward(self, dout):
        x, h1, x2, h2 = self._cache
        all_grads = []

        # FFN residual: dout 流向 x2 分支和 ffn_out 分支
        d_ffn_out = dout
        d_h2, ffn_grads = self.ffn.backward(d_ffn_out)
        dx2_ln2, dgamma2, dbeta2 = self.ln2.backward(d_h2)
        all_grads += ffn_grads
        all_grads += [(self.ln2.gamma, dgamma2), (self.ln2.beta, dbeta2)]

        dx2 = dout + dx2_ln2               # residual + ln2 path

        # Attention residual
        d_attn_out = dx2
        d_h1, attn_grads = self.attn.backward(d_attn_out)
        dx_ln1, dgamma1, dbeta1 = self.ln1.backward(d_h1)
        all_grads += attn_grads
        all_grads += [(self.ln1.gamma, dgamma1), (self.ln1.beta, dbeta1)]

        dx = dx2 + dx_ln1                  # residual + ln1 path

        return dx, all_grads


# ============================================================
# Full Model: STSTransformerRanker
# ============================================================

class STSTransformerRanker:
    """
    杀戮尖塔 Transformer 排序模型。

    输入: N 个候选选项的特征向量 (N, F)
    输出: 每个选项的评分 (N,)，用于排序推荐

    与 V1/V2 模型的核心区别: 通过自注意力机制，
    每个选项的评分受其他选项影响（捕捉相对价值）。
    """

    def __init__(self, input_dim, d_model=64, n_heads=4, d_ff=128, n_layers=2):
        self.input_dim = input_dim
        self.d_model = d_model

        np.random.seed(42)

        # 输入投影
        s = np.sqrt(2.0 / input_dim)
        self.W_in = (np.random.randn(input_dim, d_model) * s).astype(np.float32)
        self.b_in = np.zeros(d_model, dtype=np.float32)

        # Transformer 块
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]

        # 输出头
        self.W_out = (np.random.randn(d_model, 1) * 0.01).astype(np.float32)
        self.b_out = np.zeros(1, dtype=np.float32)

        self._cache = None

    def forward(self, x):
        """x: (N, input_dim) → scores (N,)"""
        h = x @ self.W_in + self.b_in     # (N, d_model)
        for block in self.blocks:
            h = block.forward(h)
        h_final = h
        scores = (h_final @ self.W_out + self.b_out).squeeze(-1)   # (N,)
        self._cache = (x, h_final)
        return scores

    def loss_and_grad(self, x, labels):
        """
        ListNet 排序损失: 对标签做 softmax 得到目标分布，
        对预测分做 softmax 得到预测分布，交叉熵作为损失。

        x: (N, F)
        labels: (N,) — 2=选中且赢, 1=选中且输, 0=未选
        """
        scores = self.forward(x)
        pred_probs = _softmax(scores)                                # (N,)
        target_probs = _softmax(labels.astype(np.float32))          # (N,)

        loss = -np.sum(target_probs * np.log(pred_probs + 1e-10))
        # softmax 交叉熵梯度: pred - target
        d_scores = pred_probs - target_probs                        # (N,)
        return loss, d_scores

    def backward(self, d_scores):
        """反向传播，返回 [(param, grad), ...] 列表。"""
        x, h_final = self._cache

        # 输出头反向
        d_s = d_scores[:, np.newaxis]                               # (N, 1)
        dW_out = h_final.T @ d_s                                    # (d_model, 1)
        db_out = d_s.sum(axis=0)                                    # (1,)
        d_h = d_s @ self.W_out.T                                    # (N, d_model)

        all_grads = [(self.W_out, dW_out), (self.b_out, db_out)]

        # Transformer 块反向（倒序）
        for block in reversed(self.blocks):
            d_h, block_grads = block.backward(d_h)
            all_grads.extend(block_grads)

        # 输入投影反向
        dW_in = x.T @ d_h                                           # (F, d_model)
        db_in = d_h.sum(axis=0)                                     # (d_model,)
        all_grads.extend([(self.W_in, dW_in), (self.b_in, db_in)])

        return all_grads

    def predict(self, x):
        """推理: 返回归一化概率 (N,)。"""
        scores = self.forward(x)
        return _softmax(scores)


# ============================================================
# Adam 优化器
# ============================================================

class Adam:
    """Adam 优化器 (numpy 实现)，按参数 id 追踪动量。"""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._m = {}
        self._v = {}

    def step(self, param_grads):
        self.t += 1
        b1t = 1.0 - self.beta1 ** self.t
        b2t = 1.0 - self.beta2 ** self.t

        for param, grad in param_grads:
            pid = id(param)
            if pid not in self._m:
                self._m[pid] = np.zeros_like(param)
                self._v[pid] = np.zeros_like(param)

            self._m[pid] = self.beta1 * self._m[pid] + (1 - self.beta1) * grad
            self._v[pid] = self.beta2 * self._v[pid] + (1 - self.beta2) * grad ** 2

            m_hat = self._m[pid] / b1t
            v_hat = self._v[pid] / b2t

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


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

def train_transformer(decisions_X, decisions_y,
                      d_model=64, n_heads=4, d_ff=128, n_layers=2,
                      n_epochs=25, lr=5e-4, batch_size=128,
                      name="model"):
    """
    训练一个 STSTransformerRanker。

    decisions_X: list of (N_i, F) feature arrays (每个决策一个)
    decisions_y: list of (N_i,) label arrays (2/1/0 相关性标签)
    返回: 训练好的 STSTransformerRanker
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

    input_dim = decisions_X[0].shape[1]
    print(f"  Transformer {name}: {len(decisions_X)} 个决策, 特征维度={input_dim}, "
          f"d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")

    model = STSTransformerRanker(input_dim, d_model, n_heads, d_ff, n_layers)
    optimizer = Adam(lr=lr)

    indices = list(range(len(decisions_X)))

    for epoch in range(n_epochs):
        np.random.shuffle(indices)
        total_loss = 0.0
        n_processed = 0

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start: batch_start + batch_size]

            # 累积梯度（按参数 id 索引）
            accum: dict[int, tuple] = {}
            batch_loss = 0.0
            count = 0

            for idx in batch_idx:
                x = decisions_X[idx]
                lbl = decisions_y[idx]

                loss, d_scores = model.loss_and_grad(x, lbl)
                batch_loss += loss

                param_grads = model.backward(d_scores)
                for param, grad in param_grads:
                    pid = id(param)
                    if pid not in accum:
                        accum[pid] = (param, np.zeros_like(grad))
                    accum[pid] = (param, accum[pid][1] + grad)

                count += 1

            if count == 0:
                continue

            # 平均梯度后更新
            avg_grads = [(p, g / count) for p, g in accum.values()]
            optimizer.step(avg_grads)

            total_loss += batch_loss
            n_processed += count

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg = total_loss / max(n_processed, 1)
            print(f"    Epoch {epoch+1:>2d}/{n_epochs}  loss={avg:.4f}")

    print(f"  Transformer {name} 训练完成")
    return model
