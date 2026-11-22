# -*- coding: utf-8 -*-
"""
人工智能概论 · 第二次作业
第二篇 15 节：MLP 学习参数的实现与可视化（XOR 问题）

纯 Python 从零实现，不需要 numpy / torch / tensorflow：
    2 -> 2 -> 1 多层感知机 + Sigmoid + 二元交叉熵 + 手写反向传播 + Adam

参数共 9 个，按扁平数组存放：
    [W1_11, W1_12, W1_21, W1_22, b1_1, b1_2, v_1, v_2, c]
其中 W1[i][j] 表示第 i 个输入到第 j 个隐藏节点的权重，
v 是隐藏层到输出层的两个权重，c 是输出偏置。

用法：
    python mlp_xor.py                      # 默认配置训练并打印结果
    python mlp_xor.py 1500 0.05 2          # 轮数 / 学习率 / 随机种子
    python mlp_xor.py --json out.json      # 额外导出每轮 loss 与参数快照
"""

import json
import math
import sys

X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 0.0]          # XOR 真值表
N = len(X)


class LCG:
    """32 位线性同余随机数：保证 Python 与网页 JavaScript 初始化完全一致。"""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def uniform(self, a, b):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return a + (b - a) * (self.s / 4294967296.0)


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def make_params(seed=2, scale=1.2):
    """9 个可学习参数：W1(4) + b1(2) + v(2) + c(1)，偏置初始为 0。"""
    rng = LCG(seed)
    W1 = [rng.uniform(-scale, scale) for _ in range(4)]   # w11 w12 w21 w22
    b1 = [0.0, 0.0]
    v = [rng.uniform(-scale, scale) for _ in range(2)]     # v1 v2
    c = 0.0
    return W1 + b1 + v + [c]


def forward(p, x):
    """一次前向传播，返回 (h1, h2, 预测概率)。"""
    z1 = x[0] * p[0] + x[1] * p[2] + p[4]
    z2 = x[0] * p[1] + x[1] * p[3] + p[5]
    h1, h2 = sigmoid(z1), sigmoid(z2)
    y_hat = sigmoid(h1 * p[6] + h2 * p[7] + p[8])
    return h1, h2, y_hat


def loss_of(p):
    total = 0.0
    for i in range(N):
        y_hat = forward(p, X[i])[2]
        y_hat = min(max(y_hat, 1e-12), 1 - 1e-12)
        total -= Y[i] * math.log(y_hat) + (1 - Y[i]) * math.log(1 - y_hat)
    return total / N


def gradients(p):
    """手写反向传播：链式法则求 9 个参数对损失的梯度（全批量平均）。"""
    g = [0.0] * 9
    for i in range(N):
        x = X[i]
        h1, h2, y_hat = forward(p, x)
        d2 = y_hat - Y[i]                      # sigmoid + BCE 的简化输出层误差
        g[6] += d2 * h1                       # dv1
        g[7] += d2 * h2                       # dv2
        g[8] += d2                            # dc
        d1a = d2 * p[6] * h1 * (1 - h1)        # 误差经 W2 回传到隐藏层
        d1b = d2 * p[7] * h2 * (1 - h2)
        g[0] += d1a * x[0]                    # dW1_11
        g[1] += d1b * x[0]                    # dW1_12
        g[2] += d1a * x[1]                    # dW1_21
        g[3] += d1b * x[1]                    # dW1_22
        g[4] += d1a                           # db1_1
        g[5] += d1b                           # db1_2
    return [v / N for v in g]


def accuracy(p):
    ok = 0
    for i in range(N):
        ok += (forward(p, X[i])[2] >= 0.5) == (Y[i] >= 0.5)
    return ok / N


def train(epochs=1500, lr=0.05, seed=2):
    """全批量 Adam 训练，返回 (最终参数, 每轮 loss, 每轮参数快照)。

    注：seed=2 可稳定收敛；改成 seed=20261122 会卡在 loss≈0.3467 的局部极小，
    正好用来观察「初始化会影响参数学习」这一现象。
    """
    p = make_params(seed)
    m, v = [0.0] * 9, [0.0] * 9
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    losses, snaps = [], []
    for t in range(1, epochs + 1):
        g = gradients(p)
        for j in range(9):
            m[j] = b1a * m[j] + (1 - b1a) * g[j]
            v[j] = b2a * v[j] + (1 - b2a) * g[j] * g[j]
            mh = m[j] / (1 - b1a ** t)
            vh = v[j] / (1 - b2a ** t)
            p[j] -= lr * mh / (math.sqrt(vh) + eps)
        losses.append(loss_of(p))
        snaps.append(p[:])
    return p, losses, snaps


def main():
    args = sys.argv[1:]
    if args and args[0] == "--json":
        epochs, lr, seed, out = 1500, 0.05, 2, args[1]
    else:
        epochs = int(args[0]) if len(args) > 0 else 1500
        lr = float(args[1]) if len(args) > 1 else 0.05
        seed = int(args[2]) if len(args) > 2 else 2
        out = None

    p, losses, snaps = train(epochs, lr, seed)
    p0 = make_params(seed)

    print("=" * 62)
    print(" 人工智能概论 · 第二次作业 / MLP 学习参数的实现")
    print(f" 结构 2→2→1 | Sigmoid | BCE | Adam | {epochs} 轮 | lr={lr} | seed={seed}")
    print("=" * 62)
    print(" 初始参数 :", " ".join(f"{v: .4f}" for v in p0))
    print(" 训练后参数:", " ".join(f"{v: .4f}" for v in p))
    print(f" 初始 loss {loss_of(p0):.6f}  ->  最终 loss {losses[-1]:.6f}")
    print(f" 初始准确率 {accuracy(p0)*100:.0f}%  ->  最终准确率 {accuracy(p)*100:.0f}%")
    print("-" * 62)
    print(" 输入      目标    预测概率    判定")
    for i in range(N):
        yh = forward(p, X[i])[2]
        print(f" {X[i]}    {int(Y[i])}      {yh:.4f}      {'✓' if (yh >= .5) == (Y[i] >= .5) else '✗'}")
    print("-" * 62)
    print(" loss 变化：", end="")
    for t in [0, epochs // 20, epochs // 5, epochs // 2, epochs - 1]:
        if t < len(losses):
            print(f" ep{t + 1}={losses[t]:.4f} ", end="")
    print()

    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"epochs": epochs, "lr": lr, "seed": seed,
                       "params": p, "loss": losses,
                       "snapshots": [snaps[i] for i in range(0, epochs, max(1, epochs // 200))]},
                      f)
        print(f" 已导出训练历史：{out}")


if __name__ == "__main__":
    main()
