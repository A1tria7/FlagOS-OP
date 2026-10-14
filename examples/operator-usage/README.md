# 连续算子调用与混合映射样例

这个样例展示同一条模型链如何让不同算子进入不同实现：

```text
embedding
  -> Q/K/V projection
  -> aten::_scaled_dot_product_attention_math
  -> GELU
  -> output projection
```

三种模式比较：

| 模式 | embedding | sdpa_math | GELU |
|---|---|---|---|
| native | native ATen | native private math | native ATen |
| direct | FlagOS-OP P800 direct kernel | FlagOS-OP P800 fast path | FlagGems direct API |
| a1-mixed | FlagOS-OP A1 | FlagOS-OP A1 | FlagGems dispatcher |

运行前提：

1. 使用仓库提供的 P800 / Kunlunxin 运行时；
2. `torch`、`flag_gems` 和 FlagOS-OP 的 `ops` 包可以正常 import；
3. 样例默认使用 `cuda:1`。环境只有一个可见设备时，把 `--device`
   改成该环境可用的设备名。

运行：

```bash
cd FlagOS-OP
PYTHONPATH="$PWD" python3 examples/operator-usage/example.py p800-kunlunxin
```

输出中每个模式都会列出实际路由。`a1-mixed` 模式还会输出 counter，
并要求两个 counter 都等于 1：

```text
[a1-mixed]
  embedding  -> FlagOS-OP A1
  sdpa_math  -> FlagOS-OP A1
  gelu       -> FlagGems dispatcher
  counters   -> {'embedding': 1, 'sdpa_math': 1}
```

最后两行输出 direct 与 `a1-mixed` 相对 native 的最大绝对误差。误差数值
会随运行时和硬件版本变化；本样例的强制检查项是输出形状、有限值和 A1
counter。

要点：

1. `a1-mixed` 使用 `flag_gems.only_enable(["gelu"])` 控制 FlagGems 范围。
2. FlagOS-OP 的 embedding 和 `sdpa_math` 分别调用各自 `register_a1()`。
3. A1 注册对象全程保存引用。
4. `counter` 分别确认 embedding 和 attention 都命中一次。
5. 每种模式都在独立子进程中运行，避免 `torch.library` 注册互相污染。
6. native baseline 在独立子进程采集，符合本仓库的 A1 测试边界。

文档化场景和 CPU / Cambricon 片段见
[安装与 FlagGems 配合使用](../../docs/installation.md)。
