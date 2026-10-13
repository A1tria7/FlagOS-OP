# 精度报告: matmul

## 1. 用例与判卷

`goldendata/inputs_spec.yaml` 共 16 类输入 × FP32/FP16/BF16 × 三种子 = 144 组。覆盖向量、矩阵、批量广播、非连续布局、尾块、零 K/空输出/空批次、零输入、大数、2×2 和 `K=1`；每组保存前向及左右输入梯度。CPU reference 用 FP32 内部计算后转回输入 dtype。非零参考按 GEMM 相对误差 `5e-2` 判定，全零参考使用 dtype 绝对容差。

## 2. 锁定环境结果

个人容器 `tyanfeng-flagos-p800-lock`，PyTorch `2.9.0+cu129`，Triton `3.0.0+03c4c9be`，`CUDA_VISIBLE_DEVICES=1,5`，逻辑 `cuda:1` 为物理 XPU 5。

| 路径 | 前向 + dA + dB | A1 命中 | 结论 |
|---|---:|---:|---|
| CPU `torch_level` 组合诊断 | 144/144 | 不适用 | 通过；不计为自研实现 |
| 自研 Triton dot/受限回退 | 144/144 | 不适用 | 通过，最坏 `7.75e-3` |
| A1 自研 Triton | 144/144 | 144 | 通过，最坏 `7.75e-3` |
| kernel 直测 | 144 组 + 哨兵 + 零 stride 梯度 | 不适用 | 通过 |
| A1 op 对照设备原生 | 15 次 | 15 | 通过，最坏相对误差 `1.087e-2` |
| A1 与 FlagGems 共存 | 15 次 | 15 | 排除其 `copy_` 后通过 |
| 目标物理 XPU 2 | 未复测 | 未复测 | 被其他任务占用 |

CPU 黄金由 `reference.py` 调用 CPU PyTorch 生成，用作设备实现的判卷依据；
表中 CPU `torch_level` 一行另外检查形状整理和 `bmm` 组合逻辑，
它不是自研 Triton 精度的替代证明。

首次完整 dot 回归为 133/135：BF16 `matrix_vector` 的两组左梯度失败。查明其反向会生成 `K=1` 矩阵乘；该形态改用自研外积/归约后，追加直接的 `rank_one` 用例，达到 144/144。随后 op 层发现 `sum()` 的零 stride 梯度使 dot 双输入梯度错误；矩阵维零 stride 进入 dot 前实化。两项均纳入 kernel 直测和黄金回归。

后续对同形态直接 dot 的重复探索曾出现正常结果，故不能断定 BF16 异常的根因是编译器；保留 P800 自动回退。将两项规避限定至运行时识别出的昆仑芯后，在物理 XPU 5 重测 `matrix_vector`/`rank_one` 18/18、kernel 144 组、op 15 次、framework 2 次及 A1 黄金 144/144。其他平台上的默认 dot 分支仍待实测。

## 3. 对照边界

锁定版 FlagGems `bmm` 在自动调参阶段报错，无法形成有效精度/性能对照。寒武纪、昇腾和壁仞尚未验证。

复现：`script/gen_golden.py`、`script/check_accuracy.py --impl triton --device p800-kunlunxin`、`--impl a1-triton`；执行前设置本轮设备映射。环境和命令见 [开发报告](development.md)。
