# 算子开发报告: matmul

## 1. 环境配置

| 项 | 值 |
|---|---|
| 设备 | Kunlunxin P800，物理 XPU 5；`CUDA_VISIBLE_DEVICES=1,5`，逻辑 `cuda:1` |
| 容器 | 个人容器 `tyanfeng-flagos-p800-lock` |
| Python / Torch | 3.10.18 / 2.9.0+cu129 |
| Triton / FlagGems | 3.0.0+03c4c9be / 4.2.1rc0 |
| 配置 | `configs/devices/p800-kunlunxin.yaml`、`configs/env/p800-kunlunxin.lock.yaml` |

环境锁检查为 `OK`。项目 profile 的正式映射为 `CUDA_VISIBLE_DEVICES=1,2`，
逻辑 `cuda:1` 指向物理 XPU 2；该卡被其他任务占用，本轮只在空闲 XPU 5
验证，不能将结果视为正式目标卡验收。个人容器没有 `pytest`。

## 2. 算子定义

### 2.1 语义与接口

```python
torch.matmul(a, b) -> torch.Tensor
```

支持 FP32、FP16、BF16 的向量点积、矩阵乘、批量矩阵乘、批次广播、非连续输入、
零 K 和空输出；前向及左右输入一阶梯度分别验证。广播后的梯度要归约回原输入
形状。高阶梯度不在当前范围。精确输入规格见
[`goldendata/inputs_spec.yaml`](../goldendata/inputs_spec.yaml)。

### 2.2 参考与数值规格

[`reference.py`](../reference.py) 用 FP32 内部计算，结果转换为输入 dtype。
16 类形状与特殊值、三 dtype、每类三种子，共 144 组 CPU 黄金；每组同时
保存前向和双输入梯度。非零参考按 GEMM 相对误差 `5e-2`，零参考按 dtype
绝对容差判定。

## 3. 实现说明

### 3.1 kernel 与反向

[`kernel/_shape.py`](../kernel/_shape.py) 处理向量升降维、批次广播及梯度
归约。[`kernel/triton_level.py`](../kernel/triton_level.py) 默认用 32x32
输出 tile 的自研 `tl.dot` kernel；反向调用两次自研 matmul。可通过
`FLAGOS_MATMUL_TRITON_ALGORITHM=outer` 单独测试外积/归约实现。

锁定 P800 栈中，BF16 `K=1` 的 dot 曾对部分梯度给出错误值；该输入形态改走
自研外积/归约。`sum()` 反向可生成矩阵维零 stride 的输入，之前的 dot 路径
对此给出错误值；进入 dot 前只实化这种矩阵维。两项规避仅在 FlagGems 运行时
识别为昆仑芯时启用，调用入口保持不变。BF16 dot 的根因尚未确定，不能据此
认定为编译器故障。P800 XPU 5 收敛后的回归为专项 18/18、三级全通过，A1
黄金 144/144、144 次命中；其他厂商的默认 dot 路径尚未验证。
`kernel/torch_level.py` 仅用于独立检查语义，不能记为自研 kernel。
本期没有厂商语言实现。

### 3.2 A1 注册与分发

[`register.py`](../register.py) 注册 `aten::matmul` 的 `AutogradCUDA`
实现并保留 `Library` 引用。P800 的张量经 XMLIR 表现为 CUDA；仅注册
`CUDA` 时，前向可命中但反向走到不适用的 `aten::matmul_backward`。
op 层同时验证 `torch.matmul`、显式 ATen 调用、CPU 不受影响和实际命中次数。

## 4. 验证结果

| 层级 | 锁定环境、物理 XPU 5 |
|---|---|
| CPU 参考 / `torch_level` 诊断 | 生成 144 组黄金；组合诊断对黄金 144/144 |
| Triton 黄金 / A1 黄金 | 各 144/144；A1 144 次命中 |
| kernel | 144 组、确定性/输入敏感哨兵、零 stride 梯度通过 |
| op | 15 次 A1 命中；排除 FlagGems `copy_` 后共存测试另有 15 次 |
| 小型应用 | FlagGems GELU + 两次 A1 matmul，通过且命中 2 次 |
| 子进程注入 | marker 后 1 次 A1 命中 |
| 真实 vLLM | 未形成基线/注入输出对照及真实模型命中证据 |

kernel 最坏前向/反向相对误差分别为 `7.463e-3` / `7.752e-3`。
[`test-report.md`](test-report.md) 和 [`accuracy.md`](accuracy.md) 保留测试入口
及判卷结果。寒武纪、昇腾、壁仞尚未实测。

## 5. 性能

BF16 两组结构化性能记录的自研 dot 为 `0.101937/0.104379 ms`，原生
为 `0.093774/0.094818 ms`。采样采用独立进程、20 次预热、100 次采样及
逐次标量读回。FlagGems `bmm` 自动调参失败，没有有效对照。仓库门禁
为 `FAIL 0 / WARN 0 / NEW 4`，因为尚未建立 matmul 基线，不能算门禁通过。
小形状重复采样波动较大，详见 [`performance.md`](performance.md)。
二维 BF16 的 A1 入口与直调自研 kernel 中位数差为 `0.048949 ms`，包含
wrapper 和 autograd 包装；这不是纯 dispatcher 时间。

## 6. 已知问题与后续

| 问题 | 当前状态 |
|---|---|
| 正式物理 XPU 2 被占用 | 待空闲后按 profile 映射复测黄金、三级和性能 |
| matmul 无同环境性能基线 | 待目标卡重采、建基线和复跑门禁 |
| FlagGems `bmm` 自动调参异常 | 不把不可用对照写成通过 |
| BF16 `K=1` dot 曾出现不稳定的大误差 | 保留 P800 自动回退；需独立复现并定位索引、编译和执行因素 |
| tiny vLLM 基线在 FlagGems `lt` kernel 报 XPU 错误 714 | 排除该算子后单卡重试停在 EngineCore 初始化并超时；仍需基线和注入对照 |
| 其他三家国产硬件 | 各自环境、A1、精度、性能待验证 |
| 物理 XPU 1 的 PyTorch 标量读回超时 | 故障在 matmul 前；三级入口先做限时设备探针，当前只认定 XPU 5 功能通过 |

## 附录: 复现入口

在个人容器项目根目录，确认物理 XPU 5 空闲后执行：

```bash
python ops/matmul/script/gen_golden.py
python ops/matmul/script/check_accuracy.py --impl torch --device cpu
CUDA_VISIBLE_DEVICES=1,5 python ops/matmul/script/check_accuracy.py --impl triton --device p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,5 python ops/matmul/script/check_accuracy.py --impl a1-triton --device p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,5 python ops/matmul/example.py p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,5 python scripts/perf_run.py --device p800-kunlunxin --pattern ops.matmul
```

原始锁定环境性能记录位于
[`perf_runs/p800-kunlunxin-triton30-xpu5/`](perf_runs/p800-kunlunxin-triton30-xpu5/)。
