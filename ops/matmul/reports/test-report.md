# matmul 测试报告

| 项 | 值 |
|---|---|
| 日期 | 2026-09-30 |
| 环境 | 个人锁定容器；Python 3.10.18、Torch 2.9.0+cu129、Triton 3.0.0+03c4c9be |
| 设备 | P800 物理 XPU 5：`CUDA_VISIBLE_DEVICES=1,5`，逻辑 `cuda:1` |
| 结论 | 替代卡算子三级全绿；正式目标卡、基线门禁和真实 vLLM 未完成 |

## 1. 测试范围与结果

| 层级 | 入口 | 本轮结果 |
|---|---|---|
| CPU 黄金生成 | `script/gen_golden.py` | 144 组前向及双输入梯度参考 |
| CPU `torch_level` 组合诊断 | `check_accuracy.py --impl torch --device cpu` | 对黄金 144/144；不计为自研实现 |
| kernel 黄金 | `check_accuracy.py --impl triton` | 144/144，前向与双输入梯度 |
| A1 黄金 | `check_accuracy.py --impl a1-triton` | 144/144，144 次命中 |
| kernel 直测 | `test/kernel_level.py` | 144 组、确定性/输入敏感哨兵、零 stride 反向通过 |
| A1 op | `test/op_level.py` | `AutogradCUDA` 15 次命中；direct ATen、CPU 隔离、单侧梯度通过 |
| FlagGems 共存 | `test/op_level.py --with-flaggems` | 排除 `copy_` 后 15 次命中，通过 |
| 应用层 | `test/framework_level.py` | FlagGems GELU + 两次 matmul；输出及三个梯度均通过，2 次命中 |
| 一键三级 | `example.py p800-kunlunxin` | kernel/op/framework 独立进程全部通过 |
| 跨进程注入 | `test/injection_probe.py` + `sitecustomize` | marker 后记录 1 次 A1 命中 |
| P800 特例收敛后回归 | XPU 5，运行时识别昆仑芯 | 专项 18/18；kernel 144 组、op 15 次、framework 2 次；A1 黄金 144/144、144 次命中 |

kernel 最坏前向相对误差 `7.463e-3`，最坏反向 `7.752e-3`；op 层最坏设备原生相对误差 `1.087e-2`；应用层输出/梯度最大误差 `3.509e-3`。

## 2. 应用与环境限制

真实 tiny vLLM 基线先因个人结果目录和镜像内 `flash_attn` 仅 root 可读而失败；仅修正个人目录所有权和个人容器层读取权限后，进入 EngineCore，但 FlagGems `lt_func_scalar_kernel_rank_1` 报 XPU kernel 错误 714。对个人测试入口同时排除 FlagGems `copy_` 和 `lt` 后，以 `--tensor-parallel-size 1` 在物理 XPU 5 重试，插件日志确认黑名单生效；EngineCore 初始化阶段没有后续输出，180 秒后超时。两次失败都发生在**未注入 matmul 的基线**，故没有模型输出对比和真实模型命中证据。日志位于个人仓库 `results/matmul/real_model/20260930T082514Z-19326/baseline.log` 与 `results/matmul/real_model/20260930T084319Z-20874/baseline.log`。入口默认读取 P800 profile 的 TP=2；单卡结果只作探索，正式配置须待目标卡可用再试。

目标物理 XPU 2 在测试期间被其他任务占用。本轮没有把物理 XPU 5 的结果记为目标卡结果。个人锁定镜像没有 `pytest`，所以本轮未运行仓库级 `tests/unit`；matmul 的脚本三级测试均实际运行。

另以 `CUDA_VISIBLE_DEVICES=5,1` 尝试将逻辑 `cuda:1` 映射至空闲物理 XPU 1。
`example.py` 在 kernel 层启动后没有继续输出，XPU 1 持续 100% 利用率、
该测试进程约占 166 MiB；终止个人测试后无残留进程。该卡的 kernel 层
**未通过也未完成**，op/framework 未运行，不能据此推断自研 matmul 在
XPU 1 上正常使用。当前可用性结论仍仅来自物理 XPU 5 的完整三级测试。
`example.py` 此后增加每层 180 秒默认超时和进程组清理，避免换卡测试
无限占卡；该保护本身尚未在 XPU 1 重跑。修订后的入口在物理 XPU 5
再次通过 kernel 144 组、op 15 次命中、framework 2 次命中；另设
`--level-timeout 1` 人为触发超时，确认个人容器无残留子进程。

随后在物理 XPU 1 的最小 PyTorch 探针中，`torch.ones(1, device="cuda:1")`
创建成功，但 `.item()` 读回超过 15 秒未返回，超时后仅个人进程被清理。
故障发生在 matmul 之前。三级入口新增限时读回探针，以便在完整测试前
识别这一类设备运行问题。物理 XPU 5 的预检和三级测试再次全部通过；
物理 XPU 1 的预检在 `--probe-timeout 15` 下报超时，未运行 kernel，
个人容器和该卡均无残留测试进程。再以 `CUDA_VISIBLE_DEVICES=1,5`
把物理 XPU 1 映射为逻辑 `cuda:0`，最小 `torch.ones(1).item()` 也在
15 秒内无返回；目前不能用调整逻辑设备序号解释该问题。

## 3. 性能状态

`scripts/perf_run.py --pattern ops.matmul` 产生 4 条结构化记录、0 跳过。当前基线没有 matmul：`FAIL 0 / WARN 0 / NEW 4`，不能判为门禁通过。A1 入口与直调自研 kernel 的二维 BF16 中位延迟差为 `0.048949 ms`，只作为替代卡探索性结果。详见 [performance.md](performance.md)。

## 4. 复现

从个人容器项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=1,5 python ops/matmul/script/check_accuracy.py --impl triton --device p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,5 python ops/matmul/script/check_accuracy.py --impl a1-triton --device p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,5 python ops/matmul/example.py p800-kunlunxin
```

正式目标卡需把映射改为 `1,2`，且先确认物理 XPU 2 空闲。
