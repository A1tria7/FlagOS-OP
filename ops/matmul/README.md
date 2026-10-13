# matmul-op: `aten::matmul` A1 实现

按 `templates/operator/` 开发。需求是 FP32/FP16/BF16 的向量、矩阵、批量及广播 matmul，前向和左右输入一阶反向分别验证。当前在昆仑芯 P800 锁定镜像的个人容器完成替代物理 XPU 5 功能测试；目标物理 XPU 2、性能基线和真实模型仍待收口。

## 一页看全

| 项 | 值 |
|---|---|
| 接口 | `torch.matmul` / `aten::matmul` |
| 接入 | A1 `AutogradCUDA`；实际命中计数 |
| 生产计算 | 自研 Triton `tl.dot`，32×32 输出 tile；昆仑芯运行时的 BF16 `K=1` 自动走自研外积/归约，目前仅 P800 实测 |
| 反向 | 两次自研 matmul + 广播维归约 |
| 正确性 | CPU 参考生成 144 组；CPU `torch_level` 诊断、直调 Triton、A1 均对黄金 144/144；三级测试全绿（物理 XPU 5） |
| 性能 | BF16 两组三维用例 dot `0.102/0.104 ms`，原生 `0.094/0.095 ms`；二维 A1 入口增量约 `0.049 ms`；门禁 4 条 `NEW` |

## 实现要点

`kernel/_shape.py` 统一处理向量升降维、批次广播和梯度归约。`kernel/triton_level.py` 默认使用 dot；`FLAGOS_MATMUL_TRITON_ALGORITHM=outer` 可切到旧外积路径。入口通过 FlagGems 的运行时设备检测识别昆仑芯，仅在该环境对 BF16 `K=1` 使用自研外积/归约，并将 dot 输入中矩阵维零 stride（例如 `sum()` 反向的全 1 梯度）实化；调用方仍只调用同一个 `matmul_triton`。这些规避来自 P800 锁定环境的异常观测，BF16 dot 的根因尚未确定。高阶梯度不支持，其他厂商的 matmul 仍须分别验证。

`register.py` 对 P800 把 profile 的 `CUDA` key 提升为 `AutogradCUDA`；只注册 `CUDA` 会在 `aten::matmul_backward` 走错路径。`kernel/torch_level.py` 是语义诊断，不能算作自研 Triton 通过。应用层使用 FlagGems GELU 与两次 A1 matmul，独立进程运行三个测试层级。

## 个人容器复现

从 `/workspace/tyanfeng/FlagOS-OP` 运行，先确认 `xpu-smi` 中目标卡空闲。当前替代卡测试使用 `CUDA_VISIBLE_DEVICES=1,5`，逻辑 `cuda:1` 对应物理 XPU 5：

```bash
python ops/matmul/script/gen_golden.py
python ops/matmul/script/check_accuracy.py --impl torch --device cpu
CUDA_VISIBLE_DEVICES=1,5 python ops/matmul/script/check_accuracy.py --impl triton --device p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,5 python ops/matmul/script/check_accuracy.py --impl a1-triton --device p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,5 python ops/matmul/example.py p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,5 python scripts/perf_run.py --device p800-kunlunxin --pattern ops.matmul
```

正式 profile 需在物理 XPU 2 空闲后把每条命令的映射改为 `CUDA_VISIBLE_DEVICES=1,2` 并重跑。物理 XPU 1 的 PyTorch 标量读回会卡住，问题发生在 matmul 之前；`example.py` 先做 30 秒设备读回预检，并对每层设置 180 秒超时。真实 vLLM 入口 `test/real_model.py` 默认读取 profile 的 TP=2；本轮单卡 tiny 模型基线先在 FlagGems `lt` 点算子失败，排除该点算子后又在 EngineCore 初始化阶段超时，仍未进入 matmul 注入。`--tensor-parallel-size 1` 只用于单卡探索，不计作正式 profile 验收。

查看 [需求](requirement.md)、[总体报告](REPORT.md)、[开发报告](reports/development.md)、[测试报告](reports/test-report.md)和[性能报告](reports/performance.md)。
