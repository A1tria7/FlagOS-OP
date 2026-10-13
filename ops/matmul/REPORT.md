# 算子总体报告: matmul

| 项 | 值 |
|---|---|
| 算子 | `aten::matmul` |
| 范围 | 向量、矩阵、批量和广播 matmul；FP32/FP16/BF16 前向与双输入一阶反向 |
| 当前平台 | Kunlunxin P800，锁定 Triton 3.0 个人容器；其他平台待测 |
| 路线 | A1，`AutogradCUDA` |
| 生产实现 | 自研 Triton `tl.dot`；识别昆仑芯运行时后，对 BF16 `K=1` 使用自研外积/归约，当前仅 P800 实测 |
| 日期 / 状态 | 2026-09-30 / 替代物理 XPU 5 功能全绿，正式交付未完成 |

## 实现矩阵

| 层级 | 文件 | 状态 |
|---|---|---|
| CPU 判卷 | [reference.py](reference.py) | FP32 内部计算，返回输入 dtype |
| Triton 前向/反向 | [kernel/triton_level.py](kernel/triton_level.py) | 默认 dot；昆仑芯的零 stride 实化和 BF16 `K=1` 自研回退由运行时设备检测自动启用 |
| torch 组合 | [kernel/torch_level.py](kernel/torch_level.py) | A1 语义诊断，不计为自研 kernel |
| A1 注册 | [register.py](register.py) | `aten::matmul` + `AutogradCUDA`，保留 Library 引用 |
| 硬件级 | [kernel/hardware_level/README.md](kernel/hardware_level/README.md) | 本期无厂商语言实现 |

## 验证矩阵

| 层级 | 锁定环境、物理 XPU 5 结果 |
|---|---|
| CPU 参考 / 组合诊断 | 生成 144 组前向和双输入梯度黄金；`torch_level` 对黄金 144/144 |
| Triton 黄金 | 144/144，最坏相对误差 `7.75e-3` |
| A1 黄金 | 144/144，144 次前向拦截 |
| kernel | 144 组、确定性/输入敏感哨兵和零 stride 梯度通过 |
| op | 15 次拦截；与 FlagGems 共存时另有 15 次通过 |
| 应用层 | FlagGems GELU + 两次 matmul，输出/梯度最大原生相对误差 `3.51e-3` |
| 跨进程注入探针 | marker 后 1 次 A1 命中 |
| P800 特例收敛后回归 | XPU 5 上专项 18/18、kernel 144 组、op 15 次、framework 2 次、A1 黄金 144/144 |
| 真实 vLLM | 基线先在 FlagGems 点算子报错；排除后初始化超时，未运行注入 |
| 物理 XPU 1 探索 | 最小 PyTorch 标量读回超时，未进入 matmul |

## 性能与结论

BF16 两组三维结构化记录：自研 dot `0.102/0.104 ms`，原生 `0.094/0.095 ms`；独立进程、20 次预热、100 次采样。二维 BF16 A1 入口比直调自研中位数多 `0.048949 ms`，包含 wrapper 和 autograd 包装。锁定版 FlagGems `bmm` 在自动调参时报 `ZeroDivisionError`。仓库门禁显示 `FAIL 0 / WARN 0 / NEW 4`，因为尚无 matmul 入库基线；不能写成门禁已通过。测试在替代物理 XPU 5，目标物理 XPU 2 正被其他任务使用，需空闲后复测。

P800 的 BF16 `K=1` dot 曾出现不稳定的大误差；现有数据尚不足以将根因归于 Triton 编译器或内核索引。当前回退保留，其他平台不会自动继承该特例。

需求、开发、测试、精度和性能细节分别见 [requirement.md](requirement.md)、[development.md](reports/development.md)、[test-report.md](reports/test-report.md)、[accuracy.md](reports/accuracy.md)、[performance.md](reports/performance.md)。
