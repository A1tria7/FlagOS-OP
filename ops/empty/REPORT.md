# 算子总体报告: empty

| 项 | 值 |
|---|---|
| 算子名称 | `empty`（分配/元数据类） |
| 语义 | [reference.py](reference.py)：分配 size 形状的未初始化张量，只保证元数据 |
| 硬件平台 | **1 个**：Cambricon MLU590（profile `cambricon`，torch_mlu；register 契约保留，但该算子不可 A1 注册） |
| 目标设备 / 路线 | `cambricon`（MLU590-M9）/ **自用/实验** |
| 日期 / 状态 | 2026-09-18 定稿 / 2026-10-12 注册契约同步 |

## 实现矩阵（三级）

| 开发级别 | 文件 | 状态 |
|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ✅ meta 推 stride + empty_strided |
| Triton 级 | — | ⬜ 不适用（无数值/设备码） |
| 硬件级 | — | ⬜ 不适用 |

## 验证矩阵（三层）

| 层级 | 入口 | 结果 |
|---|---|---|
| 一键三层 | `python3 example.py cambricon` | ✅ |
| kernel 层 | `test/kernel_level.py` | ✅ 15 项元数据检查 |
| 框架层 | `test/op_level.py` | ✅ — |
| 应用层 | `test/framework_level.py` | ✅ |
| 黄金 | `script/` | ✅ 48/48（元数据） |
| 性能 | 采样 | ✅ 0.0107ms @8192² fp32 |

## 2026-10-12 注册契约同步

- `register_a1` 已对齐 `counter` / `platform` 签名。
- 该算子保持 `NotImplementedError`，不强行做 A1 注册。
- 复验见 PR #25：MLU 13/13 三层全绿、unit tests 与真实 vLLM A1 注入通过。

## 结论

三层全绿；因是分配算子，黄金只判定元数据（无数值），哨兵 n/a。

## 交付物

[reports/development.md](reports/development.md) · [reports/test-report.md](reports/test-report.md) ·
[reports/accuracy.md](reports/accuracy.md) · [reports/performance.md](reports/performance.md)
