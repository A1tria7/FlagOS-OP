# 算子总体报告: _local_scalar_dense

| 项 | 值 |
|---|---|
| 算子名称 | `_local_scalar_dense`（host 标量类） |
| 语义 | [reference.py](reference.py)：item 的底层原语：单元素稠密张量 → host 标量 |
| 硬件平台 | **1 个**：Cambricon MLU590（profile `cambricon`，torch_mlu；A1 canonical `AutogradPrivateUse1`，兼容 `PrivateUse1`） |
| 目标设备 / 路线 | `cambricon`（MLU590-M9）/ **A1** |
| 日期 / 状态 | 2026-09-18 定稿 / 2026-10-12 注册契约同步 |

## 实现矩阵（三级）

| 开发级别 | 文件 | 状态 |
|---|---|---|
| torch 级 | [kernel/torch_level.py](kernel/torch_level.py) | ✅ |
| Triton 级 | — | ⬜ 不适用（无设备码） |
| 硬件级 | — | ⬜ 不适用 |

## 验证矩阵（三层）

| 层级 | 入口 | 结果 |
|---|---|---|
| 一键三层 | `python3 example.py cambricon` | ✅ |
| kernel 层 | `test/kernel_level.py` | ✅ 12 项 |
| 框架层 | `test/op_level.py` | ✅ aten::_local_scalar_dense@PrivateUse1（拦截 1 次） |
| 应用层 | `test/framework_level.py` | ✅ |
| 黄金 | `script/` | ✅ 36/36 |
| 性能 | 采样 | ✅ 0.0965ms |

## 2026-10-12 注册契约同步

- `register_a1` 支持 `counter["n"]` 拦截计数与显式 `platform`。
- canonical key 为 `AutogradPrivateUse1`，`PrivateUse1` 作为旧 profile /
  单算子脚本态兼容 key。
- `_local_scalar_dense` 的实现导入为包态优先、脚本态 fallback，包入口使用
  `from ops._local_scalar_dense import register_a1`。
- 复验见 PR #25：MLU 13/13 三层全绿、unit tests 与真实 vLLM A1 注入通过。

## 结论

三层全绿；标量值相等 + Python 类型 + 多元素报错。无 Triton/硬件级（该类算子固有）。

## 交付物

[reports/development.md](reports/development.md) · [reports/test-report.md](reports/test-report.md) ·
[reports/accuracy.md](reports/accuracy.md) · [reports/performance.md](reports/performance.md)
