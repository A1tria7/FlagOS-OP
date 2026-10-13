# 已开发算子（ops/）

这个目录存放**按样板开发完成并交付的算子**——每个子目录一个算子，
结构与 [templates/operator](../templates/operator/README.md) 完全一致
（kernel 三级 · test 三层 · goldendata · script · REPORT + 交付报告四件套）。

## 新算子放进来

```bash
cp -r templates/operator ops/<你的算子名>/
cd ops/<你的算子名>/
grep -rl my_op . | xargs sed -i 's/my_op/<你的算子名>/g'
```

开发流程照[全链路指南](../docs/fullstack-guide.md)七步走；完成后:

1. `python3 example.py <profile>` 一键三层全绿
2. 补齐交付报告四件套（REPORT + development + test-report + 分册，
   范本见 [softmax-fullstack](../examples/softmax-fullstack/reports/development.md)）
3. 在下方算子总览中登记一行
4. 交付路径选择见[集成指南](../docs/integration.md)（FlagGems 贡献 /
   aten 注册 / vllm_fl 插件 / 厂商身份）

## 算子总览

当前共有 **3 类硬件平台**、**16 个算子目录**：

| 硬件平台 | 覆盖算子 |
|---|---|
| `ascend910` | `sdpa` · `sdpa_math` · `embedding` |
| `p800-kunlunxin` | `sdpa` · `sdpa_math` · `embedding` |
| `cambricon` / `mlu590` | `sdpa` · `embedding` + 下文 13 个基础 ATen 算子 |

### 1. 核心多平台算子

这一组是当前主要交付的计算算子：注意力、私有 math 后端和 embedding。
它们都有独立 `PLATFORM.md` 与 `MERGE.md`，适合作为新平台接入的参考。

| 算子 | 作用 | 平台与实现 | 关键结论 |
|---|---|---|---|
| [sdpa](sdpa/) | 标准 SDPA 前向 | ascend910 / p800-kunlunxin / mlu590；Triton 与厂商委托组合 | 三平台黄金 **397/397**；[平台绑定](sdpa/PLATFORM.md) · [合并流程](sdpa/MERGE.md) |
| [sdpa_math](sdpa_math/) | `aten::_scaled_dot_product_attention_math`，额外返回完整概率图 | ascend910 / p800-kunlunxin；Triton + vendor O/LSE exact-P | 四实现黄金 **175/175**；P800 direct **1.46–2.21x** native |
| [embedding](embedding/) | 稠密 embedding 前向 / dense backward | ascend910 / p800-kunlunxin / cambricon；厂商委托 + Triton 探针 | 三平台稠密查表 / 反向；MLU 174/174 逐位 |

### 2. MLU 基础 ATen 算子

下面 13 个目录来自 PR #5 的 MLU 基础算子族。它们不是都要“比原生更快”的
旗舰计算 kernel，主要价值是：

1. 验证 MLU Torch / Triton / A1 注册链路；
2. 覆盖布局、复制、随机、autograd、分配、标量和 dtype 元数据等基础语义；
3. 支撑真实 vLLM 的多算子 A1 注入回归。

因此表格按“用途”分组，而不是把 13 行混在核心计算算子后面。

#### 布局、复制与类型转换

| 算子 | 作用 | 交付路线 | 关键结论 |
|---|---|---|---|
| [type_as](type_as/) | 把 tensor cast 到 `other.dtype`，留在原设备 | A1 · Triton | 黄金 **111/111**，位级一致 |
| [clone](clone/) | 复制 tensor，保证新存储 | A1 · Triton | 黄金 **48/48**，存储独立 |
| [contiguous](contiguous/) | 返回连续布局，连续输入保持别名 | A1 · Triton | 黄金 **114/114**；转置路径 swiz 优化约 **80x** |
| [copy_](copy_/) | 原地复制，支持布局与广播 | A1 · Triton | 黄金 **78/78** |

#### 随机与 autograd 语义

| 算子 | 作用 | 交付路线 | 关键结论 |
|---|---|---|---|
| [dropout](dropout/) | 随机 dropout，验证随机分支与统计判定 | A1 · Triton | 黄金 **189/189** |
| [detach](detach/) | 非原地 detach，返回共享存储的新 tensor | A1 · torch | 黄金 **36/36** |
| [detach_](detach_/) | 原地 detach；属于 autograd 层语义 | 自用 / 实验 | 不可 A1 注册，保持 `NotImplementedError` |

#### 分配与元数据

| 算子 | 作用 | 交付路线 | 关键结论 |
|---|---|---|---|
| [empty_like](empty_like/) | 按已有 tensor 元数据分配新 tensor | A1 · torch | 元数据黄金 **168/168** |
| [empty](empty/) | 按 size 分配未初始化 tensor | 自用 / 实验 | 不注册，避免与 `empty_strided` 递归；元数据 **48/48** |
| [empty_strided](empty_strided/) | 按 size / stride 分配 tensor | A1 · torch | 元数据 **30/30** |
| [result_type](result_type/) | 计算两个输入的类型提升结果 | 自用 / 实验 | 纯 dtype 元数据函数，不可 A1 注册 |

#### Host 标量

| 算子 | 作用 | 交付路线 | 关键结论 |
|---|---|---|---|
| [item](item/) | 单元素 tensor 转 Python 标量 | A1 · torch | 黄金 **36/36** |
| [_local_scalar_dense](_local_scalar_dense/) | `item` 底层稠密标量原语 | A1 · torch | 黄金 **36/36** |

> 这 13 个基础算子均在 MLU 上完成三层验证；真实 vLLM 多算子注入见
> `tests/framework_level/test_a1_aten_ops.py`。`detach` 虽可单独注册，
> 但会破坏真实 vLLM 的 `nn.Parameter` 构造，因此不纳入多算子注入集。

### 13 个 MLU 算子的注册契约（2026-10-12）

PR #25 已把 13 个算子的 `register_a1` 统一到当前库级签名：

```python
register_a1(
    dispatch_key="AutogradPrivateUse1",
    counter: dict | None = None,
    platform=None,
)
```

约定：

- canonical A1 key 为 **`AutogradPrivateUse1`**；
- `PrivateUse1` 继续作为旧 profile / 脚本态兼容 key；
- generic `Autograd` 不作为单平台 MLU 默认映射，避免多后端同进程时扩大
  autograd 覆盖范围；
- 10 个可注册算子支持 `counter["n"]` 拦截计数与显式 `platform`；
- `empty` / `detach_` / `result_type` 接受同一签名但保持
  `NotImplementedError`，不强行注册；
- 10 个可注册算子的实现导入均为**包态优先、单算子脚本态 fallback**；
- `clone/test/op_level.py` 包含包态实际执行代表用例，覆盖
  `ops.clone.register → ops.clone.kernel` 路径。

PR #25 复验：MLU 13/13 三层全绿、`pytest tests/unit` 通过、真实 vLLM
A1 注入通过。

> 结构同构的教学样例 [softmax-fullstack](../examples/softmax-fullstack/)
> 保留在 examples/（它的定位是"教"，ops/ 的定位是"用"）；
> 后续正式开发的算子一律进 ops/ 并在此登记。

## 约定

- **目录名 = 算子名**（小写下划线，如 `gelu_and_mul`）
- **交付自查清单**: 见 templates/operator/README「交付自查清单」节
  （2026-10-07 审计后新增——13 算子曾缺包化入口已补齐，后续算子
  合并前照单自查，勿再欠账）
- **包化导入（多算子同进程必用）**: 目录含 `__init__.py`，
  统一使用 `from ops.<算子名> import register_a1`，顶层名带包前缀、
  互不遮蔽（根因与验证见 sdpa/reports/e2e_mini_llm.md）；
  单算子脚本模式（sys.path 注入目录）继续兼容。CI 的 `ops-packaging`
  job 负责 CPU 包入口扫描；`ops/test_packaging.py` 是 NPU 真机 E2E，
  二者口径不同
- 未实现的级别**置空 + 说明**（`kernel/hardware_level/README.md`），
  不留空文件
- 命名/级别/路线术语遵循 [体系结构](../docs/architecture.md)；
  Triton kernel 必须内置 [#11](../docs/known-issues.md) device 上下文与
  [#15](../docs/known-issues.md) 尾块防护两条硬约束
- 合入本目录前: 三层测试全绿 + 黄金回归 + perf 门禁通过
  （见[集成指南交付 checklist](../docs/integration.md#checklist)）
