# item（host 标量算子）

`aten::item` 的 MLU（Cambricon）实现，路线 **A1**，
**torch 级**（无 Triton / 硬件级）。

## 定位

| 项 | 值 |
|---|---|
| 宿主 | `torch.item` |
| 语义 | 单元素张量取为 Python 标量 (device→host 同步) |
| 判定 | 标量值相等 + Python 类型 + 多元素报错 |

## 运行方式

```bash
cd ops/item
python3 example.py cambricon
python3 test/kernel_level.py cambricon
python3 test/op_level.py cambricon
python3 test/framework_level.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl torch --device mlu
python3 script/bench_perf.py --device mlu
```

## 关键点

1. 注册：aten::item@AutogradPrivateUse1（兼容 PrivateUse1；拦截 1 次）。
2. 判定：标量值相等 + Python 类型 + 多元素报错。
3. 无 Triton/硬件级（该类算子没有可编译的元素级设备码）。

## 注册契约（2026-10-12）

`register_a1` 统一支持 `counter` 与显式 `platform`：

```python
register_a1(
    dispatch_key="AutogradPrivateUse1",
    counter={"n": 0},
    platform="cambricon",
)
```

- canonical key：`AutogradPrivateUse1`
- 兼容旧 profile / 脚本态 key：`PrivateUse1`
- 实现导入：包态优先，单算子脚本态 fallback
- 包入口：`from ops.item import register_a1`

详见 [REPORT.md](REPORT.md)。
