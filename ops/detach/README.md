# detach（别名/元数据（autograd）算子）

`aten::detach` 的 MLU（Cambricon）实现，路线 **A1**，
**torch 级**（无 Triton / 硬件级）。

## 定位

| 项 | 值 |
|---|---|
| 宿主 | `torch.detach` |
| 语义 | 返回与 self 共享底层存储、脱离 autograd 的新张量 (requires_grad=False) |
| 判定 | 共享存储 + requires_grad=False + shape/dtype/stride |

## 运行方式

```bash
cd ops/detach
python3 example.py cambricon
python3 test/kernel_level.py cambricon
python3 test/op_level.py cambricon
python3 test/framework_level.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl torch --device mlu
python3 script/bench_perf.py --device mlu
```

## 关键点

1. 注册：aten::detach@AutogradPrivateUse1（兼容 PrivateUse1；拦截 1 次）。
2. 判定：共享存储 + requires_grad=False + shape/dtype/stride。
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
- 包入口：`from ops.detach import register_a1`

详见 [REPORT.md](REPORT.md)。
