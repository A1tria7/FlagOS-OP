# copy_

`aten::copy_` 的 MLU（Cambricon）实现，路线 **A1**，开发级别 **Triton 级**。

## 定位

| 项 | 值 |
|---|---|
| 宿主 | 标准 aten 算子（`torch.Tensor.copy_`） |
| 路线 | A1：`register_a1(dispatch_key="AutogradPrivateUse1", counter=None, platform=None)`；兼容 `PrivateUse1` |
| 设备 | `cambricon`（MLU590-M9） |
| 语义 | 原地写 dst 并返回 dst；src 支持广播 + cast 到 dst.dtype |

## 运行方式

```bash
cd ops/copy_
python3 example.py cambricon
python3 test/kernel_level.py cambricon
python3 test/op_level.py cambricon
python3 test/framework_level.py cambricon
python3 script/gen_golden.py --device cpu
python3 script/check_accuracy.py --impl triton --device mlu
python3 script/bench_perf.py --device mlu --impl triton
```

## 关键点 / 坑

1. **#11**: Triton 启动包 `torch_device_fn.device`。
2. **原地语义**: 必须返回 dst 本身；测试每次新建 dst。
3. **全局注册侵入性强**: copy_ 是高频内部算子，框架层用子进程隔离；
   op 层需在注册前取原生参考。
4. boundary 溢出到 ±inf 属合法 cast，判定用 isclose。
5. 自拷贝（别名）走原生回退；2D 转置型走分块 swiz。

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
- 包入口：`from ops.copy_ import register_a1`

详见 [REPORT.md](REPORT.md)。
