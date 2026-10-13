# 性能报告: matmul

## 1. 配置

| 项 | 值 |
|---|---|
| 日期 / 环境 | 2026-09-30；个人锁定容器，Triton 3.0.0+03c4c9be |
| 设备 | P800 物理 XPU 5；`CUDA_VISIBLE_DEVICES=1,5`，逻辑 `cuda:1` |
| dtype / 算法 | BF16；默认自研 Triton dot，BF16 `K=1` 为自研回退 |
| 口径 | 相同 CPU 种子输入；每实现独立进程；20 次预热、100 次采样、每次标量读回完成 |
| 入口 | `script/bench_perf.py`、`scripts/perf_run.py --pattern ops.matmul` |

## 2. 结构化记录

| shape (B,M,K,N) | 自研 dot ms | 原生 ms | 自研/原生延迟 | 自研 GFLOPS |
|---|---:|---:|---:|---:|
| (1,63,127,65) | 0.101937 | 0.093774 | 1.087x | 10.204 |
| (1,128,256,192) | 0.104379 | 0.094818 | 1.101x | 120.550 |

原始 JSON 在 [perf_runs/p800-kunlunxin-triton30-xpu5/](perf_runs/p800-kunlunxin-triton30-xpu5/)。独立重复基准使用同样的 CPU 种子和输入：小形状 dot/native 为 `0.124893/0.087466 ms`，大形状为 `0.102190/0.093717 ms`。小形状重复值波动较大，当前只能说明 dot 与原生在同一量级，不能据此宣称稳定性能持平。重复数据见 `perf_dot_p800_xpu5_*.json`。

将 BF16 `K=1` 与零 stride 规避限定到运行时识别出的昆仑芯后，在同一替代物理 XPU 5 重跑四条结构化用例，全部成功：自研小/大形状分别为 `0.103124/0.097531 ms`，原生为 `0.118553/0.091960 ms`。记录写在个人服务器工作树的 `results/perf/runs/p800-kunlunxin/`，未更新正式基线。此次自研耗时与此前同量级；原生小形状的波动说明单次重采不足以得出性能优势结论。

## 3. FlagGems 与门禁

另用 [`script/bench_dispatch.py`](../script/bench_dispatch.py) 在同一替代卡上
测二维 `(63,127) @ (127,65)`、BF16、输入需要梯度的三条路径。各路径独立
进程，20 次预热、100 次采样，每次读回首元素；A1 路径逐次检查命中。

| 路径 | 中位延迟 ms | p20-p80 ms |
|---|---:|---:|
| 原生 | 0.050884 | 0.050259-0.054169 |
| 直调自研 Triton | 0.105278 | 0.101068-0.119737 |
| A1 入口 | 0.154227 | 0.142068-0.193556 |

A1 比直调多 `0.048949 ms`，包括注册 wrapper 和 `autograd.Function`
包装，不是纯 PyTorch dispatcher 时间。原始数据为
[`dispatch_p800_xpu5_63x127x65.json`](dispatch_p800_xpu5_63x127x65.json)。
此处是二维输入，上表结构化用例为 batch=1 的三维输入，不能把两者
延迟直接相除。该增量有可见波动，待正式卡稳定环境重采。

锁定版 `flag_gems.ops` 仅暴露 `mm`、`bmm`，无 `linalg_matmul`。3D 对照调用 `flag_gems.ops.bmm`，在自动调参的 `triton.testing.do_bench` 中报 `ZeroDivisionError: float division by zero`，两组都没有有效耗时。基准 JSON 显式记录了失败原因。

`scripts/perf_compare.py --device p800-kunlunxin` 输出 `FAIL 0 / WARN 0 / NEW 4`：P800 入库基线没有 matmul 四条用例，`OK` 只表示无已知基线上的失败，**不表示 matmul 门禁通过**。不在物理 XPU 5 上覆盖正式 P800 基线；待目标物理 XPU 2 空闲、同环境重采后建立并复跑门禁。

其他国产平台未测。

## 4. 复现

```bash
CUDA_VISIBLE_DEVICES=1,5 python scripts/perf_run.py --device p800-kunlunxin --pattern ops.matmul
CUDA_VISIBLE_DEVICES=1,5 python scripts/perf_compare.py --device p800-kunlunxin
CUDA_VISIBLE_DEVICES=1,5 python ops/matmul/script/bench_perf.py --device p800-kunlunxin --shape 1 63 127 65 --dtype bfloat16
CUDA_VISIBLE_DEVICES=1,5 python ops/matmul/script/bench_dispatch.py --device p800-kunlunxin
```
