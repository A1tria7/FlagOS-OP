#!/usr/bin/env python3
"""Compare matmul forward latency on one target device."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent.parent
INPUT_SEED = 20260929
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))


def _device(value):
    if value == "cpu" or ":" in value:
        return value
    from common.device import load_profile
    return load_profile(value).torch_device


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", required=True, help="profile name or torch device")
    ap.add_argument("--shape", nargs=4, type=int, default=[1, 128, 256, 192],
                    metavar=("B", "M", "K", "N"))
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("float32", "float16", "bfloat16"))
    ap.add_argument("--impl", default="all",
                    choices=("all", "triton", "native", "flaggems"))
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    if args.impl == "all":
        results = {}
        unavailable = {}
        for name in ("triton", "native", "flaggems"):
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--device", args.device, "--dtype", args.dtype,
                       "--shape", *(str(dim) for dim in args.shape),
                       "--impl", name]
            child = subprocess.run(command, text=True, capture_output=True)
            if child.returncode:
                if name == "flaggems":
                    reason = (child.stderr.strip().splitlines() or
                              [f"exit code {child.returncode}"])[-1]
                    unavailable[name] = reason
                    print(f"flaggems: unavailable ({reason})")
                    continue
                sys.stderr.write(child.stdout + child.stderr)
                return child.returncode
            result = json.loads(child.stdout.strip().splitlines()[-1])
            if result is None:
                unavailable[name] = "no linalg_matmul or bmm entry"
                print("flaggems: unavailable")
                continue
            results[name] = result
            print(f"{name:9s} {result['latency_ms']:.4f} ms  "
                  f"{result['tflops']:.2f} TFLOPS  "
                  f"rel={result['relative_error']:.2e}")
        report = {"device": _device(args.device), "shape": args.shape,
                  "dtype": args.dtype, "warmup": 20, "iters": 100,
                  "input_seed": INPUT_SEED,
                  "timing": "per-call scalar readback",
                  "isolation": "one process per implementation",
                  "results": results, "unavailable": unavailable}
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2) + "\n")
            print(f"saved {args.json_out}")
        return 0

    from common.perf import bench
    from kernel.triton_level import matmul_triton

    device = _device(args.device)
    batch, m, k, n = args.shape
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cpu").manual_seed(INPUT_SEED)
    a = torch.randn(batch, m, k, generator=generator).to(
        device=device, dtype=dtype) * 0.25
    b = torch.randn(batch, k, n, generator=generator).to(
        device=device, dtype=dtype) * 0.25
    implementations = {"triton": matmul_triton, "native": torch.matmul}
    entry_name = {"triton": "matmul_triton", "native": "torch.matmul"}.get(args.impl)
    if args.impl == "flaggems":
        try:
            from flag_gems import ops
        except ImportError:
            print("null")
            return 0
        candidate = next((name for name in ("linalg_matmul", "bmm")
                          if callable(getattr(ops, name, None))), None)
        if candidate is None:
            print("null")
            return 0
        implementations["flaggems"] = getattr(ops, candidate)
        entry_name = f"flag_gems.ops.{candidate}"

    reference = torch.matmul(a, b).float()
    fn = implementations[args.impl]
    output = fn(a, b)
    error = ((output.float() - reference).abs().max().item()
             / max(reference.abs().max().item(), 1e-6))
    if error > 5e-2:
        raise AssertionError(f"{args.impl} failed accuracy before timing: {error}")
    # The scalar readback synchronizes each call on the target backend.
    latency = bench(lambda: fn(a, b).reshape(-1)[0].item(),
                    warmup=20, iters=100, device=device)
    tflops = 2 * batch * m * k * n / latency / 1e9
    print(json.dumps({"latency_ms": latency, "tflops": tflops,
                      "relative_error": error, "input_seed": INPUT_SEED,
                      "entry": entry_name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
