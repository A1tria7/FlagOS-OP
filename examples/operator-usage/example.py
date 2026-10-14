#!/usr/bin/env python3
"""连续算子调用样例: native / direct kernel / FlagGems+A1 混合映射。

运行:

    python3 examples/operator-usage/example.py p800-kunlunxin

示例链路:

    token ids
      -> embedding
      -> Q/K/V projection (native ATen matmul)
      -> aten::_scaled_dot_product_attention_math
      -> GELU
      -> output projection (native ATen matmul)

三种映射在同一输入上对比:

1. native        全部走未注册原生路径
2. direct        embedding/sdpa_math 直调 FlagOS-OP kernel, GELU 直调 FlagGems
3. a1-mixed      embedding/sdpa_math 走 A1, GELU 走 FlagGems dispatcher

每种映射都在独立子进程中执行，避免 torch.library 注册互相污染。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

B, H, S, D = 2, 4, 32, 64
HEAD_DIM = D // H
VOCAB = 128
FFN = 128


def _inputs(device: str, dtype):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(20261014)

    def normal(*shape):
        return (
            torch.randn(*shape, generator=generator).mul_(0.1)
            .to(dtype).to(device)
        )

    return {
        "ids": torch.randint(
            0, VOCAB, (B, S), generator=generator).to(device),
        "embedding_weight": normal(VOCAB, D),
        "wq": normal(D, D),
        "wk": normal(D, D),
        "wv": normal(D, D),
        "wout": normal(D, FFN),
    }


def _finish(x, inputs):
    import torch

    batch, _, _, _ = x.shape
    return x.transpose(1, 2).reshape(batch, x.shape[2], -1) @ inputs["wout"]


def _run_mapping(mapping: str, device: str, artifact: Path) -> dict:
    import torch

    data = _inputs(device, torch.float16)
    ids = data["ids"]
    weight = data["embedding_weight"]

    if mapping == "native":
        embedded = torch.nn.functional.embedding(ids, weight)
        q = (embedded @ data["wq"]).view(B, S, H, HEAD_DIM).transpose(1, 2)
        k = (embedded @ data["wk"]).view(B, S, H, HEAD_DIM).transpose(1, 2)
        v = (embedded @ data["wv"]).view(B, S, H, HEAD_DIM).transpose(1, 2)
        context, probabilities = (
            torch.ops.aten._scaled_dot_product_attention_math(
                q, k, v, None, 0.0, True, None, enable_gqa=False,
            )
        )
        activated = torch.nn.functional.gelu(context)
        result = _finish(activated, data)
        routes = {
            "embedding": "native ATen",
            "sdpa_math": "native private math",
            "gelu": "native ATen",
        }
        counters = {}

    elif mapping == "direct":
        from flag_gems import gelu as flaggems_gelu
        from ops.embedding.kernel import p800_kunlunxin as embedding_backend
        from ops.sdpa_math.kernel.p800_fast_level import (
            sdpa_math_p800_fast,
        )

        embedded = embedding_backend.embedding(weight, ids)
        q = (embedded @ data["wq"]).view(B, S, H, HEAD_DIM).transpose(1, 2)
        k = (embedded @ data["wk"]).view(B, S, H, HEAD_DIM).transpose(1, 2)
        v = (embedded @ data["wv"]).view(B, S, H, HEAD_DIM).transpose(1, 2)
        context, probabilities = sdpa_math_p800_fast(
            q, k, v, None, 0.0, True, None, enable_gqa=False,
        )
        activated = flaggems_gelu(context)
        result = _finish(activated, data)
        routes = {
            "embedding": "FlagOS-OP direct P800 kernel",
            "sdpa_math": "FlagOS-OP direct P800 fast path",
            "gelu": "FlagGems direct API",
        }
        counters = {}

    elif mapping == "a1-mixed":
        import flag_gems
        from ops.embedding import register_a1 as register_embedding
        from ops.sdpa_math import register_a1 as register_sdpa_math

        # 将 FlagGems 限定为本链路实际消费的周边算子。
        flag_gems.only_enable(include=["gelu"])

        embedding_calls = {"n": 0}
        attention_calls = {"n": 0}
        # 保存注册对象引用，直到进程退出。
        embedding_lib = register_embedding(
            "AutogradCUDA",
            counter=embedding_calls,
            platform="p800-kunlunxin",
        )
        attention_libs = register_sdpa_math(
            "AutogradCUDA",
            counter=attention_calls,
            impl="p800",
        )
        assert embedding_lib is not None and attention_libs is not None

        embedded = torch.nn.functional.embedding(ids, weight)
        q = (embedded @ data["wq"]).view(B, S, H, HEAD_DIM).transpose(1, 2)
        k = (embedded @ data["wk"]).view(B, S, H, HEAD_DIM).transpose(1, 2)
        v = (embedded @ data["wv"]).view(B, S, H, HEAD_DIM).transpose(1, 2)
        context, probabilities = (
            torch.ops.aten._scaled_dot_product_attention_math(
                q, k, v, None, 0.0, True, None, enable_gqa=False,
            )
        )
        activated = torch.nn.functional.gelu(context)
        result = _finish(activated, data)
        routes = {
            "embedding": "FlagOS-OP A1",
            "sdpa_math": "FlagOS-OP A1",
            "gelu": "FlagGems dispatcher",
        }
        counters = {
            "embedding": embedding_calls["n"],
            "sdpa_math": attention_calls["n"],
        }
        assert embedding_calls["n"] == 1, embedding_calls
        assert attention_calls["n"] == 1, attention_calls
    else:
        raise ValueError(f"unknown mapping: {mapping}")

    assert result.shape == (B, S, FFN)
    assert probabilities.shape == (B, H, S, S)
    assert torch.isfinite(result.float()).all()
    assert torch.isfinite(probabilities.float()).all()

    torch.save(
        {
            "result": result.detach().cpu(),
            "probabilities": probabilities.detach().cpu(),
            "activated": activated.detach().cpu(),
        },
        artifact,
    )
    return {
        "mapping": mapping,
        "routes": routes,
        "counters": counters,
        "result_sum": float(result.float().sum()),
        "probability_sum": float(probabilities.float().sum()),
    }


def _spawn(mapping: str, device: str, artifact: Path) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--mapping", mapping,
        "--device", device,
        "--artifact", str(artifact),
    ]
    process = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, timeout=300,
    )
    if process.returncode:
        print(process.stdout)
        print(process.stderr)
        raise RuntimeError(f"{mapping} child failed")
    lines = [line for line in process.stdout.splitlines()
             if line.startswith("RESULT ")]
    if not lines:
        print(process.stdout)
        print(process.stderr)
        raise RuntimeError(f"{mapping} child returned no result")
    return json.loads(lines[-1][len("RESULT "):])


def _compare(left: Path, right: Path) -> dict:
    import torch

    a = torch.load(left, weights_only=False)
    b = torch.load(right, weights_only=False)
    return {
        key: float((a[key].float() - b[key].float()).abs().max())
        for key in ("result", "probabilities", "activated")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?", default="p800-kunlunxin")
    parser.add_argument("--child", action="store_true", help="internal mode")
    parser.add_argument("--mapping", choices=("native", "direct", "a1-mixed"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()

    if args.profile != "p800-kunlunxin":
        raise ValueError("this runnable sample currently targets p800-kunlunxin")

    if args.child:
        if not args.mapping or not args.artifact:
            raise ValueError("child mode requires --mapping and --artifact")
        result = _run_mapping(args.mapping, args.device, args.artifact)
        print("RESULT " + json.dumps(result))
        return 0

    with tempfile.TemporaryDirectory(prefix="flagos-op-usage-") as temp:
        artifact_dir = Path(temp)
        results = {}
        for mapping in ("native", "direct", "a1-mixed"):
            artifact = artifact_dir / f"{mapping}.pt"
            results[mapping] = _spawn(mapping, args.device, artifact)
            print(f"[{mapping}]")
            for name, route in results[mapping]["routes"].items():
                print(f"  {name:10s} -> {route}")
            if results[mapping]["counters"]:
                print(f"  counters   -> {results[mapping]['counters']}")

        native = artifact_dir / "native.pt"
        print("\nNumerical comparison vs native:")
        for mapping in ("direct", "a1-mixed"):
            diff = _compare(native, artifact_dir / f"{mapping}.pt")
            print(f"  {mapping:8s} -> {diff}")

    print("\nPASS: three mapping modes completed in isolated subprocesses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
