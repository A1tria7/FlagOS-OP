"""A1 registration for aten::matmul; registration lives for the process."""
from __future__ import annotations

import os

import torch

try:
    from .kernel.triton_level import matmul_backward_triton, matmul_triton
    from .kernel.torch_level import matmul_backward_torch, matmul_torch
except ImportError:
    from kernel.triton_level import matmul_backward_triton, matmul_triton
    from kernel.torch_level import matmul_backward_torch, matmul_torch


def dispatch_key_for_profile(profile) -> str:
    # P800's CUDA key catches forward but leaves aten::matmul_backward broken.
    if profile.vendor == "kunlunxin" and profile.dispatch_key == "CUDA":
        return "AutogradCUDA"
    return profile.dispatch_key


class _MatmulA1Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, backend):
        ctx.backend = backend
        ctx.save_for_backward(a, b)
        return (matmul_triton if backend == "triton" else matmul_torch)(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        if torch.is_grad_enabled():
            raise RuntimeError("matmul A1 does not support higher-order gradients")
        a, b = ctx.saved_tensors
        backward = (matmul_backward_triton if ctx.backend == "triton"
                    else matmul_backward_torch)
        grad_a, grad_b = backward(a, b, grad_output, *ctx.needs_input_grad[:2])
        return grad_a, grad_b, None


def register_a1(dispatch_key: str, counter: dict | None = None,
                backend: str = "triton", hit_log: str | None = None,
                hit_marker: str | None = None):
    if backend not in ("triton", "torch"):
        raise ValueError(f"unsupported matmul backend: {backend}")

    def impl(self, other):
        if counter is not None:
            counter["forward"] = counter.get("forward", 0) + 1
        if hit_log and (hit_marker is None or os.path.exists(hit_marker)):
            with open(hit_log, "a", encoding="utf-8") as log:
                log.write(f"{os.getpid()} {tuple(self.shape)} {tuple(other.shape)}\n")
        return _MatmulA1Function.apply(self, other, backend)

    lib = torch.library.Library("aten", "IMPL")
    lib.impl("matmul", impl, dispatch_key)
    return lib
