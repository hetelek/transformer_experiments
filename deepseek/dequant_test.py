#!/usr/bin/env python3
"""
Test script to verify that Triton-based dequantization matches a pure-Python reference.
Requires CUDA (for Triton kernels) and Triton installed.
"""
import sys
import torch

from kernel import act_quant, weight_dequant

from dequantize import weight_dequant_2

def main():
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available; skipping Triton dequantization test.")
        sys.exit(0)
    device = torch.device("cuda")

    # Block size used by both implementations
    block_size = 128
    torch.manual_seed(0)

    # Create a random test tensor with dimensions multiple of block_size
    M = block_size * 3
    N = block_size * 2
    x = torch.randn(M, N, dtype=torch.float32, device=device).contiguous()

    # Quantize to FP8 (Triton)
    x_q, s = act_quant(x, block_size)
    # Ensure contiguous
    x_q = x_q.contiguous()
    s = s.contiguous()

    # Dequantize via Triton kernel (float32)
    y_triton = weight_dequant(x_q, s, block_size)

    # Reference dequantization (pure Python)
    # Cast to float16 for Python path (block-wise ops safe in half precision)
    x_q_fp16 = x_q.to(torch.float16)
    s_fp16 = s.to(torch.float16)
    y_ref = weight_dequant_2(x_q_fp16, s_fp16)
    # Cast reference output back to float32 and move to CUDA for comparison
    y_ref = y_ref.to(device=device, dtype=torch.float32)

    # Compare outputs
    diff = (y_triton - y_ref).abs().max()
    print(f"Max abs difference: {diff.item():.6f}")
    tol = 1e-3
    if diff > tol:
        print(f"TEST FAILED: difference {diff.item():.6f} > tol {tol}")
        sys.exit(1)
    print("TEST PASSED: Triton and Python dequant outputs match within tolerance.")

if __name__ == "__main__":
    main()