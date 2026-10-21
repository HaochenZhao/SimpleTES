# Paper best-solution restart seed.
# Copied from instance-7bf43cb3/db_state_115425/best_program.py
# First-round best: score 0.35770232805784163 (~2.796 ms), node 48644250391841f5a28b97a3d53233aa, chain 22.
# EVOLVE-BLOCK-START
import torch
import triton
import triton.language as tl
from typing import Tuple, Dict
from utils import DisableCuDNNTF32

# ----------------------------------------------------------------------
# Autotune configuration utilities
# ----------------------------------------------------------------------
def _layernorm_autotune_configs():
    configs = []
    for BLOCK_M in (64, 128, 256):
        for BLOCK_D in (32, 64, 128, 256):
            threads = BLOCK_M
            if threads > 1024:
                continue
            num_warps = max(4, (threads + 31) // 32)
            configs.append(
                triton.Config(
                    {"BLOCK_M": BLOCK_M, "BLOCK_D": BLOCK_D},
                    num_warps=num_warps,
                    num_stages=3,
                )
            )
    # fallback configuration
    configs.append(
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_D": 128},
            num_warps=8,
            num_stages=3,
        )
    )
    return configs


def _proj_gate_autotune_configs():
    configs = []
    for BLOCK_M in (64, 128):
        for BLOCK_H in (64, 128, 256):
            for BLOCK_K in (16, 32, 64):
                threads = BLOCK_M
                if threads > 1024:
                    continue
                num_warps = max(4, (threads + 31) // 32)
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_M": BLOCK_M,
                            "BLOCK_H": BLOCK_H,
                            "BLOCK_K": BLOCK_K,
                        },
                        num_warps=num_warps,
                        num_stages=3,
                    )
                )
    # fallback configuration
    configs.append(
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_H": 256, "BLOCK_K": 32},
            num_warps=8,
            num_stages=3,
        )
    )
    return configs


def _linear_sigmoid_autotune_configs():
    # reuse the same space as projection kernel
    return _proj_gate_autotune_configs()


def _batched_gemm_autotune_configs():
    configs = []
    for BLOCK_M in (64, 128, 256):
        for BLOCK_N in (64, 128, 256):
            for BLOCK_K in (32, 64, 128):
                threads = max(BLOCK_M, BLOCK_N)
                if threads > 1024:
                    continue
                num_warps = max(4, (threads + 31) // 32)
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_M": BLOCK_M,
                            "BLOCK_N": BLOCK_N,
                            "BLOCK_K": BLOCK_K,
                        },
                        num_warps=num_warps,
                        num_stages=5,
                    )
                )
    # asymmetric configs for larger N
    configs.append(
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},
            num_warps=8,
            num_stages=5,
        )
    )
    configs.append(
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=8,
            num_stages=5,
        )
    )
    # fallback
    configs.append(
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=8,
            num_stages=5,
        )
    )
    return configs


def _final_linear_autotune_configs():
    configs = []
    for BLOCK_M in (64, 128, 256):
        for BLOCK_N in (64, 128, 256):
            for BLOCK_K in (32, 64, 128):
                threads = max(BLOCK_M, BLOCK_N)
                if threads > 1024:
                    continue
                num_warps = max(4, (threads + 31) // 32)
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_M": BLOCK_M,
                            "BLOCK_N": BLOCK_N,
                            "BLOCK_K": BLOCK_K,
                        },
                        num_warps=num_warps,
                        num_stages=4,
                    )
                )
    # asymmetric configs for large dim
    configs.append(
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},
            num_warps=8,
            num_stages=4,
        )
    )
    configs.append(
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=8,
            num_stages=4,
        )
    )
    # fallback
    configs.append(
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_warps=8,
            num_stages=4,
        )
    )
    return configs


# ----------------------------------------------------------------------
# LayerNorm kernel (FP32 input → FP16 output)
# ----------------------------------------------------------------------
@triton.autotune(configs=_layernorm_autotune_configs(), key=["M", "D"])
@triton.jit
def _layernorm_kernel(
    x_ptr,          # (M, D) FP32 input
    weight_ptr,     # (D,) FP32 weight
    bias_ptr,       # (D,) FP32 bias
    out_ptr,        # (M, D) FP16 output
    M,
    D,
    stride_xm,
    stride_xd,
    stride_w,   # placeholder (unused)
    stride_b,   # placeholder (unused)
    stride_out_m,
    stride_out_d,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EPS: tl.constexpr = 1e-5,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_M
    offs_m = start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # First pass: compute mean & variance per row
    sum_val = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for d in range(0, D, BLOCK_D):
        offs_d = d + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask = mask_m[:, None] & mask_d[None, :]

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_d[None, :] * stride_xd,
            mask=mask,
            other=0.0,
        )
        sum_val += tl.sum(x, axis=1)
        sum_sq += tl.sum(x * x, axis=1)

    dim_f = tl.full([], D, tl.float32)
    mean = sum_val / dim_f
    var = sum_sq / dim_f - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Second pass: normalize and apply affine transform
    for d in range(0, D, BLOCK_D):
        offs_d = d + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        mask = mask_m[:, None] & mask_d[None, :]

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_d[None, :] * stride_xd,
            mask=mask,
            other=0.0,
        )
        x_norm = (x - mean[:, None]) * inv_std[:, None]

        w = tl.load(weight_ptr + offs_d, mask=mask_d, other=0.0)
        b = tl.load(bias_ptr + offs_d, mask=mask_d, other=0.0)

        out = (x_norm * w[None, :] + b[None, :]).to(tl.float16)

        tl.store(
            out_ptr + offs_m[:, None] * stride_out_m + offs_d[None, :] * stride_out_d,
            out,
            mask=mask,
        )


# ----------------------------------------------------------------------
# Joint projection + gating (produces left/right in B,H,N,N layout)
# ----------------------------------------------------------------------
@triton.autotune(configs=_proj_gate_autotune_configs(), key=["M", "D", "H"])
@triton.jit
def _proj_gate_two_sided_opt(
    x_ptr,                     # (M, D) FP16 input (flattened)
    left_proj_w_ptr,
    left_gate_w_ptr,
    right_proj_w_ptr,
    right_gate_w_ptr,
    left_out_ptr,              # (B, H, N, N) FP16
    right_out_ptr,             # (B, H, N, N) FP16
    B, N, D, H,
    stride_xm, stride_xd,
    stride_lp_h, stride_lp_k,
    stride_lg_h, stride_lg_k,
    stride_rp_h, stride_rp_k,
    stride_rg_h, stride_rg_k,
    stride_left_b, stride_left_h, stride_left_i, stride_left_k,
    stride_right_b, stride_right_h, stride_right_i, stride_right_k,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)   # rows over M = B*N*N
    pid_h = tl.program_id(1)   # hidden dimension

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    M_total = B * N * N
    mask_m = offs_m < M_total
    mask_h = offs_h < H

    # decode (b,i,k) from flattened row index
    BN = N * N
    b = offs_m // BN
    rem = offs_m - b * BN
    i = rem // N
    k = rem - i * N

    # FP32 accumulators for stability
    acc_lproj = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc_lgate = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc_rproj = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc_rgate = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for kk in range(0, D, BLOCK_K):
        offs_k = kk + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D

        # Load X tile (M x K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xd,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        # left projection weight (K x H)
        lp = tl.load(
            left_proj_w_ptr + offs_k[:, None] * stride_lp_k + offs_h[None, :] * stride_lp_h,
            mask=mask_k[:, None] & mask_h[None, :],
            other=0.0,
        )
        # left gate weight (K x H)
        lg = tl.load(
            left_gate_w_ptr + offs_k[:, None] * stride_lg_k + offs_h[None, :] * stride_lg_h,
            mask=mask_k[:, None] & mask_h[None, :],
            other=0.0,
        )
        # right projection weight (K x H)
        rp = tl.load(
            right_proj_w_ptr + offs_k[:, None] * stride_rp_k + offs_h[None, :] * stride_rp_h,
            mask=mask_k[:, None] & mask_h[None, :],
            other=0.0,
        )
        # right gate weight (K x H)
        rg = tl.load(
            right_gate_w_ptr + offs_k[:, None] * stride_rg_k + offs_h[None, :] * stride_rg_h,
            mask=mask_k[:, None] & mask_h[None, :],
            other=0.0,
        )

        acc_lproj = tl.dot(x, lp, acc_lproj)
        acc_lgate = tl.dot(x, lg, acc_lgate)
        acc_rproj = tl.dot(x, rp, acc_rproj)
        acc_rgate = tl.dot(x, rg, acc_rgate)

    # Sigmoid gating
    lgate_sig = 1.0 / (1.0 + tl.exp(-acc_lgate))
    rgate_sig = 1.0 / (1.0 + tl.exp(-acc_rgate))

    left_fp32 = acc_lproj * lgate_sig
    right_fp32 = acc_rproj * rgate_sig

    left_fp16 = left_fp32.to(tl.float16)
    right_fp16 = right_fp32.to(tl.float16)

    # Store results in (B, H, N, N) layout
    left_base = left_out_ptr + b * stride_left_b + i * stride_left_i + k * stride_left_k
    right_base = right_out_ptr + b * stride_right_b + i * stride_right_i + k * stride_right_k

    left_ptrs = left_base[:, None] + offs_h[None, :] * stride_left_h
    right_ptrs = right_base[:, None] + offs_h[None, :] * stride_right_h

    store_mask = mask_m[:, None] & mask_h[None, :]
    tl.store(left_ptrs, left_fp16, mask=store_mask)
    tl.store(right_ptrs, right_fp16, mask=store_mask)


# ----------------------------------------------------------------------
# Linear + sigmoid for the out_gate vector
# ----------------------------------------------------------------------
@triton.autotune(configs=_linear_sigmoid_autotune_configs(), key=["M", "D", "H"])
@triton.jit
def _linear_sigmoid_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    M,
    D,
    H,
    stride_xm,
    stride_xk,
    stride_w_h,
    stride_w_k,
    stride_out_m,
    stride_out_h,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_m = offs_m < M
    mask_h = offs_h < H

    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for k in range(0, D, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        w = tl.load(
            weight_ptr + offs_k[:, None] * stride_w_k + offs_h[None, :] * stride_w_h,
            mask=mask_k[:, None] & mask_h[None, :],
            other=0.0,
        )
        acc = tl.dot(x, w, acc)

    gate_fp32 = 1.0 / (1.0 + tl.exp(-acc))
    tl.store(
        out_ptr + offs_m[:, None] * stride_out_m + offs_h[None, :] * stride_out_h,
        gate_fp32.to(tl.float16),
        mask=mask_m[:, None] & mask_h[None, :],
    )


# ----------------------------------------------------------------------
# Fuse output LayerNorm + out_gate multiplication
# ----------------------------------------------------------------------
@triton.jit
def _outnorm_gate_kernel(
    out_ptr,
    gate_ptr,
    norm_w_ptr,
    norm_b_ptr,
    outnorm_ptr,
    total_positions: tl.int32,
    hidden_dim: tl.int32,
    eps: tl.float32,
    stride_out_p,
    stride_out_h,
    stride_gate_p,
    stride_gate_h,
    stride_outnorm_p,
    stride_outnorm_h,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= total_positions:
        return

    out_row = out_ptr + pid * stride_out_p
    gate_row = gate_ptr + pid * stride_gate_p
    outnorm_row = outnorm_ptr + pid * stride_outnorm_p

    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden_dim

    out_vec = tl.load(out_row + offs_h * stride_out_h, mask=mask_h, other=0.0)
    gate_vec = tl.load(gate_row + offs_h * stride_gate_h, mask=mask_h, other=0.0)

    out_f32 = out_vec.to(tl.float32)
    gate_f32 = gate_vec.to(tl.float32)

    # LayerNorm (FP32) over hidden dimension
    sum_val = tl.sum(out_f32)
    sum_sq = tl.sum(out_f32 * out_f32)
    dim_f = tl.full([], hidden_dim, tl.float32)
    mean = sum_val / dim_f
    var = sum_sq / dim_f - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    w_norm = tl.load(norm_w_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    b_norm = tl.load(norm_b_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)

    normalized = (out_f32 - mean) * inv_std * w_norm + b_norm
    outnorm_f32 = normalized * gate_f32

    tl.store(
        outnorm_row + offs_h * stride_outnorm_h,
        outnorm_f32.to(tl.float16),
        mask=mask_h,
    )


# ----------------------------------------------------------------------
# Batched GEMM kernel for the N³ contraction (B*H, N, N) @ (B*H, N, N)ᵀ
# ----------------------------------------------------------------------
@triton.autotune(configs=_batched_gemm_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def _batched_gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_a_batch,
    stride_a_row,
    stride_a_col,
    stride_b_batch,
    stride_b_row,
    stride_b_col,
    stride_c_batch,
    stride_c_row,
    stride_c_col,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(2)  # batch dimension
    pid_m = tl.program_id(0)  # row tile
    pid_n = tl.program_id(1)  # col tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a = tl.load(
            A_ptr
            + pid_b * stride_a_batch
            + offs_m[:, None] * stride_a_row
            + offs_k[None, :] * stride_a_col,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            B_ptr
            + pid_b * stride_b_batch
            + offs_k[:, None] * stride_b_row
            + offs_n[None, :] * stride_b_col,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    c_ptrs = C_ptr + pid_b * stride_c_batch \
             + offs_m[:, None] * stride_c_row \
             + offs_n[None, :] * stride_c_col
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


# ----------------------------------------------------------------------
# Final linear projection kernel (M x hidden_dim) @ (hidden_dim x dim) -> (M x dim)
# ----------------------------------------------------------------------
@triton.autotune(configs=_final_linear_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def _final_linear_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_a_m,
    stride_a_k,
    stride_b_k,
    stride_b_n,
    stride_c_m,
    stride_c_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kk in range(0, K, BLOCK_K):
        offs_k = kk + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a = tl.load(
            A_ptr + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            B_ptr + offs_k[:, None] * stride_b_k + offs_n[None, :] * stride_b_n,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    # Store result as FP32
    c_ptrs = C_ptr + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# ----------------------------------------------------------------------
# Main custom kernel exposing the TriMul forward pass
# ----------------------------------------------------------------------
def custom_kernel(
    data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]
) -> torch.Tensor:
    """
    Forward‑only outgoing TriMul implementation using Triton kernels.
    Returns a FP32 tensor of shape [B, N, N, dim].
    """
    input_tensor, mask, weights, config = data
    device = input_tensor.device
    dim = int(config["dim"])
    hidden_dim = int(config["hidden_dim"])
    nomask = bool(config.get("nomask", True))

    with DisableCuDNNTF32():
        # --------------------------------------------------------------
        # 1️⃣ Input LayerNorm (FP32 stats → FP16 result)
        # --------------------------------------------------------------
        B, N, _, _ = input_tensor.shape
        M = B * N * N  # total positions (batch * seq_len^2)

        # Flatten for LayerNorm: shape (M, dim)
        x_flat = input_tensor.reshape(M, dim).contiguous()          # (M, dim) FP32
        x_norm = torch.empty((M, dim), dtype=torch.float16, device=device)

        stride_xm = x_flat.stride(0)
        stride_xd = x_flat.stride(1)

        grid_ln = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        _layernorm_kernel[grid_ln](
            x_ptr=x_flat,
            weight_ptr=weights["norm.weight"],
            bias_ptr=weights["norm.bias"],
            out_ptr=x_norm,
            M=M,
            D=dim,
            stride_xm=stride_xm,
            stride_xd=stride_xd,
            stride_w=weights["norm.weight"].stride(0),   # placeholder
            stride_b=weights["norm.bias"].stride(0),     # placeholder
            stride_out_m=x_norm.stride(0),
            stride_out_d=x_norm.stride(1),
        )

        # --------------------------------------------------------------
        # 2️⃣ Cast model weights to FP16 where appropriate
        # --------------------------------------------------------------
        left_proj_w   = weights["left_proj.weight"].to(torch.float16).contiguous()
        right_proj_w  = weights["right_proj.weight"].to(torch.float16).contiguous()
        left_gate_w   = weights["left_gate.weight"].to(torch.float16).contiguous()
        right_gate_w  = weights["right_gate.weight"].to(torch.float16).contiguous()
        out_gate_w    = weights["out_gate.weight"].to(torch.float16).contiguous()
        # keep output LayerNorm params in FP32 for numerical stability
        to_out_norm_w = weights["to_out_norm.weight"]
        to_out_norm_b = weights["to_out_norm.bias"]
        to_out_w      = weights["to_out.weight"].to(torch.float16).contiguous()   # (dim, hidden_dim)

        # --------------------------------------------------------------
        # 3️⃣ Allocate buffers for left/right projections (B, H, N, N) FP16
        # --------------------------------------------------------------
        left = torch.empty((B, hidden_dim, N, N), dtype=torch.float16, device=device)
        right = torch.empty((B, hidden_dim, N, N), dtype=torch.float16, device=device)

        # --------------------------------------------------------------
        # 4️⃣ Joint projection + gating (outputs directly in B,H,N,N layout)
        # --------------------------------------------------------------
        grid_proj = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(hidden_dim, meta["BLOCK_H"]),
        )
        _proj_gate_two_sided_opt[grid_proj](
            x_ptr=x_norm,
            left_proj_w_ptr=left_proj_w,
            left_gate_w_ptr=left_gate_w,
            right_proj_w_ptr=right_proj_w,
            right_gate_w_ptr=right_gate_w,
            left_out_ptr=left,
            right_out_ptr=right,
            B=B,
            N=N,
            D=dim,
            H=hidden_dim,
            stride_xm=stride_xm,
            stride_xd=stride_xd,
            stride_lp_h=left_proj_w.stride(0),
            stride_lp_k=left_proj_w.stride(1),
            stride_lg_h=left_gate_w.stride(0),
            stride_lg_k=left_gate_w.stride(1),
            stride_rp_h=right_proj_w.stride(0),
            stride_rp_k=right_proj_w.stride(1),
            stride_rg_h=right_gate_w.stride(0),
            stride_rg_k=right_gate_w.stride(1),
            stride_left_b=left.stride(0),
            stride_left_h=left.stride(1),
            stride_left_i=left.stride(2),
            stride_left_k=left.stride(3),
            stride_right_b=right.stride(0),
            stride_right_h=right.stride(1),
            stride_right_i=right.stride(2),
            stride_right_k=right.stride(3),
        )

        # --------------------------------------------------------------
        # 5️⃣ Compute out_gate = sigmoid(x_norm @ out_gate_w.T)
        # --------------------------------------------------------------
        grid_gate = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(hidden_dim, meta["BLOCK_H"]),
        )
        gate = torch.empty((M, hidden_dim), dtype=torch.float16, device=device)
        _linear_sigmoid_kernel[grid_gate](
            x_ptr=x_norm,
            weight_ptr=out_gate_w,
            out_ptr=gate,
            M=M,
            D=dim,
            H=hidden_dim,
            stride_xm=stride_xm,
            stride_xk=stride_xd,
            stride_w_h=out_gate_w.stride(0),
            stride_w_k=out_gate_w.stride(1),
            stride_out_m=gate.stride(0),
            stride_out_h=gate.stride(1),
        )

        # --------------------------------------------------------------
        # 6️⃣ Optional binary mask (broadcast over hidden dimension)
        # --------------------------------------------------------------
        if not nomask and mask is not None and mask.numel() > 0:
            mask_fp16 = mask.to(dtype=torch.float16, device=device).unsqueeze(1)  # (B,1,N,N)
            left.mul_(mask_fp16)
            right.mul_(mask_fp16)

        # --------------------------------------------------------------
        # 7️⃣ Core N³ contraction via batched GEMM (B*H, N, N) @ (B*H, N, N)ᵀ
        # --------------------------------------------------------------
        batch_dim = B * hidden_dim
        left_gemm = left.view(batch_dim, N, N)   # (B*H, N, N)
        right_gemm = right.view(batch_dim, N, N) # (B*H, N, N)

        out_gemm = torch.empty((batch_dim, N, N), dtype=torch.float16, device=device)

        stride_a_batch = left_gemm.stride(0)
        stride_a_row   = left_gemm.stride(1)
        stride_a_col   = left_gemm.stride(2)

        stride_b_batch = right_gemm.stride(0)
        # Transposed view for the second operand
        stride_b_row   = right_gemm.stride(2)   # K dimension after transpose
        stride_b_col   = right_gemm.stride(1)   # N dimension after transpose

        stride_c_batch = out_gemm.stride(0)
        stride_c_row   = out_gemm.stride(1)
        stride_c_col   = out_gemm.stride(2)

        grid_gemm = lambda meta: (
            triton.cdiv(N, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
            batch_dim,
        )
        _batched_gemm_kernel[grid_gemm](
            A_ptr=left_gemm,
            B_ptr=right_gemm,
            C_ptr=out_gemm,
            M=N,
            N=N,
            K=N,
            stride_a_batch=stride_a_batch,
            stride_a_row=stride_a_row,
            stride_a_col=stride_a_col,
            stride_b_batch=stride_b_batch,
            stride_b_row=stride_b_row,
            stride_b_col=stride_b_col,
            stride_c_batch=stride_c_batch,
            stride_c_row=stride_c_row,
            stride_c_col=stride_c_col,
        )

        # reshape back to (B, N, N, hidden_dim)
        out = (
            out_gemm.view(B, hidden_dim, N, N)
                .permute(0, 2, 3, 1)
                .contiguous()
        )  # (B, N, N, hidden_dim)

        # --------------------------------------------------------------
        # 8️⃣ Fuse output LayerNorm + out_gate multiplication
        # --------------------------------------------------------------
        total_positions = B * N * N
        out_flat = out.view(total_positions, hidden_dim).contiguous()
        gate_flat = gate.view(total_positions, hidden_dim).contiguous()
        outnorm_flat = torch.empty_like(out_flat)  # FP16 buffer

        stride_out_p = out_flat.stride(0)
        stride_out_h = out_flat.stride(1)
        stride_gate_p = gate_flat.stride(0)
        stride_gate_h = gate_flat.stride(1)
        stride_outnorm_p = outnorm_flat.stride(0)
        stride_outnorm_h = outnorm_flat.stride(1)

        grid_norm = (total_positions,)
        _outnorm_gate_kernel[grid_norm](
            out_ptr=out_flat,
            gate_ptr=gate_flat,
            norm_w_ptr=to_out_norm_w,
            norm_b_ptr=to_out_norm_b,
            outnorm_ptr=outnorm_flat,
            total_positions=total_positions,
            hidden_dim=hidden_dim,
            eps=1e-5,
            stride_out_p=stride_out_p,
            stride_out_h=stride_out_h,
            stride_gate_p=stride_gate_p,
            stride_gate_h=stride_gate_h,
            stride_outnorm_p=stride_outnorm_p,
            stride_outnorm_h=stride_outnorm_h,
            BLOCK_H=hidden_dim,
        )

        # --------------------------------------------------------------
        # 9️⃣ Final linear projection (hidden_dim → dim) fully Triton‑based
        # --------------------------------------------------------------
        to_out_T = to_out_w.t().contiguous()   # (hidden_dim, dim)

        final_out = torch.empty((total_positions, dim), dtype=torch.float32, device=device)

        stride_a_m = outnorm_flat.stride(0)
        stride_a_k = outnorm_flat.stride(1)
        stride_b_k = to_out_T.stride(0)
        stride_b_n = to_out_T.stride(1)
        stride_c_m = final_out.stride(0)
        stride_c_n = final_out.stride(1)

        grid_final = lambda meta: (
            triton.cdiv(total_positions, meta["BLOCK_M"]),
            triton.cdiv(dim, meta["BLOCK_N"]),
        )
        _final_linear_kernel[grid_final](
            A_ptr=outnorm_flat,
            B_ptr=to_out_T,
            C_ptr=final_out,
            M=total_positions,
            N=dim,
            K=hidden_dim,
            stride_a_m=stride_a_m,
            stride_a_k=stride_a_k,
            stride_b_k=stride_b_k,
            stride_b_n=stride_b_n,
            stride_c_m=stride_c_m,
            stride_c_n=stride_c_n,
        )

        output = final_out.view(B, N, N, dim).contiguous()
        return output
# EVOLVE-BLOCK-END