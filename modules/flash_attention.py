import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
#from triton.tools.tensor_descriptor import TensorDescriptor

#DEVICE = triton.runtime.driver.active.get_active_torch_function()
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def _fused_attn_fw_inner(
    Q,k_ptr,v_ptr,
    acc,l_i,m_i,
    S,
    q_idx, 
    q_offset,k_t_offsets, v_offsets, kv_offsets,
    BLOCK_Q: tl.constexpr,BLOCK_KV: tl.constexpr,
    DIAG: tl.constexpr,
    stride_K_S, stride_V_S,
    softmax_scale
):
    if DIAG:
        lo = q_idx * BLOCK_Q
        hi = (q_idx + 1) * BLOCK_Q
        lo = tl.multiple_of(lo, BLOCK_Q) #compiler optimization
    else: 
        lo, hi = 0, q_idx * BLOCK_Q

    k_t_offsets += lo * stride_K_S
    v_offsets += lo * stride_V_S
    kv_offsets += lo

    for j in tl.range(lo, hi, BLOCK_KV):
        #flag for compiler optimization
        j = tl.multiple_of(j, BLOCK_KV)

        #load K, V to SRAM
        mask_kv = kv_offsets < S
        K_T = tl.load(k_ptr + k_t_offsets, mask=mask_kv[None, :], other=0.) # shape (Dh, BLOCK_KV)
        V = tl.load(v_ptr + v_offsets, mask=mask_kv[:, None], other=0.) # shape (BLOCK_KV, Dh)

        #compute s_ij=QK^T
        s_ij = tl.dot(Q, K_T) * softmax_scale 
        if DIAG:
            causal_mask = q_offset[:, None] >= (kv_offsets[None, :])
            s_ij += tl.where(causal_mask, 0, -1.0e6) # shape (BLOCK_Q, BLOCK_KV)

        #compute m_ij rowmax, p, and 
        m_i_new = tl.maximum(m_i, tl.max(s_ij, axis=1)).to(tl.float32) # shape is (BLOCK_Q)
        s_ij = s_ij.to(tl.float32) - m_i_new[:, None] # shape (BLOCK_Q, BLOCK_KV)
        P = tl.exp2(s_ij) # shape (BLOCK_Q, BLOCK_KV)

        # compute updated denom
        l_new = tl.sum(P, axis=1) # shape (BLOCK_Q)
        alpha = tl.exp2(m_i - m_i_new) # shape (BLOCK_SIZE_Q)
        l_i =  ((l_i*alpha) + l_new) # shape (BLOCK_Q)

        #acc = P @ V + acc * alpha
        acc = acc * alpha[:, None] # shape (BLOCK_Q, Dh)
        acc = tl.dot(P, V, acc=acc) # shape (BLOCK_Q, Dh)
        m_i = m_i_new

        #update pointers
        k_t_offsets += BLOCK_KV * stride_K_S
        v_offsets += BLOCK_KV * stride_V_S
        kv_offsets += BLOCK_KV

    return acc, l_i, m_i 


@triton.jit
def fused_attn_fw(
    Q, K, V, O, #ptrs to matrices, size in (B, H, s_ij, Dh)
    LSE, #log sum exp for backwards pass
    B: tl.constexpr,H: tl.constexpr,S: tl.constexpr,Dh: tl.constexpr, #(B,H,s_ij,Dh)
    scale, #1/sqrt(d)
    BLOCK_Q: tl.constexpr, #block size of Q dimension
    BLOCK_KV: tl.constexpr,#block size of KV dimension
):
    ##Openai is weird ...
    #advised via forums for compiler to avoid crappy register alloc/codegen
    tl.static_assert(BLOCK_KV <= Dh) 

    ln2: tl.constexpr = 1.4426950408889634 #(1/tl.math.log(2))
    scale *= ln2

    #get associated head
    q_idx = tl.program_id(0)
    block_kv_ptr = tl.program_id(1)
    batch_idx = block_kv_ptr // H
    head_idx = block_kv_ptr % H

    #strides (to calc pointers)
    batch_stride = H*S*Dh 
    head_stride  = S*Dh

    #create pointers
    q_ptr = Q + batch_idx*batch_stride + head_idx*head_stride 
    k_ptr = K + batch_idx*batch_stride + head_idx*head_stride 
    v_ptr = V + batch_idx*batch_stride + head_idx*head_stride
    o_ptr = O + batch_idx*batch_stride + head_idx*head_stride

    #offsets space on 1 dimension
    q_offset = q_idx*BLOCK_Q + tl.arange(0,BLOCK_Q)
    kv_offset = tl.arange(0,BLOCK_KV)
    dh_offset = tl.arange(0,Dh)

    #final offsets, 2D for each tensor
    q_offsets           = q_offset [:, None] * Dh + dh_offset[None, :]       # [block_q_size, Dh]
    k_t_offsets = dh_offset[:, None]      + kv_offset[None, :] * Dh  # [Dh, block_k_size]
    v_offsets           = kv_offset[:, None] * Dh + dh_offset[None, :]       # [block_v_size, Dh]

    #load q, padding mask
    q_mask = q_offset < S
    q = tl.load(q_ptr + q_offsets, mask=q_mask[:, None], other=0.)  # [BLOCK_Q, Dh]

    #base vars for running softmax
    m_i = tl.zeros([BLOCK_Q],dtype=tl.float32) - float("inf") #running max
    l_i = tl.zeros([BLOCK_Q],dtype=tl.float32) + 1            #running sum (denom)
    acc = tl.zeros([BLOCK_Q,Dh],dtype=tl.float32)             #accumulator tensor(Q,)

    #inner loop
    #NOTE: we pass in q_offset kv_offset and not the 2D q_offsets kv_offsets
    acc, l_i, m_i = _fused_attn_fw_inner(q,k_ptr,v_ptr,acc,l_i,m_i,S,q_idx,q_offset,k_t_offsets,v_offsets,kv_offset,BLOCK_Q,BLOCK_KV,False,Dh,Dh,scale)
    acc, l_i, m_i = _fused_attn_fw_inner(q,k_ptr,v_ptr,acc,l_i,m_i,S,q_idx,q_offset,k_t_offsets,v_offsets,kv_offset,BLOCK_Q,BLOCK_KV,True,Dh,Dh,scale)   

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    o_offsets = q_offset[:, None] * Dh + dh_offset[None, :]
    o_mask = (q_offset < S)[:, None]

    # write output [BLOCK_Q, Dh] and per-row LSE [BLOCK_Q]
    #store logsumexp for backwards pass
    lse_ptr = LSE + (batch_idx * H + head_idx) * S
    row_mask = q_mask
    tl.store(o_ptr + o_offsets, acc, mask=o_mask)
    tl.store(lse_ptr + q_offset, m_i, mask=row_mask)

@triton.jit
def backpropt_attention_propagate():
    pass

#compute 
def forward_attention_compute(q, k, v, sm_scale, dtype=torch.float32):
    #Temporary assertions: change for testing
    assert q.shape == k.shape == v.shape, "q, k, v must have identical shape (B, H, S, Dh)"
    assert q.device == k.device == v.device, "q, k, v must be on the same device"
    assert q.ndim == 4, "expected shape (B, H, S, Dh)"
    assert q.is_cuda, "Triton kernel requires CUDA tensors"
    assert q.dtype == k.dtype == v.dtype, "q, k, v dtypes must match"
    assert q.dtype in (torch.float16, torch.bfloat16, torch.float32), "supported dtypes: fp16, bf16, fp32"

    B, H, S, Dh = q.shape
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    O = torch.empty_like(q, dtype=q.dtype)
    LSE = torch.empty((B, H, S), device=q.device, dtype=dtype)

    # Keep KV tile <= Q tile for correctness with current two-pass causal split.
    BLOCK_Q = 64 if S >= 64 else triton.next_power_of_2(S)
    BLOCK_KV = BLOCK_Q

    grid = (triton.cdiv(S, BLOCK_Q), B * H)
    fused_attn_fw[grid](
        q,
        k,
        v,
        O,
        LSE,
        B,
        H,
        S,
        Dh,
        sm_scale,
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV
    )
    return O

def back_prop_attention_compute():
    pass

#basic sanity test for FW pass
def test_flash_attention_kernel(B=2, H=4, S=128, Dh=64, device=None, atol=5e-2, rtol=1e-2):
    if device is None:
        device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Please Use Cuda to run Test")

    torch.manual_seed(0)
    # Use fp16 by default to match common flash-attention usage.
    q = torch.randn((B, H, S, Dh), dtype=torch.float32, device=device)
    k = torch.randn((B, H, S, Dh), dtype=torch.float32, device=device)
    v = torch.randn((B, H, S, Dh), dtype=torch.float32, device=device)

    sm_scale = 1.0 / math.sqrt(Dh)
    triton_out = forward_attention_compute(q, k, v, sm_scale)
    pytorch_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sm_scale)

    torch.testing.assert_close(triton_out, pytorch_out, atol=atol, rtol=rtol)
    print(f"passed fwd: B={B}, H={H}, S={S}, Dh={Dh}, dtype={q.dtype}")

#map datatypes
def _normalize_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        m = {
            "fp16": torch.float16, "float16": torch.float16, "torch.float16": torch.float16,
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "torch.bfloat16": torch.bfloat16,
            "fp32": torch.float32, "float32": torch.float32, "torch.float32": torch.float32,
        }
        if dtype in m:
            return m[dtype]
    print("WARNING: Invalid type provided, defaulting to f32")
    return torch.float32

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["S"],
        x_vals=[128, 256, 512, 1024, 2048],
        line_arg="provider",
        line_vals=["triton", "sdpa"],
        line_names=["TritonKernel", "TorchSDPA"],
        ylabel="Latency (ms)",
        plot_name="flash_attention_forward_benchmark",
        args={},
    )
)
def benchmark_flash_attention_kernel(S, provider, B=2, H=8, Dh=64, dtype=tl.float32):
    #bug with testing fp16 or other mismatching floating points, changing above to f32 produces bug
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarking.")
    device = torch.device("cuda")
    torch.manual_seed(0)
    dtype=_normalize_dtype(dtype)
    q = torch.randn((B, H, S, Dh), dtype=dtype, device=device)
    k = torch.randn((B, H, S, Dh), dtype=dtype, device=device)
    v = torch.randn((B, H, S, Dh), dtype=dtype, device=device)
    sm_scale = 1.0 / math.sqrt(Dh)

    if provider == "triton":
        fn = lambda: forward_attention_compute(q, k, v, sm_scale)
    else:
        fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sm_scale)

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms

if __name__ == "__main__":
    #NOTE: this implementation is based on Flash Attention Original and Version 2
    #Sanity test for flash_attention_kernel forward pass
    test_flash_attention_kernel()
    benchmark_flash_attention_kernel.run(show_plots=True, print_data=True, save_path="../output/plots")
    #Sanity test for flash_attention_kernel backwards pass
    #TODO: back prop
    #integrate with 