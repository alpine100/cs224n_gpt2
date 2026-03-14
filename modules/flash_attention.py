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
    k_t_offsets         = dh_offset[:, None]      + kv_offset[None, :] * Dh  # [Dh, block_k_size]
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
def backpropt_attention_propagate(
    Q, K, V, O, dO, dQ, dK, dV, LSE, Delta,
    sm_scale,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, Dh: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr
):
    # setup and outer loop over KV blocks
    # instead of a for-loop, triton parallelizes the outer loop over the grid
    # program ID 0 acts as our `j` index (1 <= j <= Tc)
    kv_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    batch_idx = bh_idx // H
    head_idx = bh_idx % H

    batch_stride = H * S * Dh
    head_stride = S * Dh
    offset_bh = batch_idx * batch_stride + head_idx * head_stride

    kv_offset = kv_idx * BLOCK_KV + tl.arange(0, BLOCK_KV)
    dh_offset = tl.arange(0, Dh)

    k_t_offsets = dh_offset[:, None] + kv_offset[None, :] * Dh
    v_t_offsets = dh_offset[:, None] + kv_offset[None, :] * Dh
    k_offsets = kv_offset[:, None] * Dh + dh_offset[None, :]
    v_offsets = kv_offset[:, None] * Dh + dh_offset[None, :]

    k_ptr = K + offset_bh
    v_ptr = V + offset_bh
    dk_ptr = dK + offset_bh
    dv_ptr = dV + offset_bh

    # load K_j, V_j from HBM to on-chip SRAM
    k_mask = kv_offset < S
    K_T = tl.load(k_ptr + k_t_offsets, mask=k_mask[None, :], other=0.0)
    V_T = tl.load(v_ptr + v_t_offsets, mask=k_mask[None, :], other=0.0)

    # initialize dK_j = 0, dV_j = 0 on SRAM
    dk = tl.zeros([BLOCK_KV, Dh], dtype=tl.float32)
    dv = tl.zeros([BLOCK_KV, Dh], dtype=tl.float32)

    # start index for causal masking, Q blocks before the current KV block 
    start_q_idx = kv_idx * BLOCK_KV
    lo = tl.multiple_of(start_q_idx, BLOCK_Q)

    q_ptr = Q + offset_bh
    do_ptr = dO + offset_bh
    dq_ptr = dQ + offset_bh
    
    lse_ptr = LSE + (batch_idx * H + head_idx) * S
    delta_ptr = Delta + (batch_idx * H + head_idx) * S

    ln2 = 1.4426950408889634

    # for 1 <= i <= Tr do
    for q_offset_base in range(lo, S, BLOCK_Q):
        q_offset_base = tl.multiple_of(q_offset_base, BLOCK_Q)
        q_offset = q_offset_base + tl.arange(0, BLOCK_Q)
        q_mask = q_offset < S

        q_offsets = q_offset[:, None] * Dh + dh_offset[None, :]
        
        # load Q_i, O_i, dO_i, dQ_i, l_i, m_i from HBM to SRAM
        q = tl.load(q_ptr + q_offsets, mask=q_mask[:, None], other=0.0)
        do = tl.load(do_ptr + q_offsets, mask=q_mask[:, None], other=0.0)
        lse = tl.load(lse_ptr + q_offset, mask=q_mask, other=0.0)
        delta = tl.load(delta_ptr + q_offset, mask=q_mask, other=0.0)

        # on chip, compute S_ij = tau * Q_i * K_j^T
        s_ij = tl.dot(q, K_T) * sm_scale * ln2

        # on chip, compute S_ij_masked = mask(S_ij)
        causal_mask = q_offset[:, None] >= kv_offset[None, :]
        s_ij = tl.where(causal_mask, s_ij, -1.0e6)

        # on chip, compute P_ij = diag(l_i)^-1 exp(S_ij - m_i)
        p_ij = tl.exp2(s_ij - lse[:, None])
        p_ij = tl.where(causal_mask, p_ij, 0.0)

        # on chip, compute dV_j <- dV_j + (P_ij)^T * dO_i
        dv += tl.dot(tl.trans(p_ij.to(q.dtype)), do)

        # on chip, compute dP_ij = dO_i * (V_j)^T
        dp_ij = tl.dot(do, V_T)

        # on chip, compute D_i = rowsum(dO_i * O_i)
        # on chip, compute dS_ij = P_ij * (dP_ij - D_i)
        ds_ij = p_ij * (dp_ij - delta[:, None])

        # on chip, compute dK_j <- dK_j + tau * (dS_ij)^T * Q_i
        dk += tl.dot(tl.trans(ds_ij.to(q.dtype)), q) * sm_scale

        # write dQ_i <- dQ_i + tau * dS_ij * K_j to HBM
        # since multiple KV loop blocks write to the same Q indices, 
        # we must use tl.atomic_add to safely accumulate in HBM.
        dq = tl.dot(ds_ij.to(q.dtype), tl.trans(K_T)) * sm_scale
        tl.atomic_add(dq_ptr + q_offsets, dq, mask=q_mask[:, None])

    # write dK_j <- dK_j, dV_j <- dV_j to HBM
    tl.store(dk_ptr + k_offsets, dk.to(K_T.dtype), mask=k_mask[:, None])
    tl.store(dv_ptr + v_offsets, dv.to(K_T.dtype), mask=k_mask[:, None])

    #FYI (for backprop sanity check)
    #64 x 64 = 4096 elements for 4 of the following: K_T block, V_T, K_blk, Q_T
    #fp 32: 4 bytes
    #~33 tiles × 4096 × 4 bytes ≈ 132KB

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
    return O, LSE

def back_prop_attention_compute(do, q, k, v, O, LSE, sm_scale):

    #FYI: using default fp32 for BP results in resources error, even when reducing num_stages buffer from 3 --> 1
    #triton.runtime.errors.OutOfResources: out of resource: shared memory (bytes), 
    #Required: 114688, Hardware limit: 101376. Reducing block sizes or `num_stages` may help.
    '''do = do.contiguous()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()'''
    f16 = torch.float16
    q_k = q.to(f16).contiguous()
    k_k = k.to(f16).contiguous()
    v_k = v.to(f16).contiguous()
    O_k = O.to(f16).contiguous()
    do_k = do.to(f16).contiguous()

    B, H, S, Dh = q.shape

    # dQ is initialized to zero because of the atomic_adds in the kernel
    dQ = torch.zeros_like(q)
    # dK and dV are fully overwritten so empty is fine
    dK = torch.empty_like(k)
    dV = torch.empty_like(v)

    # precomputing D_i = rowsum(dO_i * O_i)
    Delta = torch.sum(do * O, dim=-1).contiguous()

    BLOCK_Q = 64 if S >= 64 else triton.next_power_of_2(S)
    BLOCK_KV = BLOCK_Q

    grid = (triton.cdiv(S, BLOCK_KV), B * H)
    
    compiled = backpropt_attention_propagate.warmup(
        q, k, v, O, do, dQ, dK, dV, LSE, Delta,
        sm_scale, B, H, S, Dh,
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV,
        grid=(triton.cdiv(S, BLOCK_KV), B * H),
    )
    #print(f"num_stages used: {compiled.metadata.num_stages}")
    #print(f"shared memory:   {compiled.metadata.shared} bytes")
    '''
    memory debug analysis
    num_stages used: 3
    shared memory:   132096 bytes
    '''

    #memory (shared memory:   132096 bytes)
    #default num_stagers is 2 or 3, reduce to 1 if need memory
    #num_stages and warps are compiler opt. flags
    backpropt_attention_propagate[grid](
        q_k, k_k, v_k, O_k, do_k,
        dQ, dK, dV,
        LSE, Delta,
        sm_scale,
        B, H, S, Dh,
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV,
        num_stages=1,
        num_warps=4,
    )
    
    return dQ, dK, dV

#basic sanity test for FW pass
def test_flash_attention_forward(B=2, H=4, S=128, Dh=64, device=None, atol=5e-2, rtol=1e-2):
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
    triton_out, _ = forward_attention_compute(q, k, v, sm_scale)
    pytorch_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sm_scale)
    #scaled dot_product_attention uses different backends (causes diff performance???)
    #weird compiler

    torch.testing.assert_close(triton_out, pytorch_out, atol=atol, rtol=rtol)
    print(f"passed fwd: B={B}, H={H}, S={S}, Dh={Dh}, dtype={q.dtype}")

#basic sanity test for BW pass
def test_flash_attention_backward(B=2, H=4, S=128, Dh=64, device=None):
    if device is None:
        device = torch.device("cuda")
        
    torch.manual_seed(0)
    sm_scale = 1.0 / math.sqrt(Dh)
    
    # Create tensors and set requires_grad=True so PyTorch natively tracks them
    q_pt = torch.randn((B, H, S, Dh), dtype=torch.float32, device=device, requires_grad=True)
    k_pt = torch.randn((B, H, S, Dh), dtype=torch.float32, device=device, requires_grad=True)
    v_pt = torch.randn((B, H, S, Dh), dtype=torch.float32, device=device, requires_grad=True)
    
    # Clone tensors for Triton. We don't need requires_grad=True here because we are 
    # doing the math manually!
    q_tr = q_pt.detach().clone()
    k_tr = k_pt.detach().clone()
    v_tr = v_pt.detach().clone()

    # PyTorch Native Forward & Backward
    out_pt = F.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=True, scale=sm_scale)
    gO = torch.rand_like(out_pt) # Fake upstream gradient
    out_pt.backward(gO)

    # Triton Manual Forward & Backward
    # 1. Run forward to get O and LSE
    O_tr, LSE_tr = forward_attention_compute(q_tr, k_tr, v_tr, sm_scale)
    # 2. Run backward using the fake gradient (gO) and the saved tensors
    dQ_tr, dK_tr, dV_tr = back_prop_attention_compute(gO, q_tr, k_tr, v_tr, O_tr, LSE_tr, sm_scale)

    # Compare gradients natively
    torch.testing.assert_close(dQ_tr, q_pt.grad, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(dK_tr, k_pt.grad, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(dV_tr, v_pt.grad, atol=5e-2, rtol=1e-2)
    print("Passed backward pass gradients!")

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

'''
NOTE: for values greater than 130172
torch.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 21.95 GiB of which 730.12 MiB is free. 
Including non-PyTorch memory, this process has 21.23 GiB memory in use. 
Of the allocated memory 20.98 GiB is allocated by PyTorch, and 45.09 MiB is reserved by PyTorch but unallocated. 
If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to 
avoid fragmentation.  See documentation for Memory Management  
(https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
'''
configs = []
for mode in ["Forward Pass","Back Prop"]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["Sequence_Length"],
            x_vals=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32678, 65356, 130712],
            x_log=True,
            line_arg="provider",
            line_vals=["triton", "sdpa"],
            line_names=["TritonKernel", "TorchSDPA"],
            ylabel="Latency (ms)",
            plot_name=f"{mode} Latency Benchmark",
            args={"mode": mode},
        )
    )
@triton.testing.perf_report(configs)
def benchmark_flash_attention_kernel(mode, Sequence_Length, provider, B=2, H=8, Dh=64, dtype=tl.float32):
    #bug with testing fp16 or other mismatching floating points, changing above to f32 produces bug
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarking.")
    device = torch.device("cuda")
    torch.manual_seed(0)
    assert mode in ["Forward Pass","Back Prop"]
    dtype=_normalize_dtype(dtype)
    q = torch.randn((B, H, Sequence_Length, Dh), dtype=dtype, device=device)
    k = torch.randn((B, H, Sequence_Length, Dh), dtype=dtype, device=device)
    v = torch.randn((B, H, Sequence_Length, Dh), dtype=dtype, device=device)
    sm_scale = 1.0 / math.sqrt(Dh)

    #forward mode
    if mode == "Forward Pass":
        if provider == "triton":
            fn = lambda: forward_attention_compute(q, k, v, sm_scale)[0]
        else:
            fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sm_scale)
    #backward mode
    else:
        O, LSE = forward_attention_compute(q, k, v, sm_scale)
        gO = torch.rand_like(q)

        #tensors are not tracked in Torch by default, cloning does this
        #other option is define tensor with requires_grad=True, but it messes up forward
        q_pt = q.clone().requires_grad_(True)
        k_pt = k.clone().requires_grad_(True)
        v_pt = v.clone().requires_grad_(True)

        if provider == "triton":
            fn = lambda: back_prop_attention_compute(gO, q, k, v, O, LSE, sm_scale)
        else:
            def backward_pass_fn():
                out = F.scaled_dot_product_attention(q_pt, k_pt, v_pt, is_causal=True, scale=sm_scale)
                out.backward(gO)
                #avoid accumulating zero values to exclude overhead of zeroing kernel
                q_pt.grad = None
                k_pt.grad = None
                v_pt.grad = None
            fn = backward_pass_fn

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    torch_dtype = _normalize_dtype(dtype)
    Benchmark.report_metrics(provider, mode, B, H, Sequence_Length, Dh, torch_dtype, ms)
    return ms

if __name__ == "__main__":
    #NOTE: this implementation is based on Flash Attention Original and Version 2
    #NOTE: this Benchmark is for standalone Flash Attention benchmarking
    from benchmark_metrics import Benchmark

    #Sanity test for flash_attention_kernel forward pass
    test_flash_attention_forward()
    #Sanity test for flash_attention_kernel backwards pass
    test_flash_attention_backward()
    #bench mark both
    benchmark_flash_attention_kernel.run(show_plots=True, print_data=True, save_path="../output/plots")
    Benchmark.plot_benchmark_results()
    