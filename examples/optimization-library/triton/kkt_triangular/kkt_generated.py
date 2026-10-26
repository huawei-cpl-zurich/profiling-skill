"""Generated source-derived featurewise KKT; see frozen P04 contract."""
import triton
import triton.language as tl

@triton.jit
def kkt_zero(A,N:tl.constexpr,BLOCK:tl.constexpr):
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    tl.store(A+x,0.,x<N)


@triton.jit(do_not_specialize=['T'])
def kkt_full_inter(k, g, beta, A, cu_seqlens, chunk_indices, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_c, i_b = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_i, i_j = (i_c // NC, i_c % NC)
    for i_h in range(H):
        if IS_VARLEN:
            i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
            bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
            T_val = eos - bos
        else:
            bos, eos = (i_b * T, i_b * T + T)
            T_val = T
        should_compute = i_t * BT + i_i * BC < T_val and i_i != i_j
        if should_compute:
            k_ptr = k + (bos * H + i_h) * K
            g_ptr = g + (bos * H + i_h) * K
            A_ptr = A + (bos * H + i_h) * BT
            p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T_val,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,))
            b_beta = tl.load(p_beta, boundary_check=(0,))
            b_A = tl.zeros([BC, BC], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_k = tl.make_block_ptr(k_ptr, (T_val, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
                p_g = tl.make_block_ptr(g_ptr, (T_val, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
                b_kt = tl.make_block_ptr(k_ptr, (K, T_val), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
                p_gk = tl.make_block_ptr(g_ptr, (K, T_val), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
                o_k = i_k * BK + tl.arange(0, BK)
                m_k = o_k < K
                b_gn = tl.load(g_ptr + (i_t * BT + i_i * BC) * H * K + o_k, mask=m_k, other=0)
                b_g = tl.load(p_g, boundary_check=(0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1)) * tl.exp(b_g - b_gn[None, :])
                b_gk = tl.load(p_gk, boundary_check=(0, 1))
                b_kt = tl.load(b_kt, boundary_check=(0, 1)) * tl.exp(b_gn[:, None] - b_gk)
                b_A += tl.dot(b_k, b_kt, allow_tf32=False)
            b_A *= b_beta[:, None]
            b_A = tl.where(i_i > i_j, b_A, 0.0)
            p_A = tl.make_block_ptr(A_ptr, (T_val, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))

@triton.jit(do_not_specialize=['T'])
def kkt_lower_inter(k, g, beta, A, cu_seqlens, chunk_indices, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_c, i_b = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_i, i_j = (i_c // NC, i_c % NC)
    for i_h in range(H):
        if IS_VARLEN:
            i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
            bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
            T_val = eos - bos
        else:
            bos, eos = (i_b * T, i_b * T + T)
            T_val = T
        should_compute = i_t * BT + i_i * BC < T_val and i_i > i_j
        if should_compute:
            k_ptr = k + (bos * H + i_h) * K
            g_ptr = g + (bos * H + i_h) * K
            A_ptr = A + (bos * H + i_h) * BT
            p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T_val,), (H,), (i_t * BT + i_i * BC,), (BC,), (0,))
            b_beta = tl.load(p_beta, boundary_check=(0,))
            b_A = tl.zeros([BC, BC], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_k = tl.make_block_ptr(k_ptr, (T_val, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
                p_g = tl.make_block_ptr(g_ptr, (T_val, K), (H * K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
                b_kt = tl.make_block_ptr(k_ptr, (K, T_val), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
                p_gk = tl.make_block_ptr(g_ptr, (K, T_val), (1, H * K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
                o_k = i_k * BK + tl.arange(0, BK)
                m_k = o_k < K
                b_gn = tl.load(g_ptr + (i_t * BT + i_i * BC) * H * K + o_k, mask=m_k, other=0)
                b_g = tl.load(p_g, boundary_check=(0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1)) * tl.exp(b_g - b_gn[None, :])
                b_gk = tl.load(p_gk, boundary_check=(0, 1))
                b_kt = tl.load(b_kt, boundary_check=(0, 1)) * tl.exp(b_gn[:, None] - b_gk)
                b_A += tl.dot(b_k, b_kt, allow_tf32=False)
            b_A *= b_beta[:, None]
            p_A = tl.make_block_ptr(A_ptr, (T_val, BT), (H * BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))

@triton.jit(do_not_specialize=['T'])
def kkt_diagonal(k, g, beta, A, cu_seqlens, chunk_indices, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_i, i_b = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    for i_h in range(H):
        if IS_VARLEN:
            i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
            bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32))
            T_val = eos - bos
        else:
            bos, eos = (i_b * T, i_b * T + T)
            T_val = T
        should_compute = i_t * BT + i_i * BC < T_val
        if should_compute:
            o_i = tl.arange(0, BC)
            o_k = tl.arange(0, BK)
            m_k = o_k < K
            m_A = i_t * BT + i_i * BC + o_i < T_val
            o_A = (bos + i_t * BT + i_i * BC + o_i) * H * BT + i_h * BT + i_i * BC
            p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T_val, K), (H * K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
            p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T_val, K), (H * K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
            p_beta = beta + (bos + i_t * BT + i_i * BC + o_i) * H + i_h
            b_k = tl.load(p_k, boundary_check=(0, 1)) * tl.load(p_beta, mask=m_A, other=0)[:, None]
            b_g = tl.load(p_g, boundary_check=(0, 1))
            p_kt = k + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
            p_gk = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
            for j in range(0, min(BC, T_val - i_t * BT - i_i * BC)):
                b_kt = tl.load(p_kt, mask=m_k, other=0).to(tl.float32)
                b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
                b_A = tl.sum(b_k * b_kt[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
                o_i_tmp = o_i.to(tl.float32)
                b_A = tl.where(o_i_tmp > j, b_A, 0.0)
                tl.store(A + o_A + j, b_A, mask=m_A)
                p_kt += H * K
                p_gk += H * K
