"""Versioned PRIVATE graph references. Never export this file to kernel agents.

The historical references remain byte-for-byte untouched in benchmarks/*.
GDN v2 preserves the native solver behind a truthful opaque layout bridge;
BSA v1 tensorizes runtime metadata. Neither specializes on tensor values.
"""
from functools import lru_cache
import importlib.util
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


@lru_cache(None)
def original(task):
    path = Path(__file__).resolve().parents[1] / 'benchmarks' / task / 'baseline.py'
    spec = importlib.util.spec_from_file_location(f'private_original_{task}', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def matmul_model():
    return original('matmul').Model()


@torch.library.custom_op('private_campaign_reference::unit_lower_solve', mutates_args=())
def unit_lower_solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Preserve native solver math, but advertise its actual contiguous output.
    return torch.linalg.solve_triangular(a, b, upper=False, unitriangular=True).contiguous()


@unit_lower_solve.register_fake
def _unit_lower_solve_fake(a, b):
    return torch.empty_like(b, memory_format=torch.contiguous_format)


class GDNGraphV2(original('gdn').Model):
    def forward(self, q, k, v, g, beta, A_log, dt_bias, initial_state,
                scale, chunk_size, use_gate_in_kernel, use_bias,
                use_initial_state, output_final_state, use_qk_l2norm,
                use_beta_sigmoid, allow_neg_eigval, state_v_first):
        # Same computation as original forward; only torch.manual_seed(42) omitted.
        batch, _, heads, key_dim = q.shape
        value_heads, value_dim = v.shape[2:]
        if value_heads % heads:
            raise ValueError('value heads must be divisible by query heads')
        query, key = q.float(), k.float()
        if use_qk_l2norm:
            query = F.normalize(query, p=2.0, dim=-1, eps=1e-6)
            key = F.normalize(key, p=2.0, dim=-1, eps=1e-6)
        query = query.repeat_interleave(value_heads // heads, dim=2)
        key = key.repeat_interleave(value_heads // heads, dim=2)
        if use_gate_in_kernel:
            gate_input = g.float()
            if use_bias:
                gate_input = gate_input + dt_bias.float()
            gate = -torch.exp(A_log.float()) * F.softplus(gate_input)
        else:
            gate = g.float()
        beta_value = beta.float()
        if use_beta_sigmoid:
            beta_value = torch.sigmoid(beta_value)
            if allow_neg_eigval:
                beta_value = beta_value * 2.0
        if use_initial_state:
            state = initial_state.float().clone()
            if state_v_first:
                state = state.transpose(-2, -1).contiguous()
        else:
            state = torch.zeros(batch, value_heads, key_dim, value_dim,
                                device=q.device, dtype=torch.float32)
        output = self._chunk_kda(query, key, v, gate, beta_value,
                                 scale, state, chunk_size)
        if state_v_first:
            state = state.transpose(-2, -1).contiguous()
        return output.to(v.dtype), state if output_final_state else None

    def _chunk_kda(self, query, key, v, gate, beta, scale, state, C):
        # Original chunk algorithm, replacing only two solver calls with the
        # opaque native solver layout bridge. No mathematical substitution.
        B, T, HV, K = query.shape
        V, dev = v.shape[-1], query.device
        C = int(C)
        Tp = ((T + C - 1) // C) * C
        pad = Tp - T
        qf, kf = query.transpose(1, 2).float(), key.transpose(1, 2).float()
        vf, gf, bf = v.transpose(1, 2).float(), gate.transpose(1, 2).float(), beta.transpose(1, 2).float()
        per_k = gf.dim() == 4
        qp, kp, vp = (F.pad(tensor, (0, 0, 0, pad)) for tensor in (qf, kf, vf))
        gp = F.pad(gf, (0, 0, 0, pad)) if per_k else F.pad(gf, (0, pad))
        bp = F.pad(bf, (0, pad))
        S = state.float().contiguous()
        o = torch.zeros(B, HV, Tp, V, device=dev, dtype=torch.float32)
        eye = torch.eye(C, device=dev, dtype=torch.float32).expand(B, HV, C, C).clone()
        for c in range(Tp // C):
            qc, kc, vc = (tensor[:, :, c * C:(c + 1) * C] for tensor in (qp, kp, vp))
            gc, bc = gp[:, :, c * C:(c + 1) * C], bp[:, :, c * C:(c + 1) * C]
            gk = gc.cumsum(2) / math.log(2.)
            if per_k:
                gk_last = gk[:, :, -1:, :]
                decay = torch.exp2(torch.clamp(gk.unsqueeze(3) - gk.unsqueeze(2), max=0.))
                Aqk = torch.tril(torch.einsum('bhik,bhjk,bhijk->bhij', qc, kc, decay)) * scale
                L = torch.tril(torch.einsum('bhik,bhjk,bhijk->bhij', kc, kc, decay)
                               * bc.unsqueeze(-1), diagonal=-1)
                gl = torch.exp2(gk_last).transpose(-1, -2)
            else:
                gk_last = gk[:, :, -1:]
                decay = torch.exp2(torch.clamp(gk.unsqueeze(-1) - gk.unsqueeze(2), max=0.))
                Aqk = torch.tril(torch.einsum('bhik,bhjk,bhij->bhij', qc, kc, decay)) * scale
                L = torch.tril(torch.einsum('bhik,bhjk,bhij->bhij', kc, kc, decay)
                               * bc.unsqueeze(-1), diagonal=-1)
                gl = torch.exp2(gk_last).unsqueeze(-1)
            gk4 = torch.exp2(gk) if per_k else torch.exp2(gk).unsqueeze(-1)
            w = unit_lower_solve(eye + L, bc.unsqueeze(-1) * kc * gk4)
            u = unit_lower_solve(eye + L, bc.unsqueeze(-1) * vc)
            v_new = u - w @ S
            o[:, :, c * C:(c + 1) * C] = (torch.einsum('bhik,bhkv->bhiv', qc * gk4, S) * scale
                                         + torch.einsum('bhij,bhjv->bhiv', Aqk, v_new))
            kg = kc * (torch.exp2(gk_last - gk) if per_k
                       else torch.exp2(gk_last - gk).unsqueeze(-1))
            S = gl * S + torch.einsum('bhik,bhiv->bhkv', kg, v_new)
        state.copy_(S)
        return o.transpose(1, 2).contiguous()[:, :T]


class BSAGraphV1(torch.nn.Module):
    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type,
                streaming_info, base_blockmask, softmax_scale,
                is_causal, exact_streaming):
        # Packed all-sequence attention; masks remove cross-sequence pairs.
        tq, heads, dim = q.shape
        tk, kvheads, _ = k.shape
        groups = heads // kvheads
        qi = torch.arange(tq, device=q.device)
        ki = torch.arange(tk, device=q.device)
        qb = (qi[:, None] >= cu_seqlens_q[None, 1:]).sum(-1)
        kb = (ki[:, None] >= cu_seqlens_k[None, 1:]).sum(-1)
        qr = qi - cu_seqlens_q[qb]
        kr = ki - cu_seqlens_k[kb]
        sq = cu_seqlens_q[qb + 1] - cu_seqlens_q[qb]
        sk = cu_seqlens_k[qb + 1] - cu_seqlens_k[qb]
        row, col = qr[None, :, None], kr[None, None, :]
        sq, sk = sq[None, :, None], sk[None, :, None]
        causal_keep = col <= row + sk - sq
        types = head_mask_type[:, None, None]
        stream = streaming_info.reshape(heads, 2)
        sink, local = stream[:, 0, None, None], stream[:, 1, None, None]
        if exact_streaming:
            assert is_causal, 'exact streaming requires causal attention'
            stream_keep = causal_keep & ~((col < row + sk - sq - (local - 1))
                                         & (col >= sink))
        else:
            ncol = (sk + 127) // 128
            start = ((sq - sk).clamp(min=0)) // 128
            mr = ((sk - sq).clamp(min=0) + 127) // 128 + 1 + row // 128 - start
            if not is_causal:
                mr = ncol
            win = ((col // 128 >= (mr - local).clamp(min=0))
                   & (col // 128 < torch.minimum(mr, ncol)))
            stream_keep = win | (col // 128 < sink)
            if is_causal:
                stream_keep = stream_keep & (row // 128 >= start)
        # Padding permits zero sparse heads without a data-dependent Python branch.
        # Sparse rank follows current runtime head order, not a fixture pattern.
        pad = torch.ones_like(base_blockmask[:, :1])
        if base_blockmask.shape[1] == 0:
            pad = torch.ones((base_blockmask.shape[0], 1, *base_blockmask.shape[2:]),
                             dtype=torch.bool, device=q.device)
        masks = torch.cat((pad, base_blockmask), dim=1)
        rank = (head_mask_type == 1).to(torch.int64).cumsum(0)
        rank = torch.where(head_mask_type == 1, rank, 0)
        sparse_keep = masks[qb[None, :, None], rank[:, None, None],
                            row // 128, col // 128]
        keep = torch.where(types == 1, sparse_keep,
                           torch.where(types == -1, stream_keep, True))
        keep = keep & (qb[None, :, None] == kb[None, None, :])
        if is_causal:
            keep = keep & causal_keep
        key = k.float()[:, :, None, :].expand(tk, kvheads, groups, dim).reshape(tk, heads, dim)
        val = v.float()[:, :, None, :].expand(tk, kvheads, groups, dim).reshape(tk, heads, dim)
        scale = dim ** -.5 if softmax_scale is None else softmax_scale
        scores = torch.einsum('thd,shd->hts', q.float() * scale, key)
        weights = torch.softmax(scores.masked_fill(~keep, float('-inf')), dim=-1)
        weights = weights.masked_fill(~keep, 0.)
        return torch.einsum('hts,shd->thd', weights, val).to(q.dtype)


def gdn_model():
    return GDNGraphV2()


def bsa_model():
    return BSAGraphV1()
