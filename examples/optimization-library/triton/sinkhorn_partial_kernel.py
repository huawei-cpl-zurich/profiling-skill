# Adapted from MindSpeedOps sinkhorn.py deterministic_reduce_kernel, lines 288–326.
# Source pin 4efe2b675c2d51517c3fdd14c721020b8d41e241; Apache-2.0.
# Component only: supplied partials, source grid (1,), hc_mult=4.
import triton
import triton.language as tl


@triton.jit
def deterministic_reduce_kernel(PS, PB, S, B, N: tl.constexpr,
                                VECTOR: tl.constexpr, BLOCK: tl.constexpr):
    if VECTOR:
        rows = tl.arange(0, BLOCK)
        c2 = tl.arange(0, 2)
        c8 = tl.arange(0, 8)
        s = tl.load(PS + rows[:, None]*2 + c2[None, :], rows[:, None] < N, 0)
        b = tl.load(PB + rows[:, None]*8 + c8[None, :], rows[:, None] < N, 0)
        tl.store(S+c2, tl.sum(s, axis=0))
        tl.store(B+c8, tl.sum(b, axis=0))
    else:
        acc0 = tl.zeros((), tl.float32)
        acc1 = tl.zeros((), tl.float32)
        for i in range(N):
            acc0 += tl.load(PS+i*2)
            acc1 += tl.load(PS+i*2+1)
        tl.store(S, acc0)
        tl.store(S+1, acc1)
        for j in range(4):
            pre = tl.zeros((), tl.float32)
            post = tl.zeros((), tl.float32)
            for i in range(N):
                pre += tl.load(PB+i*8+j)
                post += tl.load(PB+i*8+4+j)
            tl.store(B+j, pre)
            tl.store(B+4+j, post)
