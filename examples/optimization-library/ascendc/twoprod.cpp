// Independently extracted dense FP32 dot, not an upstream SpMV benchmark.
#include "twoprod_helpers.h"
#include <stdint.h>
using namespace AscendC;
static_assert(WIDTH > 0 && WIDTH <= 4096);
constexpr uint32_t pitch = ((WIDTH + 7) / 8) * 8;
#if AFTER
#define ENTRY wave006_twoprod_after
#else
#define ENTRY wave006_twoprod_before
#endif
// Same sequential short-row accumulation and DS merge/finalization in both arms.
template<bool GUARDED>
__aicore__ inline float Dot(const LocalTensor<float>& a, const LocalTensor<float>& b,
                           const LocalTensor<float>& accum) {
    float s = 0.0f, c = 0.0f;
    for (int32_t j = 0; j < WIDTH; ++j) {
        float p, pErr;
        SpmvTwoProdSel<GUARDED>(a.GetValue(j), b.GetValue(j), p, pErr);
        float t = s + p;
        float bp = t - s;
        c += (s - (t - bp)) + (p - bp);
        c += pErr;
        s = t;
    }
    accum.SetValue(0, 0.0f); accum.SetValue(8, 0.0f);
    auto ptr = reinterpret_cast<__ubuf__ float*>(accum.GetPhyAddr());
    SpmvDsMerge(ptr, ptr + 8, 0, s, c);
    return ptr[0] + ptr[8];
}
extern "C" __global__ __aicore__ __attribute__((aiv)) void ENTRY(GM_ADDR a, GM_ADDR b, GM_ADDR y) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufA, bufB, bufWk, bufOut;
    pipe.InitBuffer(bufA, pitch * sizeof(float));
    pipe.InitBuffer(bufB, pitch * sizeof(float));
    pipe.InitBuffer(bufWk, 256 * sizeof(float));
    pipe.InitBuffer(bufOut, 16 * sizeof(float));
    auto la = bufA.Get<float>(), lb = bufB.Get<float>();
    auto wk = bufWk.Get<float>(), out = bufOut.Get<float>();
    GlobalTensor<float> ga, gb, gy;
    ga.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), ROWS * WIDTH);
    gb.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b), ROWS * WIDTH);
    // [dot, actual GPU fast flag] per row. Publication work is identical.
    gy.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), ROWS * 2);
    DataCopyExtParams input{1, WIDTH * sizeof(float), 0, 0, 0};
    DataCopyPadExtParams<float> padding{false, 0, 0, 0.0f};
    DataCopyExtParams output{1, 2 * sizeof(float), 0, 0, 0};
    for (uint32_t row = 0; row < ROWS; ++row) {
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(la, ga[row * WIDTH], input, padding);
        DataCopyPad(lb, gb[row * WIDTH], input, padding);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0); WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        bool safe = false;
#if AFTER
        float maxB = SpmvVecMaxAbs(lb, WIDTH, wk);
        float rowLimit;
        safe = SpmvTileSafeN(la, WIDTH, maxB, wk, rowLimit);
#endif
        float answer = safe ? Dot<false>(la, lb, out) : Dot<true>(la, lb, out);
        out.SetValue(0, answer); out.SetValue(1, safe ? 1.0f : 0.0f);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0); WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(gy[row * 2], out, output);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0); WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
    PipeBarrier<PIPE_ALL>();
}
extern "C" uint32_t twoprod_rows() { return ROWS; }
extern "C" uint32_t twoprod_width() { return WIDTH; }
extern "C" uint32_t twoprod_after() { return AFTER; }
extern "C" uint32_t catlass_bridge_abi_version() { return 1; }
#if AFTER
extern "C" const char* catlass_bridge_kernel_names() { return "[\"wave006_twoprod_after\"]"; }
#else
extern "C" const char* catlass_bridge_kernel_names() { return "[\"wave006_twoprod_before\"]"; }
#endif
extern "C" int catlass_bridge_launch(void** pointers, uint64_t count, void* stream) {
    if (count != 3 || !pointers[0] || !pointers[1] || !pointers[2]) return 1;
    ENTRY<<<1, nullptr, stream>>>((uint8_t*)pointers[0], (uint8_t*)pointers[1], (uint8_t*)pointers[2]);
    return 0;
}
