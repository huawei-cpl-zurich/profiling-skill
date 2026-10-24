// Source-backed standalone adaptation of required scalar padding repair and
// aligned count-form Duplicate; not a full layer-normalization benchmark.
#include "kernel_operator.h"
#include <stdint.h>
using namespace AscendC;
#if W006_N05_AFTER
#define W006_N05_ENTRY wave006_aligned_padding_after
#else
#define W006_N05_ENTRY wave006_aligned_padding_before
#endif
extern "C" __global__ __aicore__ __attribute__((aiv)) void W006_N05_ENTRY(GM_ADDR x, GM_ADDR y) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, W006_N05_ROWS * W006_N05_PITCH * sizeof(float));
    auto local = buffer.Get<float>();
    GlobalTensor<float> input, output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), W006_N05_ROWS * W006_N05_PITCH);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), W006_N05_ROWS * W006_N05_PITCH);
    for (uint32_t row = 0; row < W006_N05_ROWS; ++row)
        DataCopy(local[row * W006_N05_PITCH], input[row * W006_N05_PITCH], W006_N05_PITCH);
    // Same conservative completion/ownership barriers in both arms, including
    // the scalar fallback: no asynchronous copy can race a scalar UB access.
    PipeBarrier<PIPE_ALL>();
    for (uint32_t row = 0; row < W006_N05_ROWS; ++row) {
#if W006_N05_AFTER && W006_N05_WIDTH % 8 == 0 && W006_N05_PITCH > W006_N05_WIDTH
        Duplicate(local[row * W006_N05_PITCH + W006_N05_WIDTH], 0.0f, W006_N05_PITCH - W006_N05_WIDTH);
#else
        for (uint32_t col = W006_N05_WIDTH; col < W006_N05_PITCH; ++col)
            local.SetValue(row * W006_N05_PITCH + col, 0.0f);
#endif
    }
    PipeBarrier<PIPE_ALL>();
    for (uint32_t row = 0; row < W006_N05_ROWS; ++row)
        DataCopy(output[row * W006_N05_PITCH], local[row * W006_N05_PITCH], W006_N05_PITCH);
    PipeBarrier<PIPE_ALL>();
}
extern "C" uint32_t aligned_padding_rows() { return W006_N05_ROWS; }
extern "C" uint32_t aligned_padding_width() { return W006_N05_WIDTH; }
extern "C" uint32_t aligned_padding_pitch() { return W006_N05_PITCH; }
extern "C" uint32_t aligned_padding_after() { return W006_N05_AFTER; }
extern "C" uint32_t catlass_bridge_abi_version() { return 1; }
#if W006_N05_AFTER
extern "C" const char* catlass_bridge_kernel_names() { return "[\"wave006_aligned_padding_after\"]"; }
#else
extern "C" const char* catlass_bridge_kernel_names() { return "[\"wave006_aligned_padding_before\"]"; }
#endif
extern "C" int catlass_bridge_launch(void** pointers, uint64_t count, void* stream) {
    if (count != 2 || !pointers[0] || !pointers[1]) return 1;
    W006_N05_ENTRY<<<1, nullptr, stream>>>((uint8_t*)pointers[0], (uint8_t*)pointers[1]);
    return 0;
}
