// Independent tiny adaptation, not the upstream LN operator.
// Successive scratch mechanism: ops-rec e489c0bf6a2d5f2d05b3205fcacc01b62a908b10,
// ln_mul_kernel.h:224-278, Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
// TBuf/InitBuffer/Get API pattern: CATLASS 40aced927fdaaa0f3eaa9c6f5c5d859e06392f55,
// local_tensor_buffer.hpp:209-221, CANN Open Software License Agreement Version 2.0.
#include "kernel_operator.h"
#include <stdint.h>
using namespace AscendC;
#if AFTER
#define ENTRY wave006_scratch_reuse_after
#else
#define ENTRY wave006_scratch_reuse_before
#endif
extern "C" __global__ __aicore__ __attribute__((aiv)) void ENTRY(GM_ADDR x, GM_ADDR y) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> xBuffer, yBuffer, scratchA;
    pipe.InitBuffer(xBuffer, TILE * sizeof(float));
    pipe.InitBuffer(yBuffer, TILE * sizeof(float));
    pipe.InitBuffer(scratchA, TILE * sizeof(float));
    auto localX = xBuffer.Get<float>();
    auto localY = yBuffer.Get<float>();
    auto a = scratchA.Get<float>();
#if !AFTER || LATE
    TBuf<TPosition::VECCALC> scratchB;
    pipe.InitBuffer(scratchB, TILE * sizeof(float));
    auto b = scratchB.Get<float>();
#else
    auto b = a; // One owned tensor; A's final read completes before B's write.
#endif
    GlobalTensor<float> input, output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), ROWS * WIDTH);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), ROWS * WIDTH);
    for (uint32_t offset = 0; offset < ROWS * WIDTH; offset += TILE) {
        uint32_t count = ROWS * WIDTH - offset;
        if (count > TILE) count = TILE;
        DataCopyExtParams transfer{1, count * uint32_t(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(localX, input[offset], transfer, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Mul(a, localX, localX, count);
        PipeBarrier<PIPE_V>();
        Add(localY, a, localX, count);
        PipeBarrier<PIPE_V>();
        Muls(b, localY, 0.5f, count);
        PipeBarrier<PIPE_V>();
#if LATE
        Add(localY, b, a, count);
#else
        Adds(localY, b, -0.25f, count);
#endif
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(output[offset], localY, transfer);
        PipeBarrier<PIPE_ALL>(); // Complete copyout before reusing X/Y on the next tile.
    }
}
extern "C" uint32_t scratch_reuse_rows() { return ROWS; }
extern "C" uint32_t scratch_reuse_width() { return WIDTH; }
extern "C" uint32_t scratch_reuse_tile() { return TILE; }
extern "C" uint32_t scratch_reuse_late() { return LATE; }
extern "C" uint32_t scratch_reuse_after() { return AFTER; }
extern "C" uint32_t catlass_bridge_abi_version() { return 1; }
#if AFTER
extern "C" const char* catlass_bridge_kernel_names() { return "[\"wave006_scratch_reuse_after\"]"; }
#else
extern "C" const char* catlass_bridge_kernel_names() { return "[\"wave006_scratch_reuse_before\"]"; }
#endif
extern "C" int catlass_bridge_launch(void** pointers, uint64_t count, void* stream) {
    if (count != 2 || !pointers[0] || !pointers[1]) return 1;
    ENTRY<<<1, nullptr, stream>>>((uint8_t*)pointers[0], (uint8_t*)pointers[1]);
    return 0;
}
