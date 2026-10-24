// Native A2/A3 study: change only the number of GM-to-UB copy commands.
#include "kernel_operator.h"
#include <stdint.h>
using namespace AscendC;
constexpr uint32_t pitch = ((COLUMNS + 7) / 8) * 8;
constexpr uint32_t elements = ROWS * pitch;
#if AFTER
#define ENTRY wave006_row_dma_after
#else
#define ENTRY wave006_row_dma_before
#endif
extern "C" __global__ __aicore__ __attribute__((aiv)) void ENTRY(GM_ADDR x, GM_ADDR y) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, elements * sizeof(float));
    auto local = buffer.Get<float>();
    GlobalTensor<float> input, output;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), ROWS * COLUMNS);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), ROWS * COLUMNS);
    Duplicate(local, 0.0f, elements);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    // GM blockLen/sourceStride use bytes; UB destinationStride uses 32-byte blocks.
#if AFTER && ROWS <= 128
    DataCopyExtParams batched{ROWS, COLUMNS * sizeof(float), 0, 0, 0};
    DataCopyPad(local, input, batched, pad);
#else
    DataCopyExtParams single{1, COLUMNS * sizeof(float), 0, 0, 0};
    for (uint32_t row = 0; row < ROWS; ++row)
        DataCopyPad(local[row * pitch], input[row * COLUMNS], single, pad);
#endif
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Adds(local, local, 1.0f, elements);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams out{1, COLUMNS * sizeof(float), 0, 0, 0};
    for (uint32_t row = 0; row < ROWS; ++row)
        DataCopyPad(output[row * COLUMNS], local[row * pitch], out);
    PipeBarrier<PIPE_ALL>();
}
extern "C" uint32_t row_dma_rows() { return ROWS; }
extern "C" uint32_t row_dma_columns() { return COLUMNS; }
extern "C" uint32_t row_dma_after() { return AFTER; }
extern "C" uint32_t catlass_bridge_abi_version() { return 1; }
#if AFTER
extern "C" const char* catlass_bridge_kernel_names() { return "[\"wave006_row_dma_after\"]"; }
#else
extern "C" const char* catlass_bridge_kernel_names() { return "[\"wave006_row_dma_before\"]"; }
#endif
extern "C" int catlass_bridge_launch(void** pointers, uint64_t count, void* stream) {
    if (count != 2 || !pointers[0] || !pointers[1]) return 1;
    ENTRY<<<1, nullptr, stream>>>((uint8_t*)pointers[0], (uint8_t*)pointers[1]);
    return 0;
}
