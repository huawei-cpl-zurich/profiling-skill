// Independent indexed-fetch adaptation of ops-sparse CANNOpenSoftware2.0 sources.
#include "kernel_operator.h"
#include <stdint.h>
using namespace AscendC;
constexpr uint32_t tile = 4096;
#if AFTER
#define ENTRY wave006_ub_gather_after
#else
#define ENTRY wave006_ub_gather_before
#endif
extern "C" __global__ __aicore__ __attribute__((aiv)) void ENTRY(GM_ADDR x, GM_ADDR ix, GM_ADDR y) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> tableBuffer, indexBuffer, outputBuffer;
    pipe.InitBuffer(tableBuffer, GATHER_K * sizeof(float));
    pipe.InitBuffer(indexBuffer, tile * sizeof(int32_t));
    pipe.InitBuffer(outputBuffer, tile * sizeof(float));
    auto table = tableBuffer.Get<float>();
    auto indices = indexBuffer.Get<int32_t>();
    auto result = outputBuffer.Get<float>();
    GlobalTensor<float> input, output;
    GlobalTensor<int32_t> index;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), GATHER_K);
    index.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ix), GATHER_N);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), GATHER_N);
    DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    DataCopyExtParams copyTable{1, GATHER_K * sizeof(float), 0, 0, 0};
    DataCopyPad(table, input, copyTable, pad);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t start = 0; start < GATHER_N; start += tile) {
        const uint32_t valid = GATHER_N - start < tile ? GATHER_N - start : tile;
        DataCopyExtParams copy{1, static_cast<uint32_t>(valid * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPad{false, 0, 0, 0};
        DataCopyPad(indices, index[start], copy, indexPad);
        PipeBarrier<PIPE_ALL>();
        Muls(indices, indices, static_cast<int32_t>(sizeof(float)), valid);
        PipeBarrier<PIPE_ALL>();
        auto offsets = indices.ReinterpretCast<uint32_t>();
#if AFTER
        Gather(result, table, offsets, 0, valid);
#else
        auto base = reinterpret_cast<__ubuf__ uint8_t*>(table.GetPhyAddr());
        auto byteOffsets = reinterpret_cast<__ubuf__ uint32_t*>(offsets.GetPhyAddr());
        auto destination = reinterpret_cast<__ubuf__ float*>(result.GetPhyAddr());
        for (uint32_t i = 0; i < valid; ++i)
            destination[i] = *reinterpret_cast<__ubuf__ float*>(base + byteOffsets[i]);
#endif
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(output[start], result, copy);
        PipeBarrier<PIPE_ALL>();
    }
}
extern "C" uint32_t ub_gather_n() { return GATHER_N; }
extern "C" uint32_t ub_gather_k() { return GATHER_K; }
extern "C" uint32_t ub_gather_after() { return AFTER; }
extern "C" uint32_t catlass_bridge_abi_version() { return 1; }
#if AFTER
extern "C" const char* catlass_bridge_kernel_names() { return "[\"wave006_ub_gather_after\"]"; }
#else
extern "C" const char* catlass_bridge_kernel_names() { return "[\"wave006_ub_gather_before\"]"; }
#endif
extern "C" int catlass_bridge_launch(void** p, uint64_t count, void* stream) {
    if (count != 3 || !p[0] || !p[1] || !p[2]) return 1;
    ENTRY<<<1, nullptr, stream>>>((uint8_t*)p[0], (uint8_t*)p[1], (uint8_t*)p[2]);
    return 0;
}
