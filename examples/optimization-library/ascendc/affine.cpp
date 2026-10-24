// Standalone affine adaptation of native coefficient residency; see N06 provenance.
#include "kernel_operator.h"
#include <stdint.h>
using namespace AscendC;
constexpr uint32_t pitch = (W006_N06_WIDTH + 7) / 8 * 8;
#if W006_N06_AFTER
#define W006_N06_ENTRY wave006_affine_after
#else
#define W006_N06_ENTRY wave006_affine_before
#endif
extern "C" __global__ __aicore__ __attribute__((aiv)) void W006_N06_ENTRY(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y) {
    const uint32_t block = GetBlockIdx();
    const uint32_t chunk = (W006_N06_ROWS + W006_N06_GRID - 1) / W006_N06_GRID;
    const uint32_t first = block * chunk;
    if (first >= W006_N06_ROWS) return;
    TPipe pipe;
    TBuf<TPosition::VECCALC> xb, gb, bb;
    pipe.InitBuffer(xb, pitch * sizeof(float));
    pipe.InitBuffer(gb, pitch * sizeof(float));
    pipe.InitBuffer(bb, pitch * sizeof(float));
    auto local = xb.Get<float>(); auto g = gb.Get<float>(); auto b = bb.Get<float>();
    GlobalTensor<float> input, output, gammaGM, betaGM;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), W006_N06_ROWS*W006_N06_WIDTH);
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), W006_N06_ROWS*W006_N06_WIDTH);
    gammaGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gamma), W006_N06_WIDTH);
    betaGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta), W006_N06_WIDTH);
    DataCopyExtParams copy{1, W006_N06_WIDTH*sizeof(float), 0, 0, 0};
    DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
#if W006_N06_AFTER
    DataCopyPad(g, gammaGM, copy, pad); DataCopyPad(b, betaGM, copy, pad);
#endif
    for (uint32_t row = first; row < W006_N06_ROWS && row < first+chunk; ++row) {
#if !W006_N06_AFTER
        DataCopyPad(g, gammaGM, copy, pad); DataCopyPad(b, betaGM, copy, pad);
#endif
        DataCopyPad(local, input[row*W006_N06_WIDTH], copy, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Mul(local, local, g, W006_N06_WIDTH);
        PipeBarrier<PIPE_V>();
        Add(local, local, b, W006_N06_WIDTH);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(output[row*W006_N06_WIDTH], local, copy);
        // Same conservative completion guarantees both coefficient and row lifetimes.
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
}
extern "C" uint32_t affine_rows(){return W006_N06_ROWS;}
extern "C" uint32_t affine_width(){return W006_N06_WIDTH;}
extern "C" uint32_t affine_grid(){return W006_N06_GRID;}
extern "C" uint32_t affine_after(){return W006_N06_AFTER;}
extern "C" uint32_t catlass_bridge_abi_version(){return 1;}
#if W006_N06_AFTER
extern "C" const char* catlass_bridge_kernel_names(){return "[\"wave006_affine_after\"]";}
#else
extern "C" const char* catlass_bridge_kernel_names(){return "[\"wave006_affine_before\"]";}
#endif
extern "C" int catlass_bridge_launch(void** p, uint64_t count, void* stream){
    if(count!=4 || !p[0] || !p[1] || !p[2] || !p[3])return 1;
    W006_N06_ENTRY<<<W006_N06_GRID,nullptr,stream>>>((uint8_t*)p[0],(uint8_t*)p[1],(uint8_t*)p[2],(uint8_t*)p[3]);
    return 0;
}
