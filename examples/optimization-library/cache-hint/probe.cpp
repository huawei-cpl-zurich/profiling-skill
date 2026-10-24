// Compile-only source-level reproduction of the F12 weight-hint consumer.
// This is not an executable A16W8 MSD implementation or a numerical test.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using ProbeA = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;
using ProbeB = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;
using ProbeC = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

// A holds the two decomposed int8 components in stacked rows, as in the
// source MSD matmul. Decomposition/reconstruction are deliberately untested.
extern "C" __global__ __aicore__ void f12_compile_probe(
    GM_ADDR components, GM_ADDR weights, GM_ADDR products, GM_ADDR tiling)
{
    GlobalTensor<int8_t> a, w;
    GlobalTensor<int32_t> c;
    a.SetGlobalBuffer((__gm__ int8_t*)components);
    w.SetGlobalBuffer((__gm__ int8_t*)weights);
    c.SetGlobalBuffer((__gm__ int32_t*)products);
#if F12_HINT
    w.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
    TCubeTiling config;
    for (uint32_t i = 0; i < sizeof(config) / sizeof(uint32_t); ++i) {
        ((uint32_t*)&config)[i] = ((__gm__ uint32_t*)tiling)[i];
    }
    matmul::Matmul<ProbeA, ProbeB, ProbeC> mm;
    mm.Init(&config);
    mm.SetTensorA(a);
    mm.SetTensorB(w);
    mm.IterateAll(c);
    mm.End();
}
