// Native CATLASS 40aced927fda, CANN Open Software License v2 (CANN_LICENSE).
// Both instantiations share the entire upstream small body and terminal barrier.
#define K_MAX_SHAPE_DIM 0
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_mmad_small.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/kernel/small_matmul.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass_bridge.h"
#if CATLASS_ARCH != 2201
#error Conditional M-pipe barriers require the qualified native 2201 path
#endif
using namespace Catlass;
using Operand = Gemm::GemmType<half, layout::RowMajor>;
using Policy = Gemm::MmadAtlasA2Small<2, false, false>;
using Block = Gemm::Block::BlockMmad<Policy,GemmShape<W006_TILE_M,W006_TILE_N,256>,
    GemmShape<W006_TILE_M,W006_TILE_N,64>,Operand,Operand,Operand>;
using Kernel = Gemm::Kernel::SmallMatmul<Block,void,Gemm::Block::GemmIdentityBlockSwizzle<3,0>>;
#if W006_STOCK_BARRIER
#define ENTRY wave006_conditional_barrier_after
#else
#define ENTRY wave006_conditional_barrier_before
#endif
extern "C" CATLASS_GLOBAL __attribute__((aic)) void ENTRY(Kernel::Params params) { Kernel op; op(params); }
static uint32_t m_,n_,k_,cores_;
extern "C" int catlass_bridge_configure(uint32_t m,uint32_t n,uint32_t k,uint32_t cores) {
    // Dense FP16 ND rows with no padding; all dimensions align to 16 elements.
    if (!m || !n || !k || !cores || m%16 || n%16 || k%16 || k>256 ||
        m!=W006_TILE_M || n!=W006_TILE_N || uint64_t(1)>cores) return 1;
    m_=m; n_=n; k_=k; cores_=cores; return 0;
}
extern "C" uint32_t catlass_stock_barrier_enabled() { return W006_STOCK_BARRIER; }
extern "C" uint32_t catlass_tile_m() { return W006_TILE_M; }
extern "C" uint32_t catlass_tile_n() { return W006_TILE_N; }
extern "C" uint32_t catlass_bridge_abi_version() { return 1; }
#define STR_(x) #x
#define STR(x) STR_(x)
extern "C" const char* catlass_bridge_kernel_names() { return "[\"" STR(ENTRY) "\"]"; }
extern "C" int catlass_bridge_launch(void** p,uint64_t count,void* stream) {
    if (count!=3 || !cores_ || !p[0] || !p[1] || !p[2]) return 2;
    Kernel::Arguments args{GemmCoord(m_,n_,k_),(uint8_t*)p[0],(uint8_t*)p[1],(uint8_t*)p[2]};
    auto params=Kernel::ToUnderlyingArguments(args,nullptr);
    ENTRY<<<cores_,nullptr,stream>>>(params); return 0;
}
