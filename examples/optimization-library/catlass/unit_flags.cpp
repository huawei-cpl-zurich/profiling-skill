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
#error Unit completion flags require the qualified native 2201 path
#endif
using namespace Catlass;
using Operand = Gemm::GemmType<half, layout::RowMajor>;
using Policy = Gemm::MmadAtlasA2Small<2, bool(W006_UNIT_FLAGS), false>;
using Block = Gemm::Block::BlockMmad<Policy,GemmShape<128,256,256>,
    GemmShape<128,256,64>,Operand,Operand,Operand>;
using Kernel = Gemm::Kernel::SmallMatmul<Block,void,Gemm::Block::GemmIdentityBlockSwizzle<3,0>>;
#if W006_UNIT_FLAGS
#define ENTRY wave006_unit_flags_after
#else
#define ENTRY wave006_unit_flags_before
#endif
extern "C" CATLASS_GLOBAL __attribute__((aic)) void ENTRY(Kernel::Params params) { Kernel op; op(params); }
static uint32_t m_,n_,k_,cores_;
extern "C" int catlass_bridge_configure(uint32_t m,uint32_t n,uint32_t k,uint32_t cores) {
    // Dense FP16 ND rows with no padding; all dimensions align to 16 elements.
    if (!m || !n || !k || !cores || m%16 || n%16 || k%16 || k>256 ||
        uint64_t((m+127)/128)*((n+255)/256)>cores) return 1;
    m_=m; n_=n; k_=k; cores_=cores; return 0;
}
extern "C" uint32_t catlass_unit_flag_enabled() { return W006_UNIT_FLAGS; }
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
