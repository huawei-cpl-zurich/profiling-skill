// Educational controlled adaptation of CATLASS 40aced927fda; CANN_LICENSE applies.
#define K_MAX_SHAPE_DIM 0
#include "acl/acl.h"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "single_stage_block.hpp"
#include "single_stage_kernel.hpp"
#include "catlass_bridge.h"
#if CATLASS_ARCH != 2201
#error Only native Atlas A2/A3 architecture 2201 is qualified
#endif
using namespace Catlass;
using A = Gemm::GemmType<half, layout::RowMajor>;
using Base = Gemm::Block::BlockMmad<Gemm::MmadAtlasA2Small<2,false,false>,
    GemmShape<128,256,256>,GemmShape<128,256,64>,A,A,A>;
using Block = Gemm::Block::SingleStageBlock<CONTROL_DIRECT,Base>;
using Kernel = Gemm::Kernel::SingleStageKernel<CONTROL_DIRECT,Block>;
#if CONTROL_DIRECT
#define ENTRY wave006_single_stage_direct
#else
#define ENTRY wave006_single_stage_general
#endif
extern "C" CATLASS_GLOBAL __attribute__((aic)) void ENTRY(Kernel::Params params) { Kernel op; op(params); }
static uint32_t m_,n_,k_,cores_;
extern "C" int catlass_bridge_configure(uint32_t m,uint32_t n,uint32_t k,uint32_t cores) {
    if (!m || !n || !k || !cores || m%16 || n%16 || k%16 ||
        (CONTROL_DIRECT && (uint64_t((m+127)/128)*((n+255)/256)>cores || k>256))) return 1;
    m_=m; n_=n; k_=k; cores_=cores; return 0;
}
extern "C" uint32_t catlass_bridge_abi_version() { return 1; }
#if CONTROL_DIRECT
extern "C" const char *catlass_bridge_kernel_names() { return "[\"wave006_single_stage_direct\"]"; }
#else
extern "C" const char *catlass_bridge_kernel_names() { return "[\"wave006_single_stage_general\"]"; }
#endif
extern "C" int catlass_bridge_launch(void **p,uint64_t count,void *stream) {
    if(count!=3 || !cores_ || !p[0] || !p[1] || !p[2]) return 2;
    Kernel::Arguments args{GemmCoord(m_,n_,k_),(uint8_t*)p[0],(uint8_t*)p[1],(uint8_t*)p[2]};
    auto params=Kernel::ToUnderlyingArguments(args,nullptr);
    ENTRY<<<cores_,nullptr,stream>>>(params); return 0;
}
