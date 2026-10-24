// Native CATLASS 40aced927fda, CANN Open Software License v2 (CANN_LICENSE).
// Independent SDK-default/zero translation units; opaque pointer ABI only.
// Never call ShapeInfo APIs or pass AscendC descriptors between these builds.
#if W006_ZERO_SHAPE_INFO
#define K_MAX_SHAPE_DIM 0
#endif
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_mmad_small.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/kernel/small_matmul.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass_bridge.h"
#include "shape_metadata_abi.h"
#if CATLASS_ARCH != 2201
#error Unit completion flags require the qualified native 2201 path
#endif
using namespace Catlass;
using Operand = Gemm::GemmType<half, layout::RowMajor>;
using Policy = Gemm::MmadAtlasA2Small<2, false, false>;
using Block = Gemm::Block::BlockMmad<Policy,GemmShape<128,256,256>,
    GemmShape<128,256,64>,Operand,Operand,Operand>;
using Kernel = Gemm::Kernel::SmallMatmul<Block,void,Gemm::Block::GemmIdentityBlockSwizzle<3,0>>;
#define ENTRY wave006_shape_metadata
extern "C" CATLASS_GLOBAL __attribute__((aic)) void ENTRY(Kernel::Params params) { Kernel op; op(params); }
static ShapeMetadata::State state;
extern "C" int catlass_bridge_configure(uint32_t m,uint32_t n,uint32_t k,uint32_t cores) {
    return state.configure(m,n,k,cores);
}
extern "C" uint32_t catlass_shape_dimension() { return K_MAX_SHAPE_DIM; }
extern "C" uint32_t catlass_unit_flag_enabled() { return 0; }
extern "C" uint32_t catlass_bridge_abi_version() { return 1; }
#define STR_(x) #x
#define STR(x) STR_(x)
extern "C" const char* catlass_bridge_kernel_names() { return "[\"" STR(ENTRY) "\"]"; }
extern "C" int catlass_bridge_launch(void** p,uint64_t count,void* stream) {
    if (state.validate(p,count)) return 2;
    Kernel::Arguments args{GemmCoord(state.m,state.n,state.k),(uint8_t*)p[0],(uint8_t*)p[1],(uint8_t*)p[2]};
    auto params=Kernel::ToUnderlyingArguments(args,nullptr);
    ENTRY<<<state.cores,nullptr,stream>>>(params); return 0;
}
