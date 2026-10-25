// Copyright (c) 2026 Huawei Technologies Co., Ltd. OSLA 2.0; source-derived launch bridge.
#include <cstdint>
#include <cstddef>
#if BEFORE
#include "vendor/sasum_kernel.cpp"
#else
#include "sasum_partial.cpp"
#endif
extern "C" uint32_t catlass_bridge_abi_version() { return 1; }
extern "C" const char* catlass_bridge_kernel_names() {
#if BEFORE
    return "[\"sasum_kernel\"]";
#else
    return "[\"sasum_core_partial\"]";
#endif
}
extern "C" int catlass_bridge_launch(void** p, uint64_t n, void* stream) {
    if (n != 4 || !p || !stream) return 1;
#if BEFORE
    sasum_kernel_do((uint8_t*)p[0],(uint8_t*)p[3],(uint8_t*)p[1],(uint8_t*)p[2],8,stream);
#else
    sasum_core_partial_do((uint8_t*)p[0],(uint8_t*)p[3],(uint8_t*)p[1],(uint8_t*)p[2],8,stream);
#endif
    return 0; // Source launch wrappers are void; this is not device completion proof.
}
