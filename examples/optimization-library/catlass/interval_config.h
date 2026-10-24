#pragma once
#include <stdint.h>
#ifndef AFTER
#define AFTER 0
#endif
#if AFTER != 0 && AFTER != 1
#error AFTER must select exactly one arm
#endif
static uint32_t interval_m, interval_n, interval_k, interval_cores;
extern "C" int catlass_bridge_configure(uint32_t m, uint32_t n, uint32_t k, uint32_t cores) {
    // Row tails are deliberate; half-precision DMA blocks require aligned N/K.
    if (!m || !n || !k || !cores || m>32768 || n>32768 || k>32768 || n%16 || k%16) return 1;
    interval_m=m; interval_n=n; interval_k=k; interval_cores=cores; return 0;
}
extern "C" uint32_t catlass_interval_after() { return AFTER; }
extern "C" uint32_t catlass_bridge_abi_version() { return 1; }
extern "C" const char *catlass_bridge_kernel_names() {
#if AFTER
    return "[\"wave2_interval_after\"]";
#else
    return "[\"wave2_interval_before\"]";
#endif
}
