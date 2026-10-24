#pragma once
#include <stdint.h>
// Launch is asynchronous on the supplied current ACL stream. Pointer order,
// shape/dtype and workspace requirements are part of each adapter's contract.
// Return nonzero on a host-side launch error. Never allocate or synchronize here.
extern "C" uint32_t catlass_bridge_abi_version();
extern "C" const char *catlass_bridge_kernel_names(); // JSON array, exact runtime names
extern "C" int catlass_bridge_launch(void **pointers, uint64_t count, void *stream);
