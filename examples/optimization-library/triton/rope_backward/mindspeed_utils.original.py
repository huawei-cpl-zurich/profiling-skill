# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
from functools import lru_cache


@lru_cache
def is_arch35():
    try:
        import torch_npu

        return "Ascend910_95" in torch_npu.npu.get_device_name() or "Ascend950" in torch_npu.npu.get_device_name()
    except Exception:
        return False
