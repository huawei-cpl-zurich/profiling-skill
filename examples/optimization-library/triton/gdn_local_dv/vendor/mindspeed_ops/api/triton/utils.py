# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
import itertools
import functools
import warnings
import contextlib

import torch
import triton


from typing import Any, Callable, Optional


def _cpu_device_warning():
    warnings.warn('Triton is not supported on current platform, roll back to CPU.', stacklevel=1)


@functools.cache
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        _cpu_device_warning()
        return 'cpu'


@functools.cache
def get_vector_num() -> int:
    from triton.runtime import driver
    import torch_npu

    device = torch_npu.npu.current_device()
    properties = driver.active.utils.get_device_properties(device)
    return properties["num_vectorcore"]


@functools.cache
def get_aicore_num() -> int:
    from triton.runtime import driver
    import torch_npu

    device = torch_npu.npu.current_device()
    properties = driver.active.utils.get_device_properties(device)
    return properties["num_aicore"]


def _infer_target_device(args, kwargs):
    """Infer target device using Triton's device detection."""
    backend = get_available_device()
    if backend == 'cpu':
        return torch.device('cpu')

    # 从 Triton 运行时获取当前活跃的设备索引
    try:
        device_index = triton.runtime.driver.active.get_current_device()
    except BaseException:
        device_index = 0

    return torch.device(backend, device_index)


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args)
        contiguous_kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            device = get_available_device()
            device_torch_lib = getattr(torch, device)
            ctx = device_torch_lib.device(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


def tensor_cache(fn: Optional[Callable[..., torch.Tensor]] = None, *, maxsize: int = 1) -> Any:
    """
    A decorator that caches the most recent results of a function with tensor inputs.

    This decorator will store the outputs of the decorated function for the most recent
    set of input tensors, up to `maxsize` entries. If the function is called again with
    the same input tensors, it will return the cached result.

    When maxsize=1 (default), the behavior is identical to caching only the most recent result.
    Can be used as @tensor_cache or @tensor_cache(maxsize=n).

    Args:
        fn (Callable[..., torch.Tensor], optional):
            The function to be decorated when used without parentheses.
        maxsize (int):
            Maximum number of input combinations to cache. Default is 1.

    Returns:
        Callable[..., torch.Tensor]:
        A wrapped version of the input function with caching.
    """
    if maxsize < 1:
        raise ValueError("maxsize must be at least 1")

    def _is_match(a: Any, b: Any) -> bool:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return a is b
        try:
            return a == b
        except Exception:
            return a is b

    def _make_wrapper(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        cache: list = []

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for i, (cached_args, cached_kwargs, cached_result) in enumerate(cache):
                if len(args) == len(cached_args) and len(kwargs) == len(cached_kwargs):
                    if all(_is_match(a, b) for a, b in zip(args, cached_args)) and all(
                        k in cached_kwargs and _is_match(v, cached_kwargs[k]) for k, v in kwargs.items()
                    ):
                        if i != 0:
                            cache.insert(0, cache.pop(i))
                        return cached_result

            result = fn(*args, **kwargs)
            cache.insert(0, (args, kwargs, result))
            if len(cache) > maxsize:
                cache.pop()
            return result

        return wrapper

    if fn is not None:
        return _make_wrapper(fn)
    return _make_wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache(maxsize=3)
def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1)


def get_autotune_config(
    multibuffer_list: tuple = (False,),
    unit_flag_list: tuple = (False,),
    limit_auto_multi_buffer_only_for_local_buffer_list: tuple = (False,),
    limit_auto_multi_buffer_of_local_buffer_list: tuple = ("no-l0c",),
    set_workspace_multibuffer_list: tuple = (2, 4),
    enable_hivm_auto_cv_balance_list: tuple = (True,),
    tile_mix_vector_loop_num_list: tuple = (2, 4),
    tile_mix_cube_loop_num_list: tuple = (2, 4),
):
    configs = []
    for (
        multibuffer,
        unit_flag,
        limit_auto_multi_buffer_only_for_local_buffer,
        limit_auto_multi_buffer_of_local_buffer,
    ) in itertools.product(
        list(multibuffer_list),
        list(unit_flag_list),
        list(limit_auto_multi_buffer_only_for_local_buffer_list),
        list(limit_auto_multi_buffer_of_local_buffer_list),
    ):
        base_config_dict = {
            'multibuffer': multibuffer,
            'unit_flag': unit_flag,
            'limit_auto_multi_buffer_only_for_local_buffer': limit_auto_multi_buffer_only_for_local_buffer,
            'limit_auto_multi_buffer_of_local_buffer': limit_auto_multi_buffer_of_local_buffer,
        }

        if limit_auto_multi_buffer_only_for_local_buffer:
            configs.append(triton.Config(base_config_dict))
        else:
            for (
                set_workspace_multibuffer,
                enable_hivm_auto_cv_balance,
                tile_mix_vector_loop,
                tile_mix_cube_loop,
            ) in itertools.product(
                list(set_workspace_multibuffer_list),
                list(enable_hivm_auto_cv_balance_list),
                list(tile_mix_vector_loop_num_list),
                list(tile_mix_cube_loop_num_list),
            ):
                full_config_dict = base_config_dict.copy()
                full_config_dict.update(
                    {
                        'set_workspace_multibuffer': set_workspace_multibuffer,
                        'enable_hivm_auto_cv_balance': enable_hivm_auto_cv_balance,
                        'tile_mix_vector_loop': tile_mix_vector_loop,
                        'tile_mix_cube_loop': tile_mix_cube_loop,
                    }
                )
                configs.append(triton.Config(full_config_dict))
    return configs


autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=get_available_device())
autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=get_available_device())
