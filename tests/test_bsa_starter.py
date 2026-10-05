"""Functional host dispatch checks for the neutral BSA Triton starter."""

import importlib.util
from pathlib import Path

import pytest
import torch


STARTER = Path(__file__).resolve().parents[1] / "benchmarks" / "bsa" / "starter.py"


def load_starter():
    spec = importlib.util.spec_from_file_location("bsa_starter", STARTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("exact,causal", [(False, False), (False, True), (True, True)])
def test_interpreted_kernel_matches_reference(monkeypatch, exact, causal):
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    starter = load_starter()
    baseline_spec = importlib.util.spec_from_file_location(
        "bsa_baseline", STARTER.with_name("baseline.py"))
    baseline = importlib.util.module_from_spec(baseline_spec)
    baseline_spec.loader.exec_module(baseline)
    torch.manual_seed(7)
    q = torch.randn((5, 3, 64), dtype=torch.float16)
    k = torch.randn((9, 1, 64), dtype=torch.float16)
    v = torch.randn((9, 1, 64), dtype=torch.float16)
    cuq = torch.tensor([0, 2, 5], dtype=torch.int32)
    cuk = torch.tensor([0, 4, 9], dtype=torch.int32)
    hmt = torch.tensor([0, 1, -1], dtype=torch.int32)
    sinfo = torch.tensor([0, 0, 0, 0, 1, 2], dtype=torch.int32)
    mask = torch.tensor([[[[False]]], [[[True]]]])
    args = (q, k, v, cuq, cuk, hmt, sinfo, mask, None, causal, exact)
    actual = starter.Model()(*args)
    expected = baseline.Model()(*args)
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)
    assert torch.count_nonzero(actual[:2, 1]) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_interpreted_multiblock_strided_layout(monkeypatch, dtype):
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    starter = load_starter()
    baseline_spec = importlib.util.spec_from_file_location(
        "bsa_baseline", STARTER.with_name("baseline.py"))
    baseline = importlib.util.module_from_spec(baseline_spec)
    baseline_spec.loader.exec_module(baseline)
    torch.manual_seed(11)
    q = torch.randn((129, 3, 128), dtype=dtype)[:, :, ::2]
    k = torch.randn((257, 1, 128), dtype=dtype)[:, :, ::2]
    v = torch.randn((257, 1, 128), dtype=dtype)[:, :, ::2]
    cuq = torch.tensor([0, 129], dtype=torch.int32)
    cuk = torch.tensor([0, 257], dtype=torch.int32)
    hmt = torch.tensor([0, 1, -1], dtype=torch.int32)
    sinfo = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.int32)
    mask = torch.tensor([[[[True, False, True],
                           [False, False, False]]]])
    args = (q, k, v, cuq, cuk, hmt, sinfo, mask, None, True, False)
    actual = starter.Model()(*args)
    expected = baseline.Model()(*args)
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)
    assert torch.count_nonzero(actual[128, 1]) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_sequences_launch_once_and_preserve_mapping(monkeypatch, dtype):
    starter = load_starter()
    launches = []

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(q, k, v, out, *args, **kwargs):
                launches.append((grid, args, kwargs))
                # Execute an observable stand-in for the device write.
                qs, qe, ks, ke = args[4:8]
                out[qs:qe] = (qs + ks + 1)

            return launch

    monkeypatch.setattr(starter, "_bsa_reference_fwd", FakeKernel())
    q = torch.zeros((21, 6, 64), dtype=dtype)
    k = v = torch.zeros((40, 2, 64), dtype=dtype)
    cuq = torch.tensor([0, 5, 21], dtype=torch.int32)
    cuk = torch.tensor([0, 8, 40], dtype=torch.int32)
    hmt = torch.tensor([0, 1, -1, 1, 0, -1], dtype=torch.int32)
    sinfo = torch.tensor([1, 2] * 6, dtype=torch.int32)
    mask = torch.ones((2, 2, 1, 1), dtype=torch.bool)
    out = starter.Model()(q, k, v, cuq, cuk, hmt, sinfo, mask, None, True, False)
    assert out.dtype == dtype and out.shape == q.shape
    assert len(launches) == 2
    assert [item[0] for item in launches] == [(1, 6), (1, 6)]
    assert [item[1][4:8] for item in launches] == [(0, 5, 0, 8), (5, 21, 8, 40)]
    assert torch.all(out[:5] == 1)
    assert torch.all(out[5:] == 14)
