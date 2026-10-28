import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).parents[1] / "scripts/benchmark_aclnn_bsa_gdn.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_summary_sample_sd():
    assert benchmark.summary([1, 3, 5]) == {
        "samples_us": [1, 3, 5], "mean_us": 3, "sample_sd_us": 2}
    assert benchmark.summary([1])["sample_sd_us"] is None
    with pytest.raises(ValueError):
        benchmark.summary([])


def test_replay_calls_only_warmup_plus_one():
    calls, syncs = [], []
    def call():
        calls.append(1)
        return len(calls)
    def forbidden_event():
        raise AssertionError("replay must not create timing events")
    output, timing = benchmark.execute(call, lambda: syncs.append(1), forbidden_event, 3, 10, True)
    assert output == 4 and len(calls) == 4 and len(syncs) == 2 and timing == {}


def test_measurement_uses_independent_wall_and_event_samples():
    calls = []
    class Event:
        def record(self):
            pass
        def elapsed_time(self, other):
            return 0.5
    _, result = benchmark.execute(lambda: calls.append(1), lambda: None, Event, 2, 3)
    assert len(calls) == 8
    assert len(result["synchronized_wall"]["samples_us"]) == 3
    assert result["npu_event"]["mean_us"] == 500


def test_sparse_attention_excludes_unselected_block():
    q, k = torch.zeros(4, 1, 2), torch.zeros(4, 1, 2)
    v = torch.tensor([1, 3, 100, 200]).view(4, 1, 1).expand(4, 1, 2)
    indices = torch.tensor([[[0, -1]], [[1, -1]]])
    counts = torch.ones(2, 1, dtype=torch.int64)
    result, = benchmark.bsa_reference(q, k, v, indices, counts, block=2)
    torch.testing.assert_close(result[:, 0, 0], torch.tensor([2., 2., 150., 150.]))
    indices[:] = torch.tensor([0, 1])
    counts[:] = 2
    dense, = benchmark.bsa_reference(q, k, v, indices, counts, block=2)
    torch.testing.assert_close(dense, v.float().mean(0).expand_as(dense))


def test_gdn_chunk_start_states_and_residual_values():
    k = torch.ones(1, 1, 4, 1)
    w = torch.full_like(k, 0.5)
    u = torch.ones_like(k)
    initial = torch.full((1, 1, 1, 1), 0.5)
    h, value, final = benchmark.gdn_reference(k, w, u, initial, 2)
    torch.testing.assert_close(h.flatten(), torch.tensor([0.5, 2.]))
    torch.testing.assert_close(value.flatten(), torch.tensor([0.75, 0.75, 0., 0.]))
    torch.testing.assert_close(final, torch.full_like(initial, 2.))
    checks = benchmark.check_outputs((h, value, final), (h, value, final), 0, 0)
    assert len(checks) == 3 and all(c["max_abs_error"] == 0 for c in checks)
    with pytest.raises(AssertionError):
        benchmark.check_outputs((h + 1,), (h,), 0, 0)


def test_gdn_schema_adapter_keeps_stage_contract():
    common = {"initial_state", "output_final_state", "chunk_size", "cu_seqlens"}
    current = benchmark.gdn_options(common | {"state_v_first"}, "state", 64)
    assert current["state_v_first"] is False and current["initial_state"] == "state"
    legacy = benchmark.gdn_options(common | {
        "transpose_state_layout", "use_exp2", "save_new_value"}, "state", 128)
    assert legacy["transpose_state_layout"] is False
    assert legacy["use_exp2"] is False and legacy["save_new_value"] is True
    assert legacy["chunk_size"] == 128
    with pytest.raises(RuntimeError):
        benchmark.gdn_options(common, None, 64)


def test_incorrect_output_fails_before_warmup_or_measurement():
    calls = []
    def wrong():
        calls.append(1)
        return (torch.ones(2),)
    with pytest.raises(AssertionError):
        benchmark.run_checked(wrong, (torch.zeros(2),), lambda: None,
                              lambda: pytest.fail("timing started"), 20, 10, False)
    assert len(calls) == 1


def test_checked_replay_preserves_launch_count():
    calls = []
    def correct():
        calls.append(1)
        return (torch.ones(2),)
    checks, timing = benchmark.run_checked(correct, (torch.ones(2),), lambda: None,
                                          lambda: pytest.fail("timing started"), 2, 10, True)
    assert len(calls) == 3 and checks[0]["max_abs_error"] == 0 and timing == {}
