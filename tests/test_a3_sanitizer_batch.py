from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from a3_sanitizer_batch import classify, memory_events, run_check


def test_report_detection_does_not_depend_on_nonzero_exit():
    assert classify('WARNING: WAW hazard at UB address 0x10', 0) == 'detected'
    assert classify('ERROR: invalid write at GM address 0x20', 0) == 'detected'


def test_zero_exit_alone_is_not_clean_evidence():
    assert classify('', 0) == 'unqualified-report'
    assert classify('ERROR SUMMARY: 0 errors', 0) == 'no-finding'
    assert classify('racecheck not supported', 1) == 'unsupported'
    assert classify('compiler failed to instrument', 1) == 'instrumentation-failure'
    assert classify('WAW hazard', 0, timeout=True) == 'timeout'
    assert classify("ERROR: param 'full-backtrace' contains invalid characters\nUsage: mssanitizer\nout of bounds detection", 0) == 'instrumentation-failure'
    assert classify('Warning:Register FFTS_BASE_ADDR was not reset\nSanitizer finished on kernel foo. See all detected errors above.', 0) == 'other-finding'


def test_actual_process_report_retained(tmp_path, monkeypatch):
    tool = tmp_path / 'mssanitizer'
    tool.write_text('#!/usr/bin/env python3\nimport pathlib,sys\nassert sys.argv[sys.argv.index("--") + 1:] == ["unused", "--artifacts", "output"]\np=next(x.split("=",1)[1] for x in sys.argv if x.startswith("--log-file="))\npathlib.Path(p).write_text("Start racecheck sanitizer on kernel a3_sanitizer_write\\nWAW hazard at GM address 0x10\\nSanitizer finished on kernel a3_sanitizer_write. See all detected errors above.")\n')
    tool.chmod(0o700)
    import os
    env = dict(os.environ, PATH=str(tmp_path) + ':' + os.environ['PATH'])
    folder = tmp_path / 'capture'
    result = run_check(['unused', '--artifacts', 'output'], folder, env, 'racecheck')
    assert result['classification'] == 'detected'
    assert result['reports'][0]['sha256']
    assert 'WAW hazard at GM address 0x10' in result['diagnostic_lines']
    assert (folder / 'command.json').exists()


def test_missing_tool_is_retained_failure(tmp_path):
    result = run_check(['unused'], tmp_path / 'missing', {'PATH': str(tmp_path)}, 'racecheck')
    assert result['classification'] == 'instrumentation-failure'
    assert result['exit_code'] == 127


def test_clean_summary_without_kernel_analysis_is_unqualified(tmp_path):
    tool = tmp_path / 'mssanitizer'
    tool.write_text('#!/bin/sh\nprintf "Start racecheck sanitizer on kernel unrelated\\nSanitizer finished on kernel unrelated. No error detected.\\n"\n')
    tool.chmod(0o700)
    import os
    result = run_check(['unused'], tmp_path / 'capture', dict(os.environ, PATH=str(tmp_path)), 'racecheck')
    assert result['classification'] == 'unqualified-report'
    assert not result['analysis_started']


def test_unrelated_unsupported_kernel_does_not_hide_target_analysis(tmp_path):
    tool = tmp_path / 'mssanitizer'
    tool.write_text('#!/bin/sh\nprintf "racecheck not supported on unrelated\\nStart racecheck sanitizer on kernel a3_sanitizer_write\\nSanitizer finished on kernel a3_sanitizer_write. No error detected.\\n"\n')
    tool.chmod(0o700)
    import os
    result = run_check(['unused'], tmp_path / 'capture', dict(os.environ, PATH=str(tmp_path)), 'racecheck')
    assert result['classification'] == 'no-finding'
    assert result['target_kernels'] == ['a3_sanitizer_write']
    assert result['analysis_finished']


def test_instrumented_store_evidence_preserves_core_and_address():
    text = 'serialNo:155, type:STORE, coreId:0, srcAddr:0x0, dstAddr:0x12c0c0014000, memInfoDesc:default, memInfoSrc:BYPASS, srcSpace:GM, dstSpace:GM, memSize:16384, moduleId:-1, lineNo:17, fileName:'
    rows = memory_events(text)
    assert rows[0]['dstAddr'] == '0x12c0c0014000'
    assert rows[0]['coreId'] == '0'
    assert rows[0]['lineNo'] == '17'


def test_incomplete_target_or_auxiliary_hazard_is_not_qualified(tmp_path):
    import os
    for index, report in enumerate((
        'Start racecheck sanitizer on kernel a3_sanitizer_write\\nWAW hazard at UB address 0x0',
        'Start racecheck sanitizer on kernel auxiliary\\nWAW hazard at GM address 0x10\\nSanitizer finished on kernel auxiliary. See all detected errors above.',
    )):
        tool = tmp_path / 'mssanitizer'
        tool.write_text('#!/bin/sh\nprintf "' + report + '\\n"\n')
        tool.chmod(0o700)
        result = run_check(['unused'], tmp_path / str(index), dict(os.environ, PATH=str(tmp_path)), 'racecheck')
        assert result['classification'] == 'unqualified-report'
