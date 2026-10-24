"""Functional build-driver tests; the real A3 compiler gate is separate."""
import importlib.util
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / 'examples/optimization-library/cache-hint/compile_screen.py'
spec = importlib.util.spec_from_file_location('cache_hint_screen', MODULE)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def elf(payload=b'kernel', *, section='.aicore_binary'):
    names = b'\0.shstrtab\0' + section.encode() + b'\0'
    data = bytearray(64)
    data[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<Q', data, 40, 64 + len(names) + len(payload))
    struct.pack_into('<HHH', data, 58, 64, 3, 1)
    data.extend(names + payload)
    data.extend(bytes(64))
    data.extend(struct.pack('<IIQQQQIIQQ', 1, 3, 0, 0, 64, len(names), 0, 0, 1, 0))
    data.extend(struct.pack('<IIQQQQIIQQ', 11, 1, 0, 0, 64 + len(names), len(payload), 0, 0, 1, 0))
    return bytes(data)


class BuildDriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sdk = self.root / 'sdk'
        for name in ('include/acl/acl.h', 'include/ascendc/basic_api/kernel_operator.h'):
            p = self.sdk / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        self.commands = []

    def runner(self, command, **kwargs):
        self.commands.append(command)
        if '-dM' in command:
            return subprocess.CompletedProcess(command, 0,
                '#define __NPU_ARCH__ 2201\n#define __CCE_AICORE__ 220\n#define OTHER 1\n')
        Path(command[command.index('-o') + 1]).write_bytes(elf())
        return subprocess.CompletedProcess(command, 0, 'compiled\n')

    def build(self, runner=None, name='output'):
        return screen.compile_screen(self.sdk, self.root / name, compiler=sys.executable,
                                     runner=runner or self.runner)

    def test_identical_device_sections_do_not_qualify_runtime(self):
        result = self.build()
        self.assertEqual(len(self.commands), 5)
        self.assertTrue(all(v == 'identical_device_artifacts' for v in result['comparisons'].values()))
        self.assertEqual(result['outcome'], 'runtime_contract_unqualified')
        self.assertEqual(result['scientific_forwards'], 0)
        self.assertTrue((self.root / 'output/screen.json').is_file())
        self.assertEqual(result['device_preprocessing']['macros'],
            ['#define __CCE_AICORE__ 220', '#define __NPU_ARCH__ 2201'])
        for build in result['builds']:
            self.assertEqual(build['device_sections']['.aicore_binary']['bytes'], 6)
        with self.assertRaises(FileExistsError):
            self.build()  # Failed/successful evidence cannot be overwritten.

    def test_changed_payload_is_reported_without_claiming_runtime_support(self):
        def changed(command, **kwargs):
            result = self.runner(command, **kwargs)
            if '-DF12_HINT=1' in command:
                Path(command[command.index('-o') + 1]).write_bytes(elf(b'changed'))
            return result
        result = self.build(changed)
        self.assertEqual(set(result['comparisons'].values()), {'different_device_artifacts'})
        self.assertEqual(result['outcome'], 'runtime_contract_unqualified')

    def test_failed_build_preserves_log_and_other_arm(self):
        def failed(command, **kwargs):
            if '-DF12_HINT=1' in command:
                return subprocess.CompletedProcess(command, 17, 'unsupported API\n')
            return self.runner(command, **kwargs)
        result = self.build(failed)
        self.assertEqual(set(result['comparisons'].values()), {'build_or_artifact_failed'})
        self.assertEqual(result['builds'][1]['exit_code'], 17)
        self.assertEqual((self.root / 'output/default-1.log').read_text(), 'unsupported API\n')
        self.assertIn('device_sections', result['builds'][0])

    def test_timeout_and_invalid_artifact_are_failures(self):
        def bad(command, **kwargs):
            if '-DF12_HINT=0' in command:
                raise subprocess.TimeoutExpired(command, 1, output=b'partial diagnostics')
            result = self.runner(command, **kwargs)
            if '-o' in command:
                Path(command[command.index('-o') + 1]).write_bytes(b'not ELF')
            return result
        result = self.build(bad)
        self.assertTrue(result['builds'][0]['timed_out'])
        self.assertIn('artifact_error', result['builds'][1])
        self.assertEqual(set(result['comparisons'].values()), {'build_or_artifact_failed'})
        self.assertEqual((self.root / 'output/default-0.log').read_text(), 'partial diagnostics')

    def test_host_only_or_empty_payload_cannot_pass(self):
        for data in (elf(section='.text'), elf(b''), elf()[:70]):
            with self.subTest(data=data[:10]), self.assertRaises(ValueError):
                screen.device_sections(data)
        self.assertEqual(screen.device_sections(elf(b'x', section='__aicore_rel_binary')),
                         {'__aicore_rel_binary': dict(bytes=1, sha256=screen.sha256(b'x'))})


if __name__ == '__main__':
    unittest.main()
