"""Exercise a real shared-library ABI, not a mocked native kernel."""
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from optimization_native_bridge import NativeArm, build_shared


SOURCE = r'''
#include <stdint.h>
static int calls = 0;
extern "C" uint32_t catlass_bridge_abi_version() { return 1; }
extern "C" const char *catlass_bridge_kernel_names() { return "[\"native_add\"]"; }
extern "C" int catlass_bridge_launch(void **p, uint64_t n, void *stream) {
    if (n != 3 || stream != (void*)17) return 9;
    ++calls;
    for (int i = 0; i < 4; ++i) ((float*)p[2])[i] = ((float*)p[0])[i] + ((float*)p[1])[i];
    return 0;
}
extern "C" int launch_count() { return calls; }
'''


@unittest.skipUnless(shutil.which('c++'), 'a host C++ compiler is required')
class NativeBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'native.cpp'
        self.source.write_text(SOURCE)
        self.binary = self.root / 'native.so'
        self.manifest = build_shared(['c++', '-std=c++17', '-shared', '-fPIC',
            str(self.source), '-o', str(self.binary)], source_paths=[self.source],
            binary=self.binary, output=self.root / 'build', source_revision='test-pin')
        self.arms = []

    def tearDown(self):
        for arm in self.arms:
            arm.close()
        self.temp.cleanup()

    def arm(self, **kwargs):
        values = dict(binary=self.binary, inputs=[torch.arange(4.), torch.ones(4)],
            output=torch.empty(4), sources=[self.source], source_revision='test-pin',
            stream_provider=lambda: 17, build_manifest=self.manifest)
        values.update(kwargs)
        arm = NativeArm(**values)
        self.arms.append(arm)
        return arm

    def test_pointer_launch_exact_counts_output_and_source_identity(self):
        arm = self.arm()
        expected = torch.tensor([1., 2., 3., 4.])  # Independent oracle.
        original = [value.clone() for value in arm.inputs]
        torch.testing.assert_close(arm.run(), expected)
        for _ in range(10):
            arm.run()  # The measurement loop never examines input or output.
        self.assertEqual(arm._library.launch_count(), 11)
        torch.testing.assert_close(arm.output, expected)
        for actual, prior in zip(arm.inputs, original):
            torch.testing.assert_close(actual, prior)
        self.assertEqual(arm.expected_kernels, ['native_add'])
        self.assertEqual(arm.compilation_state(), arm._initial_identity)
        self.assertEqual(arm.metadata['binary_sha256'], self.manifest['binary_sha256'])
        self.source.write_text(SOURCE + '\n// changed after preparation\n')
        self.assertNotEqual(arm.compilation_state(), arm._initial_identity)

    def test_current_stream_is_queried_and_native_error_propagates(self):
        streams = iter([17, 18])
        arm = self.arm(stream_provider=lambda: next(streams))
        arm.run()
        with self.assertRaisesRegex(RuntimeError, 'status 9'):
            arm.run()
        self.assertEqual(arm._library.launch_count(), 1)

    def test_dispatcher_profiler_records_native_boundary(self):
        arm = self.arm()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            arm.run()
        self.assertIn('optimization_native::launch', [item.key for item in profile.key_averages()])

    def test_noncontiguous_input_is_rejected_without_launch(self):
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            self.arm(inputs=[torch.arange(8.).reshape(2, 4).t(), torch.ones(4)])
        self.assertEqual(ctypes.CDLL(str(self.binary)).launch_count(), 0)

    def test_failed_build_retains_diagnostics_without_manifest_success(self):
        bad = self.root / 'broken.cpp'
        bad.write_text('not valid C++')
        with self.assertRaisesRegex(RuntimeError, 'native build failed'):
            build_shared(['c++', '-shared', str(bad), '-o', str(self.root / 'bad.so')],
                source_paths=[bad], binary=self.root / 'bad.so', output=self.root / 'failed',
                source_revision='test-pin')
        self.assertTrue((self.root / 'failed/build.log').read_text())
        self.assertNotEqual(json.loads((self.root / 'failed/build.json').read_text())['exit_code'], 0)


if __name__ == '__main__':
    unittest.main()
