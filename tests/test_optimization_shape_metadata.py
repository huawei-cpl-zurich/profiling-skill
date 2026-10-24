import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples/optimization-library/catlass'))
from shape_metadata_compile import select_metadata, classify, device_sections, compile_screen


class ShapeMetadataTests(unittest.TestCase):
    def test_selection_and_guard_before_build_access(self):
        self.assertEqual(select_metadata(), {'before': [], 'after': ['-DW006_ZERO_SHAPE_INFO=1']})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'must-not-exist'
            with patch('shape_metadata_compile.compile_pair') as compiler:
                with self.assertRaisesRegex(ValueError, 'ShapeInfo consumers'):
                    compile_screen('/nonexistent-dependency', output, request_shape_info=True)
                compiler.assert_not_called()
            self.assertFalse(output.exists())
        with self.assertRaises(ValueError):
            select_metadata('false')

    def test_noop_and_conditional_admission(self):
        before = {'.text': {'sha256': 'a'}, '.ascend.stack.size.record': {'size': 8}}
        same = {'before': 'a'*64, 'after': 'a'*64}
        different = {'before': 'a'*64, 'after': 'b'*64}
        self.assertEqual(classify(8, 0, before, before, device_object_sha256=same)['outcome'], 'compile-noop')
        changed = {'.text': {'sha256': 'b'}}
        result = classify(8, 0, before, changed, device_object_sha256=different)
        self.assertEqual(result['outcome'], 'physical-change-requires-runtime')
        self.assertEqual(list(result['cases'].values()), ['unattempted']*4)
        self.assertEqual((result['scientific_oracles'], result['npu_api_forwards'], result['profiles']), (0,0,0))
        self.assertEqual(classify(0, 0, before, changed, device_object_sha256=different)['outcome'],
                         'inconclusive-device-equivalence')
        with self.assertRaises(ValueError):
            classify(8, 1, before, changed, device_object_sha256=different)
        with self.assertRaises(ValueError):
            classify(8, 0, {}, {}, device_object_sha256=different)
        with self.assertRaises(ValueError):
            classify(8, 0, before, before, device_object_sha256={})
        with self.assertRaises(ValueError):
            classify(8, 0, before, changed, device_object_sha256=same)

    @unittest.skipUnless(shutil.which('cc'), 'host C compiler needed')
    def test_equal_raw_relocations_with_different_targets_are_inconclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'kernel.c'  # Keep the ELF source-name symbol identical.
            evidence, identities = {}, {}
            for arm, external in (('before','first'),('after','second')):
                source.write_text('extern int '+external+'; int kernel(void) { return '+external+'; }\n'
                    '__attribute__((section(".ascend.stack.size.record"))) const long stack = 32;\n')
                binary = root/(arm+'.o')
                subprocess.run(['cc','-O2','-c',str(source),'-o',str(binary)],check=True,capture_output=True)
                evidence[arm] = device_sections(binary)
                identities[arm] = hashlib.sha256(binary.read_bytes()).hexdigest()
            self.assertIn('.rela.text',evidence['before'])
            self.assertEqual(evidence['before'],evidence['after'])
            self.assertNotEqual(identities['before'],identities['after'])
            result = classify(8,0,evidence['before'],evidence['after'],device_object_sha256=identities)
            self.assertEqual(result['outcome'],'inconclusive-device-equivalence')
            self.assertFalse(result['device_elf_equal'])
            self.assertEqual(result['scientific_oracles'],0)
            self.assertEqual(result['npu_api_forwards'],0)
            self.assertEqual(result['profiles'],0)

    @unittest.skipUnless(shutil.which('cc'), 'host C compiler needed')
    def test_actual_elf_code_and_resource_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def build(label, answer, stack):
                source = root/(label+'.c')
                source.write_text('int kernel(void) { return '+str(answer)+'; }\n'
                    '__attribute__((section(".ascend.stack.size.record"))) const long stack = '+str(stack)+';\n')
                binary = source.with_suffix('.o')
                subprocess.run(['cc','-O2','-c',str(source),'-o',str(binary)], check=True, capture_output=True)
                return device_sections(binary)
            first = build('first', 1, 32)
            renamed = build('renamed', 1, 32)
            self.assertEqual(first, renamed)
            self.assertNotEqual(first['.text'], build('code', 2, 32)['.text'])
            self.assertNotEqual(first['.ascend.stack.size.record'], build('stack', 1, 64)['.ascend.stack.size.record'])
            bad = root/'bad.o'; bad.write_bytes(b'not ELF')
            with self.assertRaises(ValueError):
                device_sections(bad)

    @unittest.skipUnless(shutil.which('cc'), 'host compiler needed')
    def test_complete_screen_retains_evidence_and_restores_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'fake.c'
            source.write_text('int kernel(void) { return 1; }\n'
                '__attribute__((section(".ascend.stack.size.record"))) const long stack = 32;\n')
            binary = root/'fake.o'
            subprocess.run(['cc','-c',str(source),'-o',str(binary)],check=True,capture_output=True)
            initial = Path.cwd()
            def compile_fake(catlass, destination, *, source, variants):
                self.assertTrue(catlass.is_absolute())
                arm = next(iter(variants))
                shutil.copyfile(binary,destination/'fake-cube.o')
                (destination/'fake-host-native.ccei').write_text(
                    'catlass_shape_dimension() { return '+('8' if arm=='before' else '0')+'; }')
            with patch('shape_metadata_compile.compile_pair',side_effect=compile_fake):
                result = compile_screen('relative-dependency',root/'result')
            self.assertEqual(Path.cwd(),initial)
            self.assertEqual(result['outcome'],'compile-noop')
            saved = json.loads((root/'result/screen.json').read_text())
            self.assertEqual(saved,result)
            self.assertTrue((root/'result/before/device-sections.json').is_file())
            with patch('shape_metadata_compile.compile_pair',side_effect=RuntimeError('compiler failed')):
                with self.assertRaisesRegex(RuntimeError,'compiler failed'):
                    compile_screen('relative-dependency',root/'failed')
            self.assertEqual(Path.cwd(),initial)

    @unittest.skipUnless(shutil.which('c++'), 'host C++ compiler needed')
    def test_native_host_pointer_abi_and_independent_states(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'abi.cpp'
            source.write_text('#include "shape_metadata_abi.h"\n'
                'static ShapeMetadata::State state;\n'
                'extern "C" int configure(unsigned m,unsigned n,unsigned k,unsigned c) { return state.configure(m,n,k,c); }\n'
                'extern "C" int validate(void** p,unsigned long n) { return state.validate(p,n); }\n')
            images=[]
            for name in ('first','second'):
                binary = root/(name+'.so')
                subprocess.run(['c++','-shared','-fPIC','-I'+str(ROOT/'examples/optimization-library/catlass'),
                    str(source),'-o',str(binary)],check=True,capture_output=True)
                image = ctypes.CDLL(str(binary))
                image.configure.argtypes = [ctypes.c_uint32]*4
                image.validate.argtypes = [ctypes.POINTER(ctypes.c_void_p),ctypes.c_uint64]
                images.append(image)
            pointers=(ctypes.c_void_p*3)(1,2,3)
            first,second=images
            self.assertEqual(first.configure(128,256,32,24),0)
            self.assertEqual(first.validate(pointers,3),0)
            self.assertEqual(second.validate(pointers,3),2)
            self.assertEqual(first.validate(None,3),2)
            self.assertEqual(first.validate(pointers,2),2)
            self.assertEqual(first.configure(0,256,32,24),1)
            self.assertEqual(first.configure(128,256,272,24),1)
            self.assertEqual(first.configure(512,1024,256,1),1)
            self.assertEqual(first.configure(0xfffffff0,0xfffffff0,16,24),1)

    def test_public_cli_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'out'
            run = subprocess.run([sys.executable, str(ROOT/'examples/optimization-library/catlass/shape_metadata_compile.py'),
                '--catlass', '/absent', '--output', str(output), '--request-shape-info'], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('ShapeInfo consumers', run.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
