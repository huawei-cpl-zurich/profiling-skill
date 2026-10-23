"""Prebuilt native kernels on Torch tensors, with a dispatcher-visible launch.

The bridge never allocates, compiles, validates or synchronizes in a forward.
Libraries implement the small C ABI documented in catlass_bridge.h.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import uuid


ABI_VERSION = 1
_LIBRARIES = []
_LAUNCHES = {}
_REGISTERED = False


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_shared(command, *, source_paths, binary, output, source_revision):
    """Run an explicit compiler argv and retain content-addressed build evidence.

    The caller owns the pinned include tree/toolchain. No guessed CANN paths or
    environment activation are injected here. Failed builds retain their log.
    """
    if not command or not all(isinstance(arg, str) for arg in command):
        raise ValueError('compiler command must be a nonempty string argv')
    sources = [dict(path=str(Path(p).resolve()), sha256=digest(p)) for p in source_paths]
    if not sources or not source_revision:
        raise ValueError('source identities and source revision are required')
    binary, output = Path(binary).resolve(), Path(output).resolve()
    if binary.exists():
        raise FileExistsError(binary)
    output.mkdir(parents=True, exist_ok=True)
    binary.parent.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which(command[0])
    if compiler is None:
        raise FileNotFoundError(command[0])
    compiler_identity = dict(path=str(Path(compiler).resolve()), sha256=digest(compiler))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / 'build.log').write_text(result.stdout)
    manifest = dict(command=command, exit_code=result.returncode, sources=sources,
                    source_revision=source_revision, binary=str(binary), compiler=compiler_identity)
    if result.returncode == 0 and binary.is_file():
        manifest['binary_sha256'] = digest(binary)
    (output / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError(f'native build failed ({result.returncode}); see {output / "build.log"}')
    if not binary.is_file():
        raise RuntimeError('compiler succeeded without producing the declared binary')
    return manifest


def _register(torch):
    global _REGISTERED
    if _REGISTERED:
        return
    library = torch.library.Library('optimization_native', 'DEF')
    library.define('launch(Tensor[] inputs, Tensor(a!) output, str token) -> ()')

    def launch(inputs, output, token):
        # Tensor arguments create the real dispatcher/profiler launch boundary;
        # immutable pointer storage was prepared before correctness/capture.
        _LAUNCHES[token]._launch()

    library.impl('launch', launch, 'PrivateUse1')
    library.impl('launch', launch, 'CPU')  # Enables an executable ABI test without an NPU.
    _LIBRARIES.append(library)  # Registration must outlive all prepared arms.
    _REGISTERED = True


class NativeArm:
    """PreparedArm-compatible native callable with exact source/binary identity."""
    def __init__(self, binary, inputs, output, *, sources, source_revision,
                 stream_provider, build_manifest=None, torch_module=None):
        import torch
        self.torch = torch_module or torch
        self.binary = Path(binary).resolve()
        self.sources = [str(Path(p).resolve()) for p in sources]
        if not self.sources or not source_revision:
            raise ValueError('pinned source identity is required')
        self.metadata = dict(backend='catlass-cabi-v1', source_revision=source_revision,
                             build=build_manifest or {})
        self.inputs, self.output = list(inputs), output
        tensors = [*self.inputs, self.output]
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError('native bridge requires dense contiguous tensors')
        if any(tensor.device != output.device for tensor in tensors):
            raise ValueError('native bridge tensors must share one device')
        self._pointers = (ctypes.c_void_p * len(tensors))(*(t.data_ptr() for t in tensors))
        self._stream_provider = stream_provider
        self._library = ctypes.CDLL(str(self.binary))
        self._library.catlass_bridge_abi_version.restype = ctypes.c_uint32
        if self._library.catlass_bridge_abi_version() != ABI_VERSION:
            raise ValueError('unsupported native bridge ABI')
        self._library.catlass_bridge_kernel_names.restype = ctypes.c_char_p
        self.expected_kernels = json.loads(self._library.catlass_bridge_kernel_names().decode())
        if (not isinstance(self.expected_kernels, list) or not self.expected_kernels
                or any(not isinstance(n, str) or not n for n in self.expected_kernels)
                or len(set(self.expected_kernels)) != len(self.expected_kernels)):
            raise ValueError('library must declare unique exact kernel names')
        self._function = self._library.catlass_bridge_launch
        self._function.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64, ctypes.c_void_p]
        self._function.restype = ctypes.c_int
        self._initial_identity = self.compilation_state()
        self.metadata.update(self._initial_identity)
        _register(self.torch)
        self._token = uuid.uuid4().hex
        _LAUNCHES[self._token] = self

    def compilation_state(self):
        return dict(binary_sha256=digest(self.binary),
                    source_sha256={path: digest(path) for path in self.sources})

    def _launch(self):
        # A stream is queried on every call, so switching caller streams works.
        status = self._function(self._pointers, len(self._pointers), self._stream_provider())
        if status:
            raise RuntimeError(f'native kernel launch failed with status {status}')

    def run(self):
        self.torch.ops.optimization_native.launch(self.inputs, self.output, self._token)
        return self.output

    def close(self):
        _LAUNCHES.pop(self._token, None)
