"""Extract the two exact postprocess bodies; tiling is a launch parameter."""
import ast,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
PIN='f00c89a24901c3a6099c784bfc9a31d51818929216fde6cbf82e5e3c851e2573'
NAMES=('chunk_kda_bwd_kernel_dA_post_right','chunk_kda_bwd_kernel_dA_post_left')


def generate():
    source=(HERE/'vendor/chunk_kda_bwd.py').read_bytes()
    if hashlib.sha256(source).hexdigest()!=PIN:raise ValueError('pinned source changed')
    text=source.decode();lines=text.splitlines(keepends=True);nodes={n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    out='# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang\n# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.\n# Generated exact bodies; complete Apache2/MIT notices in vendor/.\nimport triton\nimport triton.language as tl\n\n'
    for name in NAMES:
        n=nodes[name];body=''.join(lines[n.lineno-1:n.end_lineno])
        # IS_VARLEN=False is explicit in both prepared launches; retain original JIT policy.
        out+="@triton.jit(do_not_specialize=['T'])\n"+body+'\n\n'
    for node in ast.parse(out).body:
        if isinstance(node,ast.FunctionDef):
            actual=ast.dump(ast.Module(body=node.body,type_ignores=[]),include_attributes=False)
            original=ast.dump(ast.Module(body=nodes[node.name].body,type_ignores=[]),include_attributes=False)
            if actual!=original:raise ValueError('extraction changed math')
    return out


if __name__=='__main__':(HERE/'kda_da_generated.py').write_text(generate())
