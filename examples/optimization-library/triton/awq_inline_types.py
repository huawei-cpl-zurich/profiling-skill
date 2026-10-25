"""Conservative compiler-only dot/carry/engine classification; no output selection."""
import re

def classify(ttir,adapter):
    dots=[line for line in ttir.splitlines() if re.search(r'\btt\.dot\b',line)]
    if len(dots)!=1:raise ValueError('one typed GEMM dot required')
    dot=dots[0];result=re.search(r'->\s*tensor<16x32x(f16|f32)>',dot)
    loops=re.findall(r'scf\.for\b[^\n]*iter_args[^\n]*->\s*\(tensor<16x32x(f16|f32)>\)',ttir)
    if not result or len(loops)!=1:raise ValueError('ambiguous dot result or loop carry type')
    dtype=result[1];carry=loops[0]
    if carry=='f32' and dtype!='f32':raise ValueError('unsupported FP32 carry with non-FP32 dot')
    if carry=='f16' and dtype=='f32':
        value=re.search(r'(%[\w]+)\s*=\s*tt\.dot',dot)
        if not value:raise ValueError('dot SSA missing')
        cast=re.search(r'(%[\w]+)\s*=\s*arith\.truncf\s+'+re.escape(value[1])+r'\s*:\s*tensor<16x32xf32>\s+to\s+tensor<16x32xf16>',ttir)
        if not cast or not re.search(r'scf\.yield\s+'+re.escape(cast[1]),ttir):raise ValueError('FP16 carry rounding not proven')
    engine=re.search(r'mix_mode\s*=\s*"(aic|mix|cube|MIX|CUBE)"',adapter)
    if not engine or not re.search(r'(linalg\.matmul|hivm\.[\w.]*m(?:atmul|mad))',adapter):raise ValueError('CUBE/MIX engine not proven')
    return dict(model='fp32' if carry=='f32' else 'fp16',dot_type=dtype,loop_carry_type=carry,engine=engine[1])
