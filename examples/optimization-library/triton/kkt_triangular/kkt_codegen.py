"""Extract pinned featurewise source; preserve exact bodies except declared changes."""
import ast,copy,difflib
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'vendor/src/triton_ascend_kernels/attention/fla/chunk_scaled_dot_kkt_fla.py'


def generate():
    tree=ast.parse(SOURCE.read_text());functions={x.name:x for x in tree.body if isinstance(x,ast.FunctionDef)};emitted=[]
    for old,new in [('chunk_scaled_dot_kkt_fwd_kernel_intra_sub_inter','kkt_lower_inter'),('chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra','kkt_diagonal')]:
        f=copy.deepcopy(functions[old]);f.name=new;f.decorator_list=[ast.parse('@triton.jit(do_not_specialize=["T"])\ndef placeholder():pass').body[0].decorator_list[0]]
        text=ast.unparse(f)
        if new=='kkt_lower_inter':
            assert text.count('tl.dot(b_k, b_kt)')==1
            text=text.replace('tl.dot(b_k, b_kt)','tl.dot(b_k, b_kt, allow_tf32=False)')
            before=text.replace('kkt_lower_inter','kkt_full_inter').replace('i_i > i_j','i_i != i_j')
            assert before.count('b_A *= b_beta[:, None]')==1
            before=before.replace('b_A *= b_beta[:, None]','b_A *= b_beta[:, None]\n            b_A = tl.where(i_i > i_j, b_A, 0.0)')
            emitted.append(before)
        emitted.append(text)
    zero='''@triton.jit
def kkt_zero(A,N:tl.constexpr,BLOCK:tl.constexpr):
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    tl.store(A+x,0.,x<N)
'''
    return '"""Generated source-derived featurewise KKT; see frozen P04 contract."""\nimport triton\nimport triton.language as tl\n\n'+zero+'\n\n'+'\n\n'.join(emitted)+'\n'


if __name__=='__main__':(HERE/'kkt_generated.py').write_text(generate())
