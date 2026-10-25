"""Source-bound native Catlass build; no device launches or guessed ABI structs."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,difflib
PIN='40aced927fdaaa0f3eaa9c6f5c5d859e06392f55'
BASE=Path(__file__).resolve().parent
VENDOR=BASE/'l1_a_vendor'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():return [*sorted(p for p in VENDOR.rglob('*') if p.is_file()),BASE/'l1_a_residency.cpp',Path(__file__)]


def verify_vendor():
    manifest=json.loads((VENDOR/'manifest.json').read_text())
    if manifest['pin']!=PIN:raise ValueError('wrong source pin')
    for row in manifest['files']:
        p=VENDOR/row['path'];digest=manifest['overlay']['narrow_sha256'] if row['path']=='include/catlass/gemm/block/block_mmad.hpp' else row['sha256']
        if sha(p)!=digest:raise ValueError('source mismatch '+row['path'])
    if sha(VENDOR/'block_mmad.original.hpp')!=manifest['overlay']['original_sha256']:raise ValueError('original umbrella changed')


def compile_arm(arm,output):
    if arm not in ('before','after'):raise ValueError('before/after required')
    verify_vendor();output=Path(output).resolve();manifest=output/'build.json'
    identity={str(p):sha(p) for p in sources()}
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old.get('status')=='compiled' and old['source_sha256']==identity and all(sha(output/p)==v for p,v in old['artifacts'].items()):return old
        raise RuntimeError('retained build cannot be silently replaced')
    output.mkdir(parents=True,exist_ok=True);attempt=output/('attempt-'+str(1+len(list(output.glob('attempt-*')))));attempt.mkdir();overlay=attempt/'overlay/catlass/gemm/kernel';overlay.mkdir(parents=True,exist_ok=True)
    original=(VENDOR/'include/catlass/gemm/kernel/matmul_full_loadA.hpp').read_text();modified=original
    if arm=='before':
        needle='actualBlockShape, needLoadL1);'
        if modified.count(needle)!=1:raise ValueError('source reload call mismatch')
        modified=modified.replace(needle,'actualBlockShape, true);')
    (overlay/'matmul_full_loadA.hpp').write_text(modified)
    (attempt/'reload.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='original/kernel',tofile=arm+'/kernel')))
    compiler=Path(shutil.which('bisheng') or '').resolve()
    homes=[Path(os.environ[k]).resolve() for k in ('ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME') if os.environ.get(k)]+list(compiler.parents)
    cann=next((p for p in homes if (p/'include/acl/acl.h').is_file()),None)
    if cann is None:raise RuntimeError('SDK root unavailable in named runtime')
    operators=list((cann/'include').rglob('kernel_operator.h'))
    if len(operators)!=1:raise RuntimeError('unique installed kernel_operator.h required')
    api=operators[0].resolve().parent
    includes=[attempt/'overlay',VENDOR/'include',cann/'include',api.parent,api,*[p for p in (api/'interface',api/'impl') if p.is_dir()]]
    command=[str(compiler),'-xcce','-std=c++17','-O2','-fPIC','-shared','-save-temps=obj','--cce-aicore-arch=dav-c220','-DCATLASS_ARCH=2201',*[f'-I{p}' for p in includes],str(BASE/'l1_a_residency.cpp'),'-o',str(attempt/'kernel.so')]
    record=dict(status='building',arm=arm,pin=PIN,source_sha256=identity,compiler=dict(path=str(compiler),sha256=sha(compiler)),sdk_header=dict(path=str(operators[0]),sha256=sha(operators[0])),command=command)
    (attempt/'build.json').write_text(json.dumps(record,indent=2)+'\n')
    result=subprocess.run(command,cwd=attempt,capture_output=True,text=True,timeout=480)
    (attempt/'compiler.stdout').write_text(result.stdout);(attempt/'compiler.stderr').write_text(result.stderr)
    record.update(status='compiled' if result.returncode==0 else 'compile-failed',exit=result.returncode)
    record['binary']=str(attempt/'kernel.so')
    record['artifacts']={str(p.relative_to(output)):sha(p) for p in attempt.rglob('*') if p.is_file() and p.name!='build.json'}
    (attempt/'build.json').write_text(json.dumps(record,indent=2)+'\n')
    if result.returncode:raise RuntimeError('native source compilation failed: '+result.stderr[-5000:])
    temporary=output/'compiled.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(manifest)
    return record
