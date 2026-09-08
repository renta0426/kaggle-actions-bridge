#!/usr/bin/env python3
"""One-shot private CPU executor for CMI-Flu E08 Task1.3 applicability audit."""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

REQUEST_ID="20260909-cmi-flu-strategy-e08-task13-applicability-001"
COMPETITION="cmi-flu-first-prediction-challenge"
TARGET="renta0426/cmi-flu-e08-task13-applicability-20260909-001"
TARGET_SLUG=TARGET.split("/",1)[1]
TITLE="CMI Flu E08 Task13 Applicability 20260909 001"
EXPECTED_VERSION=1
POLL_SECONDS=60
MAX_POLLS=50

def args():
 p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True); p.add_argument('--bridge-root',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); return p.parse_args()
def plain(text):return re.sub(r"\s+"," ",re.sub(r"<[^>]+>"," ",text)).casefold()
def kernel_meta(api,ref):
 owner,slug=ref.split('/',1)
 with api.build_kaggle_client() as client:
  q=ApiGetKernelRequest(); q.user_name=owner; q.kernel_slug=slug; return client.kernels.kernels_api_client.get_kernel(q).metadata
def prewrite_guard(api):
 owner,slug=TARGET.split('/',1)
 if owner!='renta0426' or slug!=TARGET_SLUG or TITLE!='CMI Flu E08 Task13 Applicability 20260909 001':raise RuntimeError('E08 target identity contract changed')
 try: discovered=api.kernels_list(user=owner,search=TARGET_SLUG,page_size=20) or []
 except Exception as e:raise RuntimeError('E08 duplicate sentinel unavailable') from e
 if any(str(getattr(x,'ref',''))==TARGET for x in discovered):raise RuntimeError('E08 duplicate sentinel found exact target; write refused')
def live_rules(api):
 pages=api.competition_list_pages(COMPETITION) or []; content={}
 for page in pages:
  d=page.to_dict() if hasattr(page,'to_dict') else dict(page); content[str(d.get('name') or '').strip().lower()]=str(d.get('content') or '')
 if 'evaluation' not in content or 'mean spearman correlation' not in plain(content['evaluation']):raise RuntimeError('E08 live evaluation guard changed')
def active_counts(api):
 active={'cpu':0,'gpu':0,'tpu':0,'unknown':0}
 for item in (api.kernels_list(user='renta0426',sort_by='dateRun',page_size=10) or [])[:10]:
  ref=str(getattr(item,'ref',''))
  if not ref:continue
  try: status=str(getattr(api.kernels_status(ref),'status','')).upper()
  except Exception:active['unknown']+=1;continue
  if not any(t in status for t in ('RUNNING','QUEUED','PENDING')):continue
  try:
   m=kernel_meta(api,ref); kind='tpu' if bool(getattr(m,'enable_tpu',False)) else ('gpu' if bool(getattr(m,'enable_gpu',False)) else 'cpu')
  except Exception:kind='unknown'
  active[kind]+=1
 return active
def digest_text(v):return hashlib.sha256(v.encode('utf-8',errors='replace')).hexdigest()
def receipt(phase,runtime,attempted,rc=None,stdout='',stderr='',confirmed=False):
 p={'request_id':REQUEST_ID,'phase':phase,'target_sha256':hashlib.sha256(TARGET.encode()).hexdigest(),'runtime_sha256':hashlib.sha256(runtime.read_bytes()).hexdigest(),'write_attempted':bool(attempted),'cli_return_code':rc,'stdout_bytes':len(stdout.encode(errors='replace')),'stdout_sha256':digest_text(stdout),'stderr_bytes':len(stderr.encode(errors='replace')),'stderr_sha256':digest_text(stderr),'exact_identity_confirmed':bool(confirmed)}; print('CMI_FLU_E08_WRITE_RECEIPT '+json.dumps(p,sort_keys=True))
def push(api,runtime,work):
 k=work/'kernel'; k.mkdir(parents=True); shutil.copyfile(runtime,k/'script.py'); metadata={'id':TARGET,'title':TITLE,'code_file':'script.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,'enable_internet':False,'competition_sources':[COMPETITION]}; (k/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
 receipt('before_write',runtime,False); c=subprocess.run(['kaggle','kernels','push','-p',str(k)],capture_output=True,text=True,timeout=180,check=False); receipt('after_cli_write',runtime,True,int(c.returncode),c.stdout,c.stderr)
 if c.returncode:raise RuntimeError(f'kaggle push failed rc={c.returncode}')
 try:m=kernel_meta(api,TARGET)
 except Exception as e:
  code=hashlib.sha256(f'{type(e).__name__}:{e}'.encode(errors='replace')).hexdigest()[:20]; print(f'CMI_FLU_E08_WRITE_AMBIGUOUS exception_type={type(e).__name__} error_code={code}'); raise RuntimeError('E08 push acknowledged but exact metadata unconfirmed') from e
 if str(getattr(m,'ref',''))!=TARGET or not bool(getattr(m,'is_private',False)) or bool(getattr(m,'enable_gpu',False)) or bool(getattr(m,'enable_tpu',False)) or bool(getattr(m,'enable_internet',False)):raise RuntimeError('pushed E08 metadata contract mismatch')
 if int(getattr(m,'current_version_number',0) or 0)!=1:raise RuntimeError('E08 direct version mismatch')
 receipt('exact_identity_confirmed',runtime,True,int(c.returncode),c.stdout,c.stderr,True)
def wait(api):
 for _ in range(MAX_POLLS):
  m=kernel_meta(api,TARGET)
  if int(getattr(m,'current_version_number',0) or 0)!=1:raise RuntimeError('E08 version changed during execution')
  status=str(getattr(api.kernels_status(TARGET),'status','')).upper()
  if 'COMPLETE' in status:return status
  if any(t in status for t in ('ERROR','CANCEL','FAIL')):raise RuntimeError('E08 remote failed')
  if not any(t in status for t in ('RUNNING','QUEUED','PENDING')):raise RuntimeError(f'unknown E08 status:{status}')
  time.sleep(POLL_SECONDS)
 raise RuntimeError('E08 polling bound exceeded')
def main():
 a=args(); token=os.environ.get('KAGGLE_API_TOKEN','')
 if not token.startswith('KGAT_'):raise SystemExit('KAGGLE_API_TOKEN contract failed')
 api=KaggleApi(); api.authenticate(); live_rules(api); prewrite_guard(api); active=active_counts(api)
 if active['unknown'] or active['cpu']>=1:raise SystemExit('CPU admission closed or resource classification unknown')
 work=a.output_dir.parent/'execution'; work.mkdir(parents=True,exist_ok=False)
 try:
  push(api,a.runtime.resolve(),work); status=wait(api); m=kernel_meta(api,TARGET)
  if int(getattr(m,'current_version_number',0) or 0)!=1:raise RuntimeError('E08 version changed before output read')
  reader=a.bridge_root.resolve()/'scripts'/'kaggle_current_output_read.py'; c=subprocess.run([sys.executable,str(reader),'--kernel',TARGET,'--expected-version','1','--allow-file','bridge-result.json:1048576','--allow-file','metrics.json:8388608','--allow-file','summary.md:1048576','--output-dir',str(a.output_dir.resolve())],timeout=240,check=False)
  if c.returncode:raise RuntimeError('E08 aggregate output read failed')
  if int(getattr(kernel_meta(api,TARGET),'current_version_number',0) or 0)!=1:raise RuntimeError('E08 version changed after output read')
  print(f'CMI_FLU_E08_REMOTE PASS status={status} version=1')
 finally:shutil.rmtree(work,ignore_errors=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
