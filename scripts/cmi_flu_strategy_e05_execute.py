#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
REQUEST_ID="20260907-cmi-flu-strategy-e05-hai-donor-strain-001"; COMPETITION="cmi-flu-first-prediction-challenge"; TARGET="renta0426/cmi-flu-e05-hai-donor-strain-20260907-001"; TARGET_SLUG=TARGET.split('/',1)[1]; TITLE="CMI Flu E05 HAI Donor Strain 20260907 001"; EXPECTED_VERSION=1; POLL_SECONDS=60; MAX_POLLS=60
REF_SHA={"strain_sequences.csv":"63eb462620d6dc710547b390364194a6073c4fdb3bc811794cc2ffab6da65887","vaccine_strains_per_season.txt":"8f6c7116f37f29df0bb21d6049d82fa28b4e42b2d10ed9394a1ae6f926bd9f35"}
def args():
 p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True); p.add_argument('--bridge-root',type=Path,required=True); p.add_argument('--reference-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); return p.parse_args()
def plain(t): return re.sub(r'\s+',' ',re.sub(r'<[^>]+>',' ',t)).casefold()
def kernel_meta(api,ref):
 owner,slug=ref.split('/',1)
 with api.build_kaggle_client() as client:
  q=ApiGetKernelRequest(); q.user_name=owner; q.kernel_slug=slug; return client.kernels.kernels_api_client.get_kernel(q).metadata
def live_rules(api):
 pages=api.competition_list_pages(COMPETITION) or []; c={}
 for page in pages:
  d=page.to_dict() if hasattr(page,'to_dict') else dict(page); n=str(d.get('name') or '').strip().lower(); c[n]=str(d.get('content') or '')
 if 'rules' not in c or 'evaluation' not in c or not any('data' in k for k in c): raise RuntimeError('live Competition pages unavailable')
 if 'mean spearman correlation' not in plain(c['evaluation']): raise RuntimeError('evaluation guard changed')
def active_counts(api):
 active={'cpu':0,'gpu':0,'tpu':0,'unknown':0}
 for item in (api.kernels_list(user='renta0426',sort_by='dateRun',page_size=10) or [])[:10]:
  ref=str(getattr(item,'ref',''))
  if not ref: continue
  try: status=str(getattr(api.kernels_status(ref),'status','')).upper()
  except Exception: active['unknown']+=1; continue
  if not any(x in status for x in ('RUNNING','QUEUED','PENDING')): continue
  try:
   m=kernel_meta(api,ref); kind='tpu' if bool(getattr(m,'enable_tpu',False)) else ('gpu' if bool(getattr(m,'enable_gpu',False)) else 'cpu')
  except Exception: kind='unknown'
  active[kind]+=1
 return active
def refuse_duplicate(api):
 refs={str(getattr(x,'ref','')) for x in (api.kernels_list(user='renta0426',search=TARGET_SLUG,page_size=10) or [])}
 if TARGET in refs: raise RuntimeError('E05 target already exists; duplicate write refused')
def push(api,runtime,refs,work):
 import hashlib
 kd=work/'kernel'; kd.mkdir(parents=True)
 shutil.copyfile(runtime,kd/'script.py')
 for name,digest in REF_SHA.items():
  src=refs/name
  if hashlib.sha256(src.read_bytes()).hexdigest()!=digest: raise RuntimeError(f'reference hash mismatch:{name}')
  shutil.copyfile(src,kd/name)
 meta={'id':TARGET,'title':TITLE,'code_file':'script.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,'enable_internet':False,'competition_sources':[COMPETITION]}; (kd/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 cp=subprocess.run(['kaggle','kernels','push','-p',str(kd)],capture_output=True,text=True,timeout=180)
 if cp.returncode: raise RuntimeError(f'kaggle push failed rc={cp.returncode}')
 m=kernel_meta(api,TARGET)
 if str(getattr(m,'ref',''))!=TARGET or not bool(getattr(m,'is_private',False)) or bool(getattr(m,'enable_gpu',False)) or bool(getattr(m,'enable_tpu',False)) or bool(getattr(m,'enable_internet',False)): raise RuntimeError('pushed E05 metadata contract mismatch')
 if int(getattr(m,'current_version_number',0) or 0)!=EXPECTED_VERSION: raise RuntimeError('E05 version mismatch')
def wait(api):
 for _ in range(MAX_POLLS):
  s=str(getattr(api.kernels_status(TARGET),'status','')).upper()
  if 'COMPLETE' in s:return s
  if any(x in s for x in ('ERROR','CANCEL','FAIL')): raise RuntimeError('E05 remote failed')
  if not any(x in s for x in ('RUNNING','QUEUED','PENDING')): raise RuntimeError(f'unknown E05 status:{s}')
  time.sleep(POLL_SECONDS)
 raise RuntimeError('E05 polling bound exceeded')
def main():
 a=args(); token=os.environ.get('KAGGLE_API_TOKEN','')
 if not token.startswith('KGAT_'): raise SystemExit('KAGGLE_API_TOKEN contract failed')
 api=KaggleApi(); api.authenticate(); live_rules(api); refuse_duplicate(api); active=active_counts(api)
 if active['unknown'] or active['cpu']>=1: raise SystemExit('CPU admission closed or resource classification unknown')
 work=a.output_dir.parent/'execution'; work.mkdir(parents=True,exist_ok=False)
 try:
  push(api,a.runtime.resolve(),a.reference_dir.resolve(),work); status=wait(api); reader=a.bridge_root.resolve()/'scripts/kaggle_current_output_read.py'; cp=subprocess.run([sys.executable,str(reader),'--kernel',TARGET,'--expected-version','1','--allow-file','bridge-result.json:1048576','--allow-file','metrics.json:12582912','--allow-file','summary.md:1048576','--output-dir',str(a.output_dir.resolve())],timeout=240)
  if cp.returncode: raise RuntimeError('aggregate output read failed')
  print(f'CMI_FLU_E05_REMOTE PASS status={status} version=1')
 finally: shutil.rmtree(work,ignore_errors=True)
 return 0
if __name__=='__main__': raise SystemExit(main())
