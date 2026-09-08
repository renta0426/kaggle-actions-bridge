#!/usr/bin/env python3
"""Build an exact, aggregate-only E04 Task1.3 Kaggle runtime."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import pathlib
import ssl
import tarfile
import urllib.request
import zipfile
from pathlib import Path

REQUEST_ID = "20260908-cmi-flu-strategy-e04-task13-rescue-001"
COMPETITION = "cmi-flu-first-prediction-challenge"
TARGET_KERNEL = "renta0426/cmi-flu-e04-task13-rescue-20260908-001"
SCIENCE_REPOSITORY = "renta0426/CMI-Flu-Invited-Prediction-Challenge"
SCIENCE_COMMIT = "6e4d786cddc7b2d02dc4671d1c005db551d19ad6"
REQUEST_PATH = "requests/cmi-flu-strategy-e04-task13-rescue-001.json"
CONFIG_REL = "configs/baseline_b021_robust.yaml"
PINNED_BLOBS = {
    "src/cmi_flu/strategy_e04.py": "73c112a8bb3b1ee7a6dd5bfbd5645a945bea6ebb",
    "src/cmi_flu/strategy_e04_contract.py": "3982541febfb4641fbf895438ae1a465bdbb3d5e",
    "src/cmi_flu/strategy_e04_synthetic.py": "436b6a971622915cc5b335060f9a850feb6108fd",
    "src/cmi_flu/task13_harmonization.py": "5c6725dc757a5ba9dd21289b1c4f09997e1afdb8",
    CONFIG_REL: "170d3211e2795c0730e481056c7bb068accf97c9",
}
FIXED_ZIP_TIME = (2020, 1, 1, 0, 0, 0)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def validate_request(root: Path) -> None:
    r = json.loads((root / REQUEST_PATH).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "request_id": REQUEST_ID,
        "competition": COMPETITION,
        "operation": "kernel_run_and_current_output_read",
        "target": TARGET_KERNEL,
        "science_repository": SCIENCE_REPOSITORY,
        "science_source_commit": SCIENCE_COMMIT,
        "science_transport": "github_exact_commit_archive_with_blob_verification",
        "expected_kernel_version": 1,
        "enable_internet": False,
        "competition_submission_attempted": False,
        "leaderboard_used_for_selection": False,
        "automatic_compute_retries": 0,
    }
    for key, value in expected.items():
        if r.get(key) != value:
            raise SystemExit(f"E04 request mismatch:{key}")
    if r.get("pinned_git_blobs") != PINNED_BLOBS:
        raise SystemExit("E04 pinned blob contract mismatch")
    if r.get("resource") != {"accelerator":"cpu","expected_runtime_minutes":30,"hard_timeout_minutes":60,"max_active_runs":1}:
        raise SystemExit("E04 resource contract mismatch")
    if r.get("allowed_output_paths") != ["bridge-result.json", "metrics.json", "summary.md"]:
        raise SystemExit("E04 output allowlist mismatch")


def fetch_science_tree() -> dict[str, bytes]:
    url = f"https://api.github.com/repos/{SCIENCE_REPOSITORY}/tarball/{SCIENCE_COMMIT}"
    req = urllib.request.Request(url, headers={"User-Agent":"kaggle-actions-bridge-e04/1"})
    with urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context()) as response:
        data = response.read(80_000_001)
    if len(data) > 80_000_000:
        raise SystemExit("science archive exceeds 80 MB compressed guard")
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        prefixes = {m.name.split("/", 1)[0] for m in members}
        if len(prefixes) != 1:
            raise SystemExit("science archive layout changed")
        prefix = next(iter(prefixes)) + "/"
        for member in members:
            if not member.name.startswith(prefix):
                raise SystemExit("science archive prefix mismatch")
            rel = pathlib.PurePosixPath(member.name[len(prefix):])
            if ".." in rel.parts:
                raise SystemExit("unsafe science archive path")
            name = rel.as_posix()
            if name == CONFIG_REL or (name.startswith("src/cmi_flu/") and name.endswith(".py")):
                stream = archive.extractfile(member)
                if stream is None:
                    raise SystemExit(f"unreadable science member:{name}")
                out[name] = stream.read()
    if CONFIG_REL not in out or "src/cmi_flu/__init__.py" not in out:
        raise SystemExit("science archive missing package/config")
    for path, expected in PINNED_BLOBS.items():
        if path not in out or git_blob_sha(out[path]) != expected:
            raise SystemExit(f"science exact blob mismatch:{path}")
    return out


def package_zip(tree: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source_path in sorted(k for k in tree if k.startswith("src/cmi_flu/") and k.endswith(".py")):
            relative = source_path[len("src/"):]
            info = zipfile.ZipInfo(relative, date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, tree[source_path])
    return buffer.getvalue()


def build_runtime(tree: dict[str, bytes]) -> str:
    bundle = package_zip(tree)
    package_sha = hashlib.sha256(bundle).hexdigest()
    config_text = tree[CONFIG_REL].decode("utf-8")
    package_b64 = base64.b64encode(bundle).decode("ascii")
    pinned = json.dumps(PINNED_BLOBS, sort_keys=True)
    return f'''#!/usr/bin/env python3
"""Frozen CMI-Flu E04 Task1.3 runtime. Aggregate outputs only; no submission."""
from __future__ import annotations
import argparse, base64, hashlib, io, json, math, re, shutil, sys, tempfile, traceback, zipfile
from pathlib import Path
REQUEST_ID={REQUEST_ID!r}; COMPETITION={COMPETITION!r}; SCIENCE_COMMIT={SCIENCE_COMMIT!r}; TARGET_KERNEL={TARGET_KERNEL!r}
PACKAGE_SHA256={package_sha!r}; PINNED_GIT_BLOBS={pinned}; PACKAGE_B64={package_b64!r}; CONFIG_TEXT={config_text!r}
CORE_FILES=("participants.tsv","investigations_260821.tsv","publicData_cytokine.tsv","publicData_ex_vivo_flow.tsv","publicData_serology_260821.tsv","2025LJI_aim.tsv","2025LJI_cytokine.tsv","2025LJI_ex_vivo_flow.tsv","2025LJI_serology.tsv","sample_submission_part1.csv","md5sum")
BANNED=('"participant_id"','"subject_group"','"row_index"','"oof_predictions"','"challenge_predictions"','"predictions"')
class BridgeContractError(RuntimeError): pass

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument("--self-test",action="store_true"); p.add_argument("--synthetic",action="store_true"); p.add_argument("--input-dir",type=Path); p.add_argument("--output-dir",type=Path,default=Path("/kaggle/working")); return p.parse_args()
def package_bytes():
 data=base64.b64decode(PACKAGE_B64.encode("ascii"),validate=True)
 if hashlib.sha256(data).hexdigest()!=PACKAGE_SHA256: raise BridgeContractError("package_sha256_mismatch")
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  if z.testzip() is not None: raise BridgeContractError("package_corrupt")
  names=set(z.namelist())
 for name in ("cmi_flu/strategy_e04.py","cmi_flu/strategy_e04_contract.py","cmi_flu/strategy_e04_synthetic.py","cmi_flu/task13_harmonization.py"):
  if name not in names: raise BridgeContractError(f"package_missing:{{name}}")
 return data
def locate_competition_data(explicit):
 candidates=[]
 if explicit is not None: candidates.append(explicit)
 candidates.append(Path("/kaggle/input")/COMPETITION); root=Path("/kaggle/input")
 if root.is_dir(): candidates.extend(sorted(p for p in root.iterdir() if p.is_dir()))
 seen=[]
 for p in candidates:
  try:r=p.expanduser().resolve()
  except OSError:continue
  if r in seen or not r.is_dir():continue
  seen.append(r)
  if all((r/n).is_file() for n in CORE_FILES):return r
 raise BridgeContractError("competition_data_mount_not_found")
def _vaccine_flag(value):
 if value is None:return False
 try:
  import pandas as pd
  if pd.isna(value):return False
 except Exception:pass
 return str(value).strip().casefold() in {{"1","1.0","true","yes","y"}}
def _canonical_strain(value):
 raw="" if value is None else str(value).strip(); raw=re.sub(r"(?i)_cell$","_MDCK",raw); raw=re.sub(r"(?i)\\s+cell$","_MDCK",raw); return raw
def derive_reference_files(input_dir,reference_dir):
 import pandas as pd
 reference_dir.mkdir(parents=True,exist_ok=True); s=pd.read_csv(input_dir/"2025LJI_serology.tsv",sep="\\t",usecols=["assay","virus_strain","virus_in_vaccine"],low_memory=False)
 h=s.loc[s["assay"].fillna("").astype(str).str.strip().str.casefold().eq("hai")].copy(); h["virus_strain"]=h["virus_strain"].map(_canonical_strain)
 challenge=tuple(sorted(x for x in h["virus_strain"].dropna().unique() if x)); vaccine=tuple(sorted(x for x in h.loc[h["virus_in_vaccine"].map(_vaccine_flag),"virus_strain"].dropna().unique() if x))
 if len(challenge)!=12 or len(vaccine)!=3 or not set(vaccine).issubset(challenge):raise BridgeContractError("derived_hai_reference_contract_changed")
 (reference_dir/"all_challenge_virus_strains.txt").write_text("\\n".join(challenge)+"\\n"); (reference_dir/"vaccine_strains_2025.txt").write_text("\\n".join(vaccine)+"\\n")
 for n,c in {{"cytokine_name_map.csv":"source,target\\n","flow_name_revised.csv":"source,target\\n","hai_map.csv":"source,target\\n","strain_sequences.csv":"virus_strain,sequence\\n","vaccine_strains_per_season.txt":"# unused by E04\\n"}}.items():(reference_dir/n).write_text(c)
def materialize(input_dir=None):
 root=Path(tempfile.mkdtemp(prefix="cmi-flu-e04-",dir="/tmp")); bundle=root/"cmi_flu_bundle.zip"; bundle.write_bytes(package_bytes()); sys.path.insert(0,str(bundle)); cfgdir=root/"configs"; cfgdir.mkdir(); cfg=cfgdir/"baseline_b021_robust.yaml"; cfg.write_text(CONFIG_TEXT)
 if input_dir is not None:
  data=root/"data"; data.mkdir(); (data/"raw").symlink_to(input_dir,target_is_directory=True); ref=root/"external"/"google-drive"/"challenge-resources"/"reference_files"; derive_reference_files(input_dir,ref)
 return root,cfg
def json_safe(v):
 import numpy as np
 if isinstance(v,dict):return {{str(k):json_safe(x) for k,x in v.items()}}
 if isinstance(v,(list,tuple)):return [json_safe(x) for x in v]
 if isinstance(v,np.generic):v=v.item()
 if isinstance(v,float) and not math.isfinite(v):return None
 return v
def validate_result(result,synthetic=False):
 if result.get("schema_version")!=1 or result.get("experiment")!="strategy_v2_e04_task13_anchor_preserving_rescue" or result.get("task")!="Task1.3":raise BridgeContractError("experiment_identity_mismatch")
 if result.get("competition_submission_attempted") is not False or result.get("leaderboard_used_for_selection") is not False or result.get("automatic_compute_retries")!=0:raise BridgeContractError("selection_submission_contract_mismatch")
 expected={{"strict_b21":"pls_1","anchor":"flow_rank__Antibody-secreting_cells_(ASC)","historical_rank":"enet_a0.1_l0.5","residual_alpha":10.0,"residual_shrinkage":0.25,"residual_score_cap":0.05,"proxy_weights":[0.25,1.0]}}
 if (result.get("frozen_conditions") or {{}})!=expected:raise BridgeContractError("frozen_condition_mismatch")
 if not isinstance(result.get("measurement_audit"),dict):raise BridgeContractError("measurement_audit_missing")
 if not synthetic and result.get("status") not in {{"complete","reference_mismatch_review_required"}}:raise BridgeContractError("real_status_invalid")
 if not synthetic and (result.get("frozen_control_reproduction") or {{}}).get("all_pass") is not True:raise BridgeContractError("frozen_control_reproduction_failed")
 text=json.dumps(result,sort_keys=True,ensure_ascii=False)
 if any(token in text for token in BANNED):raise BridgeContractError("aggregate_privacy_contract_failed")
def render_summary(result):
 lines=["# CMI-Flu strategy E04 Task1.3 gate rescue","","Aggregate-only fixed-condition evaluation; no Competition submission was attempted.","",f"- science commit: `{{SCIENCE_COMMIT}}`",f"- status: `{{result.get('status')}}`",f"- incumbent changed: `{{result.get('incumbent_changed')}}`",f"- audit keys: `{{len(result.get('measurement_audit') or {{}})}}`"]
 for group in ("strict_cv","historical_loso","strict_cv_plus_proxy"):
  for name,p in (result.get(group) or {{}}).items():
   m=p.get("metrics") or {{}}; pa=p.get("paired_vs_anchor") or {{}}; lines.append(f"- {{group}}/{{name}} study_equal={{m.get('study_equal_spearman')}} delta_anchor={{pa.get('paired_study_mean_delta')}}")
 for name,p in (result.get("challenge") or {{}}).items():
  if name.startswith("proxy_weight_"):lines.append(f"- challenge/{{name}} decision={{p.get('decision')}} bridge_sources={{p.get('strict_bridge_source_subjects')}}")
 return "\\n".join(lines)+"\\n"
def write_outputs(result,output_dir,synthetic=False):
 output_dir.mkdir(parents=True,exist_ok=True); result=json_safe(dict(result)); validate_result(result,synthetic=synthetic); metrics=output_dir/"metrics.json"; summary=output_dir/"summary.md"; bridge=output_dir/"bridge-result.json"; metrics.write_text(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False)+"\\n"); summary.write_text(render_summary(result)); receipt={{"schema_version":1,"request_id":REQUEST_ID,"science_commit":SCIENCE_COMMIT,"science_package_sha256":PACKAGE_SHA256,"pinned_git_blobs":PINNED_GIT_BLOBS,"metrics_sha256":hashlib.sha256(metrics.read_bytes()).hexdigest(),"summary_sha256":hashlib.sha256(summary.read_bytes()).hexdigest(),"competition_submission_attempted":False,"leaderboard_used_for_selection":False,"contains_participant_identifiers":False,"contains_row_level_predictions":False,"synthetic":bool(synthetic)}}; bridge.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\\n")
def self_test():
 data=package_bytes(); print(f"CMI_FLU_E04_RUNTIME_SELF_TEST PASS request_id={{REQUEST_ID}} science_commit={{SCIENCE_COMMIT}} package_bytes={{len(data)}} package_sha256={{PACKAGE_SHA256}}"); return 0
def execute(synthetic=False,input_dir=None,output_dir=Path("/kaggle/working")):
 root=None
 try:
  source=None if synthetic else locate_competition_data(input_dir); root,cfg=materialize(source); from cmi_flu.configuration import load_baseline_config; config=load_baseline_config(cfg,repository_root=root)
  if synthetic:
   from cmi_flu.strategy_e04_synthetic import run_synthetic; result,_=run_synthetic(config)
  else:
   from cmi_flu.runner import load_inputs; from cmi_flu.strategy_e04 import run_e04; inputs=load_inputs(config); result=run_e04(config,inputs)
  write_outputs(result,output_dir,synthetic=synthetic); print(f"CMI_FLU_E04_COMPLETE status={{result.get('status')}} synthetic={{synthetic}} submission=false"); return 0
 except Exception as error:
  code=hashlib.sha256(f"{{type(error).__name__}}:{{error}}".encode("utf-8",errors="replace")).hexdigest()[:20]; print(f"CMI_FLU_E04_FAILED exception_type={{type(error).__name__}} error_code={{code}}"); traceback.print_exc(); raise
 finally:
  if root is not None:shutil.rmtree(root,ignore_errors=True)
def main():
 a=parse_args()
 if a.self_test:return self_test()
 return execute(synthetic=a.synthetic,input_dir=a.input_dir,output_dir=a.output_dir.expanduser().resolve())
if __name__=="__main__":raise SystemExit(main())
'''


def main() -> int:
    a = args(); root = a.repository_root.expanduser().resolve(); validate_request(root); tree = fetch_science_tree(); runtime = build_runtime(tree); compile(runtime, "generated_e04_runtime.py", "exec"); a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(runtime, encoding="utf-8")
    print(f"CMI_FLU_E04_PREPARE_PASS science_commit={SCIENCE_COMMIT} runtime_sha256={hashlib.sha256(runtime.encode()).hexdigest()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
