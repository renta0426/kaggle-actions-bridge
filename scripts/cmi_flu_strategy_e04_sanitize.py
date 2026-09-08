#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
REQUEST_ID='20260908-cmi-flu-strategy-e04-task13-rescue-001'; SCIENCE_COMMIT='6e4d786cddc7b2d02dc4671d1c005db551d19ad6'; E04_BLOB='73c112a8bb3b1ee7a6dd5bfbd5645a945bea6ebb'; CONTRACT_BLOB='3982541febfb4641fbf895438ae1a465bdbb3d5e'; BANNED=('"participant_id"','"subject_group"','"row_index"','"oof_predictions"','"challenge_predictions"','"predictions"')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def finite_or_none(v):
 if v is None:return True
 try:return math.isfinite(float(v))
 except:return False
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=Path,required=True); a=p.parse_args(); root=a.input_dir.resolve(); files=sorted(x.name for x in root.iterdir() if x.is_file())
 if files!=['bridge-result.json','metrics.json','summary.md']:raise SystemExit(f'E04 safe-output file set mismatch:{files}')
 texts={n:(root/n).read_text(encoding='utf-8') for n in files}
 for n,t in texts.items():
  if any(x in t for x in BANNED):raise SystemExit(f'E04 privacy contract failed:{n}')
 bridge=json.loads(texts['bridge-result.json']); m=json.loads(texts['metrics.json'])
 expected={'request_id':REQUEST_ID,'science_commit':SCIENCE_COMMIT,'competition_submission_attempted':False,'leaderboard_used_for_selection':False,'contains_participant_identifiers':False,'contains_row_level_predictions':False,'synthetic':False}
 for k,v in expected.items():
  if bridge.get(k)!=v:raise SystemExit(f'E04 bridge contract mismatch:{k}:{bridge.get(k)!r}')
 if bridge.get('metrics_sha256')!=sha(root/'metrics.json') or bridge.get('summary_sha256')!=sha(root/'summary.md'):raise SystemExit('E04 output hash mismatch')
 blobs=bridge.get('pinned_git_blobs') or {}
 if blobs.get('src/cmi_flu/strategy_e04.py')!=E04_BLOB or blobs.get('src/cmi_flu/strategy_e04_contract.py')!=CONTRACT_BLOB:raise SystemExit('E04 exact science blob mismatch')
 if m.get('schema_version')!=1 or m.get('experiment')!='strategy_v2_e04_task13_anchor_preserving_rescue' or m.get('task')!='Task1.3':raise SystemExit('E04 experiment identity mismatch')
 if m.get('status')!='complete' or (m.get('frozen_control_reproduction') or {}).get('all_pass') is not True:raise SystemExit(f"E04 frozen reproduction/status failed:{m.get('status')}")
 if m.get('competition_submission_attempted') is not False or m.get('leaderboard_used_for_selection') is not False or m.get('automatic_compute_retries')!=0:raise SystemExit('E04 selection/submission contract mismatch')
 frozen=m.get('frozen_conditions') or {}; expected_f={'strict_b21':'pls_1','anchor':'flow_rank__Antibody-secreting_cells_(ASC)','historical_rank':'enet_a0.1_l0.5','residual_alpha':10.0,'residual_shrinkage':0.25,'residual_score_cap':0.05,'proxy_weights':[0.25,1.0]}
 if frozen!=expected_f:raise SystemExit('E04 frozen condition mismatch')
 audit=m.get('measurement_audit') or {}
 if not audit:raise SystemExit('E04 measurement audit missing')
 for group in ('strict_cv','historical_loso','strict_cv_plus_proxy'):
  payload=m.get(group) or {}
  if set(payload)!={'proxy_weight_0.25','proxy_weight_1'}:raise SystemExit(f'E04 condition set mismatch:{group}:{sorted(payload)}')
  for name,item in payload.items():
   metric=(item.get('metrics') or {}).get('study_equal_spearman'); delta=(item.get('paired_vs_anchor') or {}).get('paired_study_mean_delta')
   if not finite_or_none(metric) or not finite_or_none(delta):raise SystemExit(f'E04 nonfinite aggregate:{group}:{name}')
   print(f'{group}/{name} study_equal={metric} delta_vs_anchor={delta} review={(item.get("paired_vs_anchor") or {}).get("large_studies_requiring_review")}')
 challenge=m.get('challenge') or {}
 for name in ('proxy_weight_0.25','proxy_weight_1'):
  item=challenge.get(name) or {}; decision=item.get('decision')
  if decision not in {'candidate_requires_review','no_promotion','data_limited'}:raise SystemExit(f'E04 challenge decision invalid:{name}:{decision}')
  print(f'challenge/{name} decision={decision} bridge_sources={item.get("strict_bridge_source_subjects")} local_candidate={item.get("local_candidate_heuristic")}')
 ident=m.get('proxy_weight_identifiability') or {}
 if ident.get('weight_selected') is not None or ident.get('one_source_study_cannot_identify_relative_proxy_weight') is not True:raise SystemExit('E04 proxy-weight identifiability contract mismatch')
 print(f'CMI_FLU_E04_RESULT PASS aggregate_only=true submission=false science_commit={SCIENCE_COMMIT} e04_blob={E04_BLOB}')
 return 0
if __name__=='__main__':raise SystemExit(main())