#!/usr/bin/env python3
"""Validate and print the aggregate-only E08 result needed by the science repo."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

REQUEST_ID="20260909-cmi-flu-strategy-e08-task13-applicability-001"
SCIENCE_COMMIT="1343cb8a0aefc41a4da0ec98d8c606d61c13088d"
E08_BLOB="c4afc93e4eaf5eb2784720ba8bb9215de9f43374"
E08_V2_BLOB="e5542251f94f011f0e99b83d1c1b704efa1ea213"
CONTRACT_BLOB="3982541febfb4641fbf895438ae1a465bdbb3d5e"
SYNTHETIC_BLOB="436b6a971622915cc5b335060f9a850feb6108fd"
TASK13_BLOB="5c6725dc757a5ba9dd21289b1c4f09997e1afdb8"
CONFIG_BLOB="170d3211e2795c0730e481056c7bb068accf97c9"
LEVELS=["exact_e04","material_plural_only","category_non_gate","marker_category_upper_bound"]
BANNED=('"participant_id"','"population_definition"','"subject_group"','"row_index"','"oof_predictions"','"challenge_predictions"','SYN_ONLY_')

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()

def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',type=Path,required=True);a=p.parse_args();root=a.input_dir.resolve()
 files=sorted(x.name for x in root.iterdir() if x.is_file())
 if files!=['bridge-result.json','metrics.json','summary.md']:raise SystemExit(f'E08 safe-output file set mismatch:{files}')
 texts={name:(root/name).read_text(encoding='utf-8') for name in files}
 for name,text in texts.items():
  if any(token in text for token in BANNED):raise SystemExit(f'E08 privacy contract failed:{name}')
 bridge=json.loads(texts['bridge-result.json']);result=json.loads(texts['metrics.json'])
 expected={
  'request_id':REQUEST_ID,'science_commit':SCIENCE_COMMIT,
  'strategy_e08_blob_sha':E08_BLOB,'strategy_e08_v2_blob_sha':E08_V2_BLOB,
  'strategy_e04_contract_blob_sha':CONTRACT_BLOB,'strategy_e04_synthetic_blob_sha':SYNTHETIC_BLOB,
  'task13_harmonization_blob_sha':TASK13_BLOB,'config_blob_sha':CONFIG_BLOB,
  'competition_submission_attempted':False,'leaderboard_used_for_selection':False,
  'contains_participant_identifiers':False,'contains_row_level_predictions':False,'outcomes_accessed':False,'synthetic':False,
 }
 for key,value in expected.items():
  if bridge.get(key)!=value:raise SystemExit(f'E08 bridge contract mismatch:{key}:{bridge.get(key)!r}')
 if bridge.get('metrics_sha256')!=sha(root/'metrics.json') or bridge.get('summary_sha256')!=sha(root/'summary.md'):raise SystemExit('E08 output hash mismatch')
 if result.get('schema_version')!=1 or result.get('experiment')!='strategy_v2_e08_task13_applicability_audit' or result.get('ontology_version')!='e08_task13_ontology_audit_v1' or result.get('task')!='Task1.3':raise SystemExit('E08 experiment identity mismatch')
 if result.get('status')!='complete' or result.get('outcomes_accessed') is not False or result.get('competition_submission_attempted') is not False or result.get('leaderboard_used_for_selection') is not False or result.get('automatic_compute_retries')!=0:raise SystemExit('E08 execution-boundary mismatch')
 if result.get('pairing_implementation')!='participant_unique_gate_v2' or result.get('minimum_bridge_subjects')!=8 or result.get('compatibility_levels')!=LEVELS:raise SystemExit('E08 frozen contract mismatch')
 counts=result.get('bridge_counts') or {}
 if list(counts)!=LEVELS:raise SystemExit('E08 bridge level set mismatch')
 for level in LEVELS:
  item=counts.get(level) or {}
  n=item.get('2024UGA_compatible_subjects')
  if not isinstance(n,int) or n<0:raise SystemExit(f'E08 bridge count invalid:{level}')
  if item.get('meets_minimum_bridge') is not (n>=8):raise SystemExit(f'E08 bridge threshold inconsistency:{level}')
 decision=result.get('decision') or {};exact=bool(counts['exact_e04']['meets_minimum_bridge']);alias=bool(counts['material_plural_only']['meets_minimum_bridge']);ready=alias and not exact
 if decision.get('e04b_measurement_bridge_ready') is not ready or decision.get('material_plural_alias_alone_recovers_bridge') is not ready:raise SystemExit('E08 decision not derived from frozen compatibility ladder')
 inv=result.get('file_inventory') or {}
 for key in ('competition_input_file_count','competition_input_fcs_count','managed_external_manifest_fcs_mentions'):
  if not isinstance(inv.get(key),int) or inv.get(key)<0:raise SystemExit(f'E08 inventory invalid:{key}')
 if inv.get('managed_external_manifest_fcs_mentions')!=0:raise SystemExit('E08 managed external manifest frozen evidence changed')
 aggregate={
  'request_id':REQUEST_ID,'science_commit':SCIENCE_COMMIT,
  'science_blobs':{'e08':E08_BLOB,'e08_v2':E08_V2_BLOB,'contract':CONTRACT_BLOB,'synthetic':SYNTHETIC_BLOB,'task13':TASK13_BLOB,'config':CONFIG_BLOB},
  'status':result.get('status'),'ontology_version':result.get('ontology_version'),'pairing_implementation':result.get('pairing_implementation'),
  'minimum_bridge_subjects':result.get('minimum_bridge_subjects'),'bridge_counts':counts,
  'strict_pair_metadata':result.get('strict_pair_metadata'),'sdy272_proxy_intrinsic':result.get('sdy272_proxy_intrinsic'),
  'historical_candidate_screen':result.get('historical_candidate_screen'),'third_strict_like_candidates':result.get('third_strict_like_candidates'),
  'file_inventory':inv,'decision':decision,'interpretation_limits':result.get('interpretation_limits'),
  'outcomes_accessed':False,'competition_submission_attempted':False,'leaderboard_used_for_selection':False,
 }
 print('CMI_FLU_E08_AGGREGATE_SUMMARY '+json.dumps(aggregate,sort_keys=True,separators=(',',':'),ensure_ascii=False))
 print(f'CMI_FLU_E08_RESULT PASS aggregate_only=true submission=false outcomes=false science_commit={SCIENCE_COMMIT} e08_blob={E08_BLOB}')
 return 0
if __name__=='__main__':raise SystemExit(main())
