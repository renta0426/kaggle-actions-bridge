#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
REQUEST_ID="20260907-cmi-flu-strategy-e05-hai-donor-strain-001"; SCIENCE_COMMIT="e02a601f38250480b526e548a48cf7f2526c00ca"; E05_BLOB="78cde9e3a6b6e3b332352218c4b4545f769aa6c4"; BANNED=('"participant_id"','"subject_group"','"row_index"','"oof_predictions"','"challenge_predictions"','"predictions"')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def finite_or_none(v):
 if v is None:return True
 try:return math.isfinite(float(v))
 except:return False
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=Path,required=True); a=p.parse_args(); root=a.input_dir.resolve(); files=sorted(x.name for x in root.iterdir() if x.is_file())
 if files!=['bridge-result.json','metrics.json','summary.md']:raise SystemExit(f'E05 safe-output file set mismatch:{files}')
 texts={n:(root/n).read_text() for n in files}
 for n,t in texts.items():
  if any(x in t for x in BANNED):raise SystemExit(f'E05 privacy contract failed:{n}')
 bridge=json.loads(texts['bridge-result.json']); metrics=json.loads(texts['metrics.json'])
 for k,v in {'request_id':REQUEST_ID,'science_commit':SCIENCE_COMMIT,'competition_submission_attempted':False,'leaderboard_used_for_selection':False,'contains_participant_identifiers':False,'contains_row_level_predictions':False}.items():
  if bridge.get(k)!=v:raise SystemExit(f'E05 bridge contract mismatch:{k}:{bridge.get(k)!r}')
 if bridge.get('metrics_sha256')!=sha(root/'metrics.json') or bridge.get('summary_sha256')!=sha(root/'summary.md'):raise SystemExit('E05 output hash mismatch')
 if metrics.get('experiment')!='strategy_v2_e05_hai_donor_strain' or set((metrics.get('tasks') or {}))!={'Task2.1','Task2.2'}:raise SystemExit('E05 experiment/task identity mismatch')
 fc=metrics.get('feature_contract') or {}; sc=metrics.get('split_contract') or {}; wc=metrics.get('weight_contract') or {}
 if fc.get('ridge_alpha')!=10.0 or fc.get('interaction_count')!=8 or fc.get('free_participant_embedding') is not False or fc.get('raw_strain_id_one_hot') is not False:raise SystemExit('E05 feature contract mismatch')
 if sc.get('fixed_subject_strain_simultaneous_holdouts')!=4 or sc.get('simultaneous_selection_uses_outcomes') is not False:raise SystemExit('E05 split contract mismatch')
 if wc.get('rows_are_not_counted_as_independent_donors') is not True:raise SystemExit('E05 weight contract mismatch')
 for task,payload in metrics['tasks'].items():
  route=payload.get('routing') or {}; delta=route.get('historical_proxy_study_mean_delta_vs_anchor')
  if not finite_or_none(delta):raise SystemExit(f'E05 nonfinite route delta:{task}')
  cond=payload.get('study_out_conditions') or {}
  if set(cond)!={'b21_reference','phase_a_fixed_sequence','ridge_main_effects','ridge_donor_by_strain_interactions'}:raise SystemExit(f'E05 condition set mismatch:{task}')
  print(f"{task} route={route.get('routing')} positive={route.get('positive_e05_signal')} delta_vs_anchor={delta} challenge_rank_changed={route.get('challenge_rank_changed_by_at_least_one_position')} interaction_rmse={route.get('interaction_panel_proxy_rmse')} main_rmse={route.get('main_effects_panel_proxy_rmse')}")
 print(f'CMI_FLU_E05_RESULT PASS aggregate_only=true submission=false science_commit={SCIENCE_COMMIT} e05_blob={E05_BLOB}')
 return 0
if __name__=='__main__':raise SystemExit(main())
