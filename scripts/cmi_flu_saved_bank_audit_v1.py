"""Read two pinned successful private outputs and emit one no-fit aggregate audit."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
from pathlib import Path
import shutil
import sys
import tempfile

REQUEST_ID = '20260912-cmi-flu-saved-bank-audit-001'
SCIENCE_COMMIT = 'c822a0ca0bfb8406e4ea293a2d9706855bf6e181'
SCIENCE_BLOB = 'aace13086964535ffc6c3ef5a0dcfe976b54289b'
PAYLOAD = 'payloads/cmi-flu-saved-bank-audit-001/strategy_v3_saved_bank_audit.py'
OUTPUT = 'saved_bank_audit.json'
REQUEST = 'requests/cmi-flu-saved-bank-audit-001.json'
SOURCES = (
    ('v02', 'renta0426/cmi-flu-v3-v02-validation-bank-20260912-001', 1),
    ('v05', 'renta0426/cmi-flu-v3-v05-task13-scale-20260912-002', 1),
)


def load_science(root: Path):
    raw = (root / PAYLOAD).read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    if blob != SCIENCE_BLOB:
        raise ValueError('exact_science_blob_mismatch')
    spec = importlib.util.spec_from_file_location('cmi_saved_bank_audit_science', root / PAYLOAD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_request(root: Path, science) -> dict:
    request=json.loads((root / REQUEST).read_text())
    exact={
        'schema_version':1, 'request_id':REQUEST_ID,
        'operation':'read_current_outputs_and_reconcile',
        'execution_policy':'kaggle_native_capacity_v2',
        'science_commit':SCIENCE_COMMIT, 'science_blob':SCIENCE_BLOB,
        'kaggle_write_count':0, 'kaggle_compute_launch_count':0,
        'model_fit_count':0, 'competition_submission_count':0,
        'automatic_compute_retries':0, 'automatic_readout_retries':0,
        'output_filename':OUTPUT,
    }
    if any(request.get(k)!=v or type(request.get(k)) is not type(v) for k,v in exact.items()):
        raise ValueError('request_identity_or_operation_changed')
    expected=[{'key':key,'kernel':kernel,'expected_current_version':version,
               'files':{name:{'bytes':size,'sha256':sha} for name,(size,sha) in science.ARTIFACTS[key].items()}}
              for key,kernel,version in SOURCES]
    if request.get('sources')!=expected:
        raise ValueError('request_source_identity_changed')
    return request


def sanitize(result: dict) -> bytes:
    expected = {'schema_version','experiment','model_fit_count','kaggle_write_count',
                'kaggle_compute_launch_count','competition_submission_count',
                'individual_identifiers_or_prediction_vectors_emitted','paired_teacher',
                'source_unique_subjects','challenge_unique_subjects','reference_oof',
                'reference_challenge','S3_support','S2_actual_rank_contract',
                'S4_actual_rank_contract','repeat_coverage','new_candidates_created',
                'historical_results_rewritten','input_artifact_sha256','readout_provenance'}
    if set(result) != expected:
        raise ValueError('aggregate_schema_changed')
    for name in ('model_fit_count','kaggle_write_count','kaggle_compute_launch_count',
                 'competition_submission_count','new_candidates_created'):
        if type(result[name]) is not int or result[name] != 0:
            raise ValueError('no_fit_no_write_boundary_changed')
    if result['individual_identifiers_or_prediction_vectors_emitted'] is not False or result['historical_results_rewritten'] is not False:
        raise ValueError('aggregate_privacy_boundary_changed')
    banned={'participant_id','subject','subject_group','prediction','predictions','target','targets','row_index','split'}
    def walk(v):
        if isinstance(v,dict):
            if set(v) & banned: raise ValueError('row_level_key_rejected')
            for x in v.values():walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    walk(result)
    raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if len(raw)>32768:raise ValueError('aggregate_output_budget')
    return raw


def run(root: Path, output_dir: Path, *, reader=None) -> dict:
    if output_dir.exists(): raise ValueError('output_directory_must_be_fresh')
    science=load_science(root)
    validate_request(root,science)
    if reader is None:
        from kaggle_current_output_read import read_current_output
        reader=read_current_output
        bin_dir=Path(sys.executable).parent
        os.environ['PATH']=str(bin_dir)+os.pathsep+os.environ.get('PATH','')
        cli=shutil.which('kaggle')
        if cli is None or Path(cli).parent != bin_dir: raise ValueError('locked_cli_missing')
    with tempfile.TemporaryDirectory(prefix='cmi-saved-bank-audit-') as tmp:
        temp=Path(tmp)
        # The established reader proves current == expected before AND after
        # download. It captures CLI output, rejects extras and unlinks scratch.
        for key,kernel,version in SOURCES:
            allow={name:size for name,(size,_) in science.ARTIFACTS[key].items()}
            captured=io.StringIO()
            with contextlib.redirect_stdout(captured),contextlib.redirect_stderr(captured):
                reader(kernel=kernel,expected_version=version,allow=allow,output_dir=temp/key)
            science.verify_artifacts(temp/key,science.ARTIFACTS[key])
        result=science.audit_paths(temp/'v02',temp/'v05')
        result['readout_provenance']={'request_id':REQUEST_ID, 'science_commit':SCIENCE_COMMIT, 'science_blob':SCIENCE_BLOB,
            'sources':[{'kernel':kernel,'expected_current_version':version} for _,kernel,version in SOURCES],
            'read_only_output_operations':2, 'automatic_readout_retries':0}
        raw=sanitize(result)
    # Private files are already gone when the single safe result is published.
    output_dir.mkdir(parents=True)
    try:
        with (output_dir/OUTPUT).open('xb') as stream:stream.write(raw)
        if set(p.name for p in output_dir.iterdir())!={OUTPUT}:raise ValueError('output_allowlist_changed')
    except BaseException:
        shutil.rmtree(output_dir,ignore_errors=True)
        raise
    encoded=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)
    print(f'CMI_SAVED_BANK_AUDIT_SAFE_JSON name={OUTPUT} bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()} json={encoded}')
    print('CMI_SAVED_BANK_AUDIT_PASS reads=2 fit=0 notebook_write=0 compute=0 submit=0 private_cleanup=true')
    return result


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args.repository_root.resolve(),args.output_dir.resolve())
        return 0
    except Exception as exc:
        digest=hashlib.sha256(f'{type(exc).__name__}:{exc}'.encode()).hexdigest()[:20]
        print(f'CMI_SAVED_BANK_AUDIT_FAIL type={type(exc).__name__} code={digest} retries=0',file=sys.stderr)
        return 1

if __name__=='__main__':raise SystemExit(main())
