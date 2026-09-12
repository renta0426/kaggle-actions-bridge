"""Synthetic-only no-fit reconciliation regressions; portable for public bridge."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


def load_audit():
    try:
        from cmi_flu import strategy_v3_saved_bank_audit as module
        return module
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[1]
        path = root / 'src/cmi_flu/strategy_v3_saved_bank_audit.py'
        if not path.is_file():
            path = root / 'payloads/cmi-flu-saved-bank-audit-001/strategy_v3_saved_bank_audit.py'
        spec = importlib.util.spec_from_file_location('saved_audit_test_module', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


A = load_audit()


def fixture():
    n=23; ids=[f'SYN_SRC_{i:02d}' for i in range(n)]; subjects=[f'SYN_SUBJECT_{i:02d}' for i in range(n)]
    x=np.linspace(.001,.029,n); y=1.1*x + .0005
    old=pd.DataFrame({'task':'Task1.3','candidate':'b21_pls_1','prediction_space':'raw_target','target_unit':'official_absolute_flow_target','provenance':'synthetic','participant_id':ids,'subject_group':subjects,'study_group':'2024_UGA','target':y,'prediction':x*.9})
    parts=[]
    for r in range(3):
        parts.append(pd.DataFrame({'task':'Task1.3','row_index':range(n),'repeat':r,'split':[f'repeat={r}/fold={i%5}' for i in range(n)],A.TARGET:y,'raw_baseline':x,'S1':y.mean(),'S2':x*1.1,'S3':x+.001,'S4':x*1.2,'b21_compatible_pls1':x*(1+.01*r),'participant_id':ids,'subject_group':subjects,'study_group':'2024UGA'}))
    new=pd.concat(parts,ignore_index=True)
    cx=np.linspace(.04,.08,40); cid=[f'SYN_CH_{i:02d}' for i in range(40)]
    challenge=pd.DataFrame({'participant_id':cid,'subject_group':cid,'study_group':'2025LJI',A.BASELINE:cx,'raw_baseline':cx,'b21_compatible_pls1':cx*.9,'S1':y.mean(),'S2':cx*1.1,'S3':.016,'S4':cx*1.2})
    oc=pd.DataFrame({'task':'Task1.3','candidate':'b21_pls_1_fresh','prediction_space':'raw_target','target_unit':'official_absolute_flow_target','provenance':'synthetic','participant_id':cid,'prediction':cx*.9})
    s={'Task1.3':{'full_fit_parameters':{'S2':{'c':1.1},'S3':{'a':.001,'b':.015}}},'runtime':{'repair_study_alias_sha256':A.ALIAS_HASH}}
    return old,oc,new,challenge,s


class SavedBankAuditTests(unittest.TestCase):
    def test_full_success_aggregate_only_no_fit(self):
        forbidden={'fit','fit_transform','kernels_push','competition_submit','create_dataset_new'}
        previous=sys.getprofile()
        def guard(frame,event,arg):
            if event=='call' and frame.f_code.co_name in forbidden:
                raise AssertionError('forbidden call')
        try:
            sys.setprofile(guard)
            result=A.audit_frames(*fixture())
        finally: sys.setprofile(previous)
        self.assertEqual(result['model_fit_count'],0)
        self.assertEqual(result['S3_support']['above_source_max'],40)
        self.assertTrue(result['S3_support']['ECDF_collapsed'])
        self.assertTrue(result['S3_support']['slope_positive'])
        self.assertFalse(result['reference_oof']['comparison']['equal_at_1e_12'])
        self.assertTrue(result['reference_challenge']['exact_equal'])
        text=json.dumps(result,allow_nan=False)
        self.assertNotIn('SYN_SRC',text); self.assertNotIn('SYN_CH',text);self.assertNotIn('SYN_SUBJECT',text)
        self.assertTrue(result['S2_actual_rank_contract']['same_weak_order'])

    def test_row_permutation_preserves_result(self):
        args=fixture(); expected=A.audit_frames(*args)
        shuffled=tuple(f.sample(frac=1,random_state=17).reset_index(drop=True) for f in args[:4])+(args[4],)
        self.assertEqual(expected,A.audit_frames(*shuffled))

    def test_teacher_mismatch_rejected(self):
        a=list(fixture()); a[0].loc[0,'target']+=.01
        with self.assertRaisesRegex(A.AuditError,'cross_run_teacher'):A.audit_frames(*a)

    def test_incomplete_repeat_rejected(self):
        a=list(fixture());a[2]=a[2].iloc[:-1]
        with self.assertRaises(A.AuditError):A.audit_frames(*a)

    def test_duplicate_id_rejected(self):
        a=list(fixture());a[0].loc[1,'participant_id']=a[0].loc[0,'participant_id']
        with self.assertRaisesRegex(A.AuditError,'duplicate_identity'):A.audit_frames(*a)

    def test_subject_mapping_changed_rejected(self):
        a=list(fixture());a[0].loc[0,'subject_group']='SYN_DIFFERENT'
        with self.assertRaisesRegex(A.AuditError,'subject_mapping'):A.audit_frames(*a)

    def test_unknown_study_alias_rejected(self):
        a=list(fixture());a[0].loc[0,'study_group']='OTHER'
        with self.assertRaisesRegex(A.AuditError,'study_alias'):A.audit_frames(*a)

    def test_missing_subject_rejected(self):
        a=list(fixture());a[2].loc[0,'subject_group']=None
        with self.assertRaises(A.AuditError):A.audit_frames(*a)

    def test_S3_formula_mismatch_rejected(self):
        a=list(fixture());a[3].loc[0,'S3']+=.1
        with self.assertRaisesRegex(A.AuditError,'S3_saved_formula'):A.audit_frames(*a)

    def test_finite_and_schema_rejected(self):
        a=list(fixture());a[3].loc[0,'S4']=float('inf')
        with self.assertRaisesRegex(A.AuditError,'nonfinite'):A.audit_frames(*a)
        a=list(fixture());a[3]['private_debug']='secret'
        with self.assertRaisesRegex(A.AuditError,'schema'):A.audit_frames(*a)

    def test_alias_hash_is_frozen(self):
        a=list(fixture());a[4]['runtime']['repair_study_alias_sha256']='0'*64
        with self.assertRaisesRegex(A.AuditError,'alias_hash'):A.audit_frames(*a)

    def test_repeated_target_not_changed(self):
        a=list(fixture());a[2].loc[0,A.TARGET]+=.01
        with self.assertRaisesRegex(A.AuditError,'repeated_identity'):A.audit_frames(*a)

    def test_hash_allowlist_and_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'test.json';p.write_bytes(b'{}\n')
            contract={'test.json':(3,hashlib.sha256(p.read_bytes()).hexdigest())}
            A.verify_artifacts(root,contract)
            p.write_bytes(b'[]\n')
            with self.assertRaisesRegex(A.AuditError,'hash'):A.verify_artifacts(root,contract)
            p.unlink();p.symlink_to(root/'missing')
            with self.assertRaises(A.AuditError):A.verify_artifacts(root,contract)
            p.unlink();p.write_bytes(b'{}\n');(root/'extra').write_text('x')
            with self.assertRaisesRegex(A.AuditError,'allowlist'):A.verify_artifacts(root,contract)

    def test_no_fit_or_submit_calls_in_science_ast(self):
        tree=ast.parse(Path(A.__file__).read_text())
        forbidden={'fit','fit_transform','kernels_push','competition_submit','submit'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else getattr(node.func,'id','')
                self.assertNotIn(name,forbidden)

if __name__=='__main__':unittest.main()
