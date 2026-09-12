"""Exercise the real CSV->audit->sanitizer->cleanup path with synthetic inputs."""
from __future__ import annotations
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_strategy_v3_saved_bank_audit import fixture

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('saved_bank_driver',ROOT/'scripts/cmi_flu_saved_bank_audit_v1.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)


def synthetic_files():
    old,oc,new,ch,s=fixture()
    encode=lambda f:f.to_csv(index=False,float_format='%.17g',lineterminator='\n').encode()
    return {
        'v02':{'v3_v02_oof_bank.csv':encode(old),'v3_v02_challenge_bank.csv':encode(oc),'v3_v02_summary.json':b'{}\n','v3_v02_bank_manifest.json':b'{}\n'},
        'v05':{'v3_v05_task13_oof_bank.csv':encode(new),'v3_v05_task13_challenge_bank.csv':encode(ch),'v3_v05_task13_summary.json':(json.dumps(s)+'\n').encode(),'v3_v05_task13_bank_manifest.json':b'{}\n'},
    }


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'repo';self.root.mkdir()
        payload=self.root/D.PAYLOAD;payload.parent.mkdir(parents=True);shutil.copyfile(ROOT/D.PAYLOAD,payload)
        self.science=D.load_science(self.root)
        self.contents=synthetic_files()
        self.science.ARTIFACTS={k:{n:(len(b),hashlib.sha256(b).hexdigest()) for n,b in files.items()} for k,files in self.contents.items()}
        self.calls=[];self.paths=[]
        self.req={
            'schema_version':1,'request_id':D.REQUEST_ID,'operation':'read_current_outputs_and_reconcile',
            'execution_policy':'kaggle_native_capacity_v2','science_commit':D.SCIENCE_COMMIT,'science_blob':D.SCIENCE_BLOB,
            'kaggle_write_count':0,'kaggle_compute_launch_count':0,'model_fit_count':0,'competition_submission_count':0,
            'automatic_compute_retries':0,'automatic_readout_retries':0,'output_filename':D.OUTPUT,
            'sources':[{'key':key,'kernel':kernel,'expected_current_version':version,'files':{n:{'bytes':size,'sha256':sha} for n,(size,sha) in self.science.ARTIFACTS[key].items()}} for key,kernel,version in D.SOURCES],
        }
        p=self.root/D.REQUEST;p.parent.mkdir(parents=True);p.write_text(json.dumps(self.req))
        self.patch=patch.object(D,'load_science',return_value=self.science);self.patch.start();self.addCleanup(self.patch.stop)
        self.out=Path(self.tmp.name)/'safe'

    def reader(self,*,kernel,expected_version,allow,output_dir):
        self.calls.append((kernel,expected_version));self.paths.append(output_dir)
        key=next(key for key,k,v in D.SOURCES if k==kernel and v==expected_version)
        self.assertEqual(set(allow),set(self.contents[key]));output_dir.mkdir()
        for name,data in self.contents[key].items():(output_dir/name).write_bytes(data)
        print('SYN_PRIVATE_LOG_MUST_NOT_ESCAPE')

    def run_it(self,reader=None):
        log=io.StringIO();previous=sys.getprofile()
        def guard(frame,event,arg):
            if event=='call' and frame.f_code.co_name in {'fit','fit_transform','kernels_push','competition_submit'}:
                raise AssertionError('fit or write called')
        try:
            sys.setprofile(guard)
            with contextlib.redirect_stdout(log):result=D.run(self.root,self.out,reader=reader or self.reader)
        finally:sys.setprofile(previous)
        self.assertNotIn('SYN_PRIVATE',log.getvalue());self.assertNotIn('SYN_SRC',log.getvalue())
        return result,log.getvalue()

    def assert_clean(self):
        self.assertTrue(all(not p.parent.exists() for p in self.paths))

    def test_complete_real_parser_audit_sanitizer_cleanup(self):
        result,log=self.run_it()
        self.assertEqual(len(self.calls),2);self.assert_clean()
        self.assertEqual({p.name for p in self.out.iterdir()},{D.OUTPUT})
        raw=(self.out/D.OUTPUT).read_bytes();self.assertEqual(json.loads(raw),result)
        self.assertIn(hashlib.sha256(raw).hexdigest(),log)
        self.assertIn('CMI_SAVED_BANK_AUDIT_PASS',log)
        self.assertEqual(result['S3_support']['above_source_max'],40)
        self.assertEqual(result['readout_provenance']['read_only_output_operations'],2)

    def test_wrong_hash_fails_before_second_download(self):
        def bad(**kwargs):
            self.reader(**kwargs)
            p=kwargs['output_dir']/'v3_v02_summary.json';p.write_bytes(b'[]\n')
        with self.assertRaisesRegex(self.science.AuditError,'hash'):self.run_it(bad)
        self.assertEqual(len(self.calls),1);self.assert_clean();self.assertFalse(self.out.exists())

    def test_version_error_not_retried_and_cleans_first_download(self):
        def stale(**kwargs):
            if len(self.calls)==1:raise ValueError('current_version_not_expected')
            self.reader(**kwargs)
        with self.assertRaisesRegex(ValueError,'current_version_not_expected'):self.run_it(stale)
        self.assertEqual(len(self.calls),1);self.assert_clean();self.assertFalse(self.out.exists())

    def test_unexpected_output_rejected_and_cleaned(self):
        def extra(**kwargs):
            self.reader(**kwargs);(kwargs['output_dir']/'private-extra.txt').write_text('no')
        with self.assertRaisesRegex(self.science.AuditError,'allowlist'):self.run_it(extra)
        self.assert_clean();self.assertFalse(self.out.exists())

    def test_changed_request_rejected_before_read(self):
        self.req['sources'][0]['expected_current_version']=2
        (self.root/D.REQUEST).write_text(json.dumps(self.req))
        with self.assertRaisesRegex(ValueError,'source_identity'):self.run_it()
        self.assertEqual(self.calls,[])

    def test_bool_not_accepted_as_zero_operation_count(self):
        self.req['kaggle_write_count']=False
        (self.root/D.REQUEST).write_text(json.dumps(self.req))
        with self.assertRaisesRegex(ValueError,'operation_changed'):self.run_it()
        self.assertEqual(self.calls,[])

    def test_existing_destination_never_overwritten(self):
        self.out.mkdir();(self.out/'keep').write_bytes(b'unchanged')
        with self.assertRaisesRegex(ValueError,'must_be_fresh'):self.run_it()
        self.assertEqual((self.out/'keep').read_bytes(),b'unchanged');self.assertEqual(self.calls,[])

    def test_sanitizer_rejects_row_fields_and_nonfinite(self):
        result,_=self.run_it()
        bad=copy.deepcopy(result);bad['S3_support']['participant_id']='not allowed'
        with self.assertRaisesRegex(ValueError,'row_level_key'):D.sanitize(bad)
        bad=copy.deepcopy(result);bad['S3_support']['source_min']=float('nan')
        with self.assertRaises(ValueError):D.sanitize(bad)

if __name__=='__main__':unittest.main()
