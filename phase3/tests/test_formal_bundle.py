import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from member3_common import canonical_sha256, sha256_file
from prepare_formal_bundle import build, preparation_payloads, materialize
from prepare_member3_bundle import verify_bundle


class FormalBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.evidence=self.root/'review.md'; self.evidence.write_text('fixture review')
        config=dict(resource_limits=None)
        config['config_sha256']=canonical_sha256(config)
        self.config=self.root/'config.json'; self.config.write_text(json.dumps(config))
        self.decision=self.root/'decision.json'
        self.record=dict(decision='pending',human_confirmed=False,
            formal_config_sha256=config['config_sha256'],
            reviewed_evidence=[dict(path=str(self.evidence),sha256=sha256_file(self.evidence))])
        self.decision.write_text(json.dumps(self.record))
        self.contract=patch('prepare_formal_bundle.formal_telemetry_contract',return_value={})
        self.contract.start()

    def tearDown(self): self.contract.stop(); self.temp.cleanup()

    def deploy(self):
        for name,data in preparation_payloads(self.config,self.decision).items():
            target=self.root/'deployed'/name
            target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
        return self.root/'deployed'

    def test_relocation_preserves_pending_and_evidence(self):
        root=self.deploy(); output=self.root/'resolved.json'
        result=materialize(root,output)
        self.assertEqual(result['decision'],'pending'); self.assertIs(result['human_confirmed'],False)
        self.assertTrue(Path(result['reviewed_evidence'][0]['path']).is_relative_to(root.resolve()))
        self.assertEqual(json.loads(self.decision.read_text()),self.record)
        with self.assertRaisesRegex(ValueError,'overwrite'): materialize(root,output)

    def test_changed_source_rejected(self):
        self.evidence.write_text('changed')
        with self.assertRaisesRegex(ValueError,'changed'): preparation_payloads(self.config,self.decision)

    def test_approved_record_rejected(self):
        self.record['decision']='approved'; self.decision.write_text(json.dumps(self.record))
        with self.assertRaisesRegex(ValueError,'pending'): preparation_payloads(self.config,self.decision)

    def test_hash_binding_rejected(self):
        self.record['formal_config_sha256']='0'*64; self.decision.write_text(json.dumps(self.record))
        with self.assertRaisesRegex(ValueError,'binding'): preparation_payloads(self.config,self.decision)

    def test_path_escape_rejected(self):
        root=self.deploy(); p=root/'formal-preparation/delegated-decision-PORTABLE-PENDING.json'
        record=json.loads(p.read_text()); record['reviewed_evidence'][0]['path']='../review.md'
        p.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'unsafe'): materialize(root,self.root/'out.json')

    def test_changed_deployed_evidence_rejected(self):
        root=self.deploy(); (root/'formal-preparation/reviewed-evidence/01-review.md').write_text('changed')
        with self.assertRaisesRegex(ValueError,'changed'): materialize(root,self.root/'out.json')

    def test_deterministic_verified_archive(self):
        with patch('prepare_formal_bundle.bundle_files',return_value=[]):
            first=self.root/'first.tar.gz'; second=self.root/'second.tar.gz'
            result=build(self.config,self.decision,first); build(self.config,self.decision,second)
        self.assertEqual(sha256_file(first),sha256_file(second))
        self.assertEqual(result['result'],'passed'); self.assertFalse(result['executable'])
        self.assertEqual(verify_bundle(first)['kind'],'member3_formal_preparation_bundle')
        with self.assertRaisesRegex(ValueError,'overwrite'): build(self.config,self.decision,first)


if __name__=='__main__': unittest.main()
