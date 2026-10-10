import io
import json
import sys
import tarfile
import tempfile
import subprocess
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from member3_common import ROOT,read_json,verify_patched_checkout
from prepare_member3_bundle import verify_bundle

class CloudBundleTests(unittest.TestCase):
    def test_unsafe_archive_rejected_before_extract(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.tar.gz'
            with tarfile.open(p,'w:gz') as archive:
                item=tarfile.TarInfo('../escape');item.size=1;archive.addfile(item,io.BytesIO(b'x'))
            with self.assertRaisesRegex(ValueError,'unsafe'):verify_bundle(p)
    def test_reconstruct_patch_and_reject_extra_edits(self):
        source=Path('/Users/REDACTED/Desktop/SOFT3888/Member2_Acceptance/2026-10-04/sources/finetune.original.py')
        if not source.exists():self.skipTest('local reviewed original source unavailable')
        with tempfile.TemporaryDirectory() as td:
            repo=Path(td);(repo/'finetune.py').write_bytes(source.read_bytes())
            subprocess.run(['git','init','-q',td],check=True)
            subprocess.run(['git','-C',td,'add','finetune.py'],check=True)
            subprocess.run(['git','-C',td,'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture'],check=True)
            lock=read_json(ROOT/'phase3/configs/source-lock.json')
            lock['upstream']['commit']=subprocess.check_output(['git','-C',td,'rev-parse','HEAD'],text=True).strip()
            subprocess.run(['git','-C',td,'apply',str(ROOT/lock['upstream']['patch'])],check=True)
            self.assertEqual(len(verify_patched_checkout(repo,lock)),64)
            with (repo/'finetune.py').open('a') as f:f.write('\n# unexpected modification\n')
            with self.assertRaisesRegex(ValueError,'differs'):verify_patched_checkout(repo,lock)
if __name__=='__main__':unittest.main()
