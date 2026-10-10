import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from generate_lora_validation import validation_rows, user_query, adapter_identity
from member3_common import ROOT

class ValidationGenerationTests(unittest.TestCase):
    def test_complete_frozen_validation_and_user_only_query(self):
        rows=validation_rows(ROOT/'phase3/releases/bridge-train-v1/bridge-train-v1.validation.json')
        self.assertEqual(len(rows),96)
        row={'conversations':[{'from':'user','value':'image and question'},{'from':'assistant','value':'SECRET GOLD ANSWER'}]}
        self.assertEqual(user_query(row),'image and question')
        self.assertNotIn('SECRET GOLD ANSWER',user_query(row))
    def test_eval_or_subset_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'validation.json';p.write_text('[]')
            with self.assertRaises(ValueError):validation_rows(p)
    def test_missing_or_ambiguous_adapter_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            with self.assertRaises(ValueError):adapter_identity(p)
            (p/'adapter_config.json').write_text('{}');(p/'adapter_model.bin').write_bytes(b'test')
            self.assertEqual(len(adapter_identity(p)),2)
            (p/'adapter_model.safetensors').write_bytes(b'test')
            with self.assertRaises(ValueError):adapter_identity(p)
if __name__=='__main__':unittest.main()
