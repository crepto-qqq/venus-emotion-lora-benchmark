from __future__ import annotations
import copy, json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'phase3/tools'))
from member3_common import canonical_sha256, merge_patch, read_json, verify_release_hashes, validate_resource_limits, build_finetune_command
from validate_lora_config import static_checks, runtime_checks, pilot_checks, formal_checks
from validate_training_acceptance import validate as validate_acceptance

class Member3ToolsTests(unittest.TestCase):
    def setUp(self):
        self.config=read_json(ROOT/'phase3/configs/lora-pilot-v1.json')
        self.interface=read_json(ROOT/'phase3/configs/qwen-training-interface-v1.json')

    def test_release_hashes_are_exact(self): self.assertEqual(verify_release_hashes(),[])
    def test_rank2_fixture_is_rejected_as_training_recommendation(self):
        c=copy.deepcopy(self.config); c['lora']['rank']=2; self.assertTrue(any('rank-2' in x for x in static_checks(c)))
    def test_qlora_is_forbidden(self):
        c=copy.deepcopy(self.config); c['lora']['mode']='qlora'; self.assertTrue(any('QLoRA' in x for x in static_checks(c)))
    def test_more_than_three_epochs_rejected(self):
        c=copy.deepcopy(self.config); c['training']['epochs']=4; self.assertTrue(any('at most 3' in x for x in static_checks(c)))
    def test_resource_limits_are_mandatory(self): self.assertTrue(validate_resource_limits(None))
    def test_hidden_parser_inventory_blocks_pilot(self): self.assertTrue(any('parser inventory' in x for x in pilot_checks(self.config,self.interface)))
    def test_visual_target_inventory_is_rejected(self):
        c=copy.deepcopy(self.config); c['inventory_state']='inventory_verified'; c['lora'].update({'target_modules':['c_attn'],'target_source':'runtime_inventory','inventory_sha256':'x','matched_module_count':1})
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'inv.json'; p.write_text(json.dumps({'result':'passed','report_sha256':'x','matched_module_count':1,'visual_modules_targeted':True}))
            self.assertTrue(any('visual' in x for x in runtime_checks(c,p)))
    def test_absent_review_blocks_formal(self): self.assertTrue(any('pilot-review' in x for x in formal_checks(self.config,self.interface,Path('/no/such/review.json'))))
    def test_merge_patch_semantics(self): self.assertEqual(merge_patch({'a':1,'b':{'x':1,'y':2}},{'b':{'x':3,'y':None}}),{'a':1,'b':{'x':3}})
    def test_command_contains_explicit_behavior_flags(self):
        c=copy.deepcopy(self.config); c['lora']['target_modules']=['c_attn']; cmd=build_finetune_command(c,self.interface); text=' '.join(cmd)
        for flag in ['--use_lora','--q_lora','--fix_vit','--lora_r','--lora_target_modules','--bf16','--gradient_accumulation_steps','--model_max_length']:
            self.assertIn(flag,text)
    def test_invalid_post_training_acceptance_is_rejected(self):
        bad={"schema_version":1,"kind":"training_acceptance","reviewer_role":"Member 3B","decision":"accepted","formal_config_sha256":"x","adapter_sha256":"y","reload_passed":False,"clean_process_inference_passed":False,"selection_basis":"Eval80","selected_checkpoint":{}}
        self.assertGreaterEqual(len(validate_acceptance(bad)),5)

    def test_canonical_hash_ignores_formatting(self): self.assertEqual(canonical_sha256({'b':2,'a':1}),canonical_sha256({'a':1,'b':2}))

if __name__=='__main__': unittest.main()
