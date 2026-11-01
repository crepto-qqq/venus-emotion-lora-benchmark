"""Offline fixture checks. Fake adapter bytes are never described as GPU evidence."""
import copy,json,sys,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from member3_common import ROOT,read_json,write_json,canonical_sha256,sha256_file
from plan_formal_validation import plan,collect
from generate_lora_validation import adapter_identity
from validate_lora_config import delegated_review_checks,formal_checks

class FormalWorkflowTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.p=Path(self.temp.name)
  self.c=read_json(ROOT/'phase3/configs/lora-pilot-v1.json')
  self.c['kind']='lora_formal_config';self.c['training'].update(epochs=3,max_steps=180,save_steps=60,save_total_limit=3)
  self.c['planning_metadata']={'checkpoint_selection':dict(primary='lowest_validation_token_weighted_loss',tie_breakers=['highest_classification_exact_match','highest_guidance_format_valid_rate','earliest_checkpoint_step'])}
 def tearDown(self):self.temp.cleanup()
 def record(self,path,record,key='report_sha256'):
  record[key]=canonical_sha256(record,omit_key=key);write_json(path,record)
 def fixture(self):
  checkpoints=[];self.train=self.p/'training';self.val=self.p/'validation'
  gold=read_json(ROOT/self.c['data']['validation_repo_path'])
  for step,loss in [(60,1.2),(120,.8),(180,.9)]:
   a=self.train/f'checkpoint-{step}';a.mkdir(parents=True)
   (a/'adapter_model.bin').write_bytes(f'FAKE offline fixture {step}'.encode());write_json(a/'adapter_config.json',{'peft_type':'LORA','r':8});write_json(a/'trainer_state.json',{'global_step':step})
   ids=adapter_identity(a);checkpoints.append(dict(step=step,weights_sha256=ids['adapter_model.bin'],config_sha256=ids['adapter_config.json'],trainer_state_sha256=sha256_file(a/'trainer_state.json')))
   d=self.val/f'checkpoint-{step}';d.mkdir(parents=True)
   self.record(d/'run.json',dict(execution_state='completed',config_sha256=canonical_sha256(self.c,omit_key='config_sha256'),validation_sha256=sha256_file(ROOT/self.c['data']['validation_repo_path']),adapter_files=ids,adapter_bundle_sha256=canonical_sha256(ids),generation={'do_sample':False},max_new_tokens=512,sample_count=96,prediction_count=96))
   (d/'predictions.jsonl').write_text(''.join(json.dumps(dict(id=r['id'],prediction=r['conversations'][1]['value'],loss=loss,supervised_tokens=10))+'\n' for r in gold))
  self.attempt=self.p/'attempt.json'
  t=self.train/'telemetry';t.mkdir()
  self.record(t/'summary.json',dict(state='source_completed',identity={'config_sha256':'a'*64},dataset_rows_checked=480,microbatches_checked=1440,samples_observed=1440,optimizer_steps_observed=180))
  for name in ['events.jsonl','trainable-parameters.json','parameter-updates.json']:(t/name).write_text('{}\n')
  verified=dict(state='source_completed',summary_sha256=sha256_file(t/'summary.json'),events_sha256=sha256_file(t/'events.jsonl'),trainable_parameters_sha256=sha256_file(t/'trainable-parameters.json'),formal={'checkpoints':checkpoints,'parameter_updates_sha256':sha256_file(t/'parameter-updates.json')})
  self.record(self.attempt,dict(mode='formal',execution_state='completed',returncode=0,config_sha256=canonical_sha256(self.c,omit_key='config_sha256'),resolved_config_sha256='a'*64,telemetry_verification=verified),'attempt_record_sha256')
 def result(self):return collect(self.c,self.train,self.val,self.attempt)
 def test_plan_is_sequential_and_does_not_create_jobs(self):
  result=plan(self.c,self.p/'config.json',self.p/'training',self.p/'validation',Path('interface'),Path('inventory'),Path('review'))
  self.assertEqual([x['step'] for x in result['jobs']],[60,120,180]);self.assertFalse((self.p/'validation').exists())
 def test_selects_middle_checkpoint_and_does_not_claim_quality(self):
  self.fixture();r=self.result();self.assertEqual(r['provisional_selected_step'],120)
  self.assertEqual(r['candidates'][0]['classification_correct'],80);self.assertFalse(r['final_acceptance']);self.assertFalse(r['formal_Eval80_completed'])
 def test_missing_predictions_rejected(self):
  self.fixture();p=self.val/'checkpoint-120/predictions.jsonl';p.write_text('\n'.join(p.read_text().splitlines()[:-1]))
  with self.assertRaises(ValueError):self.result()
 def test_duplicate_id_rejected(self):
  self.fixture();p=self.val/'checkpoint-120/predictions.jsonl';rows=p.read_text().splitlines();rows[-1]=rows[0];p.write_text('\n'.join(rows))
  with self.assertRaises(ValueError):self.result()
 def test_missing_token_loss_rejected(self):
  self.fixture();p=self.val/'checkpoint-60/predictions.jsonl';rows=p.read_text().splitlines();r=json.loads(rows[0]);r.pop('supervised_tokens');rows[0]=json.dumps(r);p.write_text('\n'.join(rows))
  with self.assertRaises(ValueError):self.result()
 def test_replaced_adapter_rejected(self):
  self.fixture();(self.train/'checkpoint-60/adapter_model.bin').write_bytes(b'changed')
  with self.assertRaisesRegex(ValueError,'checkpoint differs'):self.result()
 def test_replaced_training_telemetry_rejected(self):
  self.fixture();(self.train/'telemetry/events.jsonl').write_text('changed')
  with self.assertRaisesRegex(ValueError,'telemetry file'):self.result()
 def test_unfinished_attempt_rejected(self):
  self.fixture();r=read_json(self.attempt);r['execution_state']='failed';self.record(self.attempt,r,'attempt_record_sha256')
  with self.assertRaises(ValueError):self.result()
 def test_changed_generation_rejected(self):
  self.fixture();p=self.val/'checkpoint-60/run.json';r=read_json(p);r['generation']['do_sample']=True;self.record(p,r)
  with self.assertRaises(ValueError):self.result()
 def decision(self):
  p=self.p/'evidence.txt';p.write_text('offline test fixture only')
  return dict(schema_version=1,kind='delegated_formal_decision',reviewer_role='delegated_user',decision_maker='Zhiyuan Zhang',human_confirmed=True,independent_member3b_review=False,delegation={'source':'user_reported_Jackie_delegation','statement':'fixture only'},formal_resource_limits=self.c['resource_limits'],release_hashes=self.c['data']['release_hashes'],checkpoint_selection=self.c['planning_metadata']['checkpoint_selection'],known_limitations=['no quality certification'],reviewed_evidence=[{'path':str(p),'sha256':sha256_file(p)}])
 def test_delegated_record_checks_bind_real_files(self):
  r=self.decision();self.assertEqual(delegated_review_checks(self.c,r),[])
  (self.p/'evidence.txt').write_text('changed');self.assertTrue(delegated_review_checks(self.c,r))
 def test_pending_or_impersonated_decision_rejected(self):
  r=self.decision()
  for key,value in [('human_confirmed',False),('reviewer_role','Member 3B'),('independent_member3b_review',True),('release_hashes',{}),('checkpoint_selection',{})]:
   broken=copy.deepcopy(r);broken[key]=value;self.assertTrue(delegated_review_checks(self.c,broken))
 def test_formal_gate_rejects_pending_even_when_bound(self):
  r=self.decision();r.update(decision='pending',formal_config_sha256=canonical_sha256(self.c,omit_key='config_sha256'))
  p=self.p/'review.json';write_json(p,r)
  with patch('validate_lora_config.pilot_checks',return_value=[]):
   self.assertIn('pilot review decision is not approved',formal_checks(self.c,{},p))
 def test_explicit_approved_fixture_does_not_impersonate_3b(self):
  r=self.decision();r.update(decision='approved',formal_config_sha256=canonical_sha256(self.c,omit_key='config_sha256'))
  p=self.p/'review.json';write_json(p,r)
  with patch('validate_lora_config.pilot_checks',return_value=[]):self.assertEqual(formal_checks(self.c,{},p),[])
 def test_formal_generator_uses_formal_validator_before_model_loading(self):
  import generate_lora_validation as generator
  p=self.p/'config.json';write_json(p,self.c)
  args=['generate_lora_validation.py','--config',str(p),'--adapter',str(self.p/'adapter'),'--output-dir',str(self.p/'output'),'--inventory',str(self.p/'inventory'),'--pilot-review',str(self.p/'review'),'--execute']
  with patch.object(sys,'argv',args),patch.object(generator.subprocess,'run',side_effect=RuntimeError('fixture stop before GPU import')) as validator:
   with self.assertRaisesRegex(RuntimeError,'fixture stop'):generator.main()
   command=validator.call_args.args[0];self.assertEqual(command[command.index('--mode')+1],'formal');self.assertIn('--pilot-review',command)

if __name__=='__main__':unittest.main()
