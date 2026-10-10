import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from member3_common import ROOT,read_json
from check_lora_model_inputs import select_samples,check_task,parameter_checks,prepare_model_batch

class SampleSelectionTests(unittest.TestCase):
    def test_formal_config_without_pilot_subset_uses_training_only(self):
        config=read_json(ROOT/'phase3/configs/lora-pilot-v1.json')
        config['kind']='lora_formal_config'
        config.pop('pilot_subset',None)
        samples=select_samples(config)
        train={row['id'] for row in read_json(ROOT/config['data']['train_repo_path'])}
        val={row['id'] for row in read_json(ROOT/config['data']['validation_repo_path'])}
        self.assertEqual(len(samples),2)
        self.assertTrue(all(row['id'] in train and row['id'] not in val for row in samples))
        self.assertEqual(samples,select_samples(config))
        self.assertTrue(samples[0]['id'].endswith('__classification'))
        self.assertTrue(samples[1]['id'].endswith('__joint_guidance'))

    def test_two_tasks_are_selected_from_frozen_training_only(self):
        config=read_json(ROOT/'phase3/configs/lora-pilot-v1.json')
        samples=select_samples(config)
        self.assertEqual(len(samples),2)
        train={row['id'] for row in read_json(ROOT/config['data']['train_repo_path'])}
        val={row['id'] for row in read_json(ROOT/config['data']['validation_repo_path'])}
        self.assertTrue(all(row['id'] in train and row['id'] not in val for row in samples))
        self.assertTrue(samples[0]['id'].endswith('__classification'))
        self.assertTrue(samples[1]['id'].endswith('__joint_guidance'))

@unittest.skipUnless(importlib.util.find_spec('torch'),'requires existing local CPU torch environment')
class RealAutogradTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch
        class ToyLoRA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base=torch.nn.Parameter(torch.ones(3,3,dtype=torch.bfloat16),requires_grad=False)
                self.lora_A=torch.nn.ParameterDict({'default':torch.nn.Parameter(torch.ones(2,3,dtype=torch.float32))})
                self.lora_B=torch.nn.ParameterDict({'default':torch.nn.Parameter(torch.zeros(3,2,dtype=torch.float32))})
                self.nonfinite=False;self.mutate=False
            def forward(self,**batch):
                if self.mutate:
                    with torch.no_grad():self.lora_B['default'].add_(1)
                loss=(self.lora_B['default']@self.lora_A['default']).sum()
                if self.nonfinite:loss=loss*float('nan')
                return SimpleNamespace(loss=loss)
        self.model=ToyLoRA()
        self.batch={'labels':torch.tensor([[-100,1,2]])}
    def test_backward_nonzero_B_zero_A_and_unchanged_weights(self):
        before={name:p.detach().clone() for name,p in self.model.named_parameters()}
        result=check_task(self.model,self.batch,self.torch,require_cuda=False)
        self.assertEqual(result['optimizer_steps'],0)
        self.assertTrue(result['backward_passed'])
        self.assertEqual(result['nonzero_gradient_names'],['lora_B.default'])
        for name,p in self.model.named_parameters():self.assertTrue(self.torch.equal(before[name],p.detach()))
        self.assertIsNone(self.model.base.grad)
    def test_int32_preprocess_labels_work_with_real_cross_entropy_after_conversion(self):
        torch=self.torch
        raw={'input_ids':torch.tensor([[1,2]],dtype=torch.int32),
             'labels':torch.tensor([[-100,2]],dtype=torch.int32),
             'attention_mask':torch.tensor([[True,True]])}
        batch=prepare_model_batch(raw,torch,'cpu')
        self.assertEqual(batch['labels'].dtype,torch.int64)
        self.assertEqual(batch['input_ids'].dtype,torch.int64)
        self.assertTrue(torch.equal(batch['labels'],raw['labels']))
        self.assertEqual(batch['attention_mask'].dtype,torch.bool)
        logits=torch.randn(2,3,requires_grad=True)
        loss=torch.nn.functional.cross_entropy(logits,batch['labels'].reshape(-1),ignore_index=-100)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
    def test_floating_labels_rejected_without_truncation(self):
        raw={'input_ids':self.torch.tensor([[1]]),'labels':self.torch.tensor([[1.5]])}
        with self.assertRaisesRegex(ValueError,'integer token'):
            prepare_model_batch(raw,self.torch,'cpu')
    def test_zero_supervision_rejected(self):
        with self.assertRaisesRegex(ValueError,'zero supervised'):
            check_task(self.model,{'labels':self.torch.tensor([[-100,-100]])},self.torch,False)
    def test_nonfinite_loss_rejected(self):
        self.model.nonfinite=True
        with self.assertRaisesRegex(ValueError,'nonfinite loss'):check_task(self.model,self.batch,self.torch,False)
    def test_trainable_base_rejected(self):
        self.model.base.requires_grad_(True)
        with self.assertRaisesRegex(ValueError,'nonvisual LoRA'):parameter_checks(self.model,self.torch,False)
    def test_wrong_adapter_dtype_rejected_without_conversion(self):
        self.model.lora_A['default'].data=self.model.lora_A['default'].data.to(self.torch.bfloat16)
        with self.assertRaisesRegex(ValueError,'float32'):parameter_checks(self.model,self.torch,False)
    def test_parameter_mutation_detected(self):
        self.model.mutate=True
        with self.assertRaisesRegex(ValueError,'weights changed'):check_task(self.model,self.batch,self.torch,False)

if __name__=='__main__':unittest.main()
