import copy
import torch
import pytest
from transformers import Trainer, TrainingArguments
from src.aligned_step import AlignedStepCODI, pack_steps, step_index
from src.model import CODI
from test_all_cuts import make_codi


def test_ordered_mapping_and_targets():
    assert [step_index(k,6) for k in range(1,7)]==[1,2,3,4,5,6]
    assert [step_index(k,2) for k in range(1,7)]==[1,1,1,2,2,2]
    assert [step_index(k,1) for k in range(1,7)]==[1]*6
    assert [step_index(k,4) for k in range(1,7)]==[1,2,2,3,4,4]
    assert [step_index(k,8) for k in range(1,7)]==[2,3,4,6,7,8]
    rows=[[torch.tensor([31,4,1]),torch.tensor([31,5,6,7,1])],[],[torch.tensor([31,9,1])]]
    p=pack_steps(rows,0)
    assert p['step_owners'].tolist()==[0,2]
    assert p['step_labels'][:3,0].tolist()==[[-100,4,1,-100,-100]]*3
    assert p['step_labels'][3:,0].tolist()==[[-100,5,6,7,1]]*3
    assert p['step_input_ids'][:,1,:3].tolist()==[[31,9,1]]*6
    assert pack_steps([[],[]],0)['step_input_ids'].shape==(6,0,1)


def inputs():
    return dict(encoder_input_ids=torch.tensor([[2,3,30],[0,4,30]]),
        decoder_input_ids=torch.tensor([[31,8,1],[31,9,1]]),
        ref_input_ids=torch.tensor([[2,3,5,8,1],[4,6,7,9,1]]),
        labels=torch.tensor([[31,8,1],[31,9,1]]),
        ref_labels=torch.tensor([[-100,-100,5,8,1],[-100,6,7,9,1]]),
        encoder_attention_mask=torch.tensor([[1,1,1],[0,1,1]]),
        ref_answer_position=torch.tensor([3,3]),model_answer_position=torch.tensor([1,1]))


@pytest.mark.parametrize('recompute',[False,True])
def test_equal_k_weights_match_independent_branch_loss_and_gradient(recompute):
    torch.manual_seed(19)
    model=make_codi(AlignedStepCODI); model.auxiliary_recompute=recompute
    reference=make_codi(CODI);reference.load_state_dict(model.state_dict())
    targets=[[torch.tensor([31,4,1]),torch.tensor([31,5,6,7,1])], [torch.tensor([31,10,1])]]
    packed=pack_steps(targets,0);p=inputs()
    actual=model(**p,**packed)
    captured=[]
    def hook(module,args,kw,out):
        if kw.get('inputs_embeds') is not None and kw['inputs_embeds'].shape[1]==1 and kw.get('output_hidden_states'):
            captured.append(out.past_key_values)
    handle=reference.codi.register_forward_hook(hook,with_kwargs=True)
    base=reference(**p);handle.remove();assert len(captured)==6
    per_k=[]
    for k,cache in enumerate(captured):
        sums=[];ns=[]
        for owner,steps in enumerate(targets):
            ids=steps[step_index(k+1,len(steps))-1].unsqueeze(0)
            out=reference.codi(input_ids=ids,past_key_values=tuple(tuple(t[owner:owner+1] for t in layer) for layer in cache))
            sums.append(torch.nn.functional.cross_entropy(out.logits[:,:-1].reshape(-1,32),ids[:,1:].reshape(-1),reduction='sum'))
            ns.append(ids.shape[1]-1)
        per_k.append(sum(sums)/sum(ns))
    expected=base['loss']+.05*sum(per_k)
    torch.testing.assert_close(actual['loss'],expected)
    torch.testing.assert_close(actual['logits'],base['logits'])
    counts=model.last_auxiliary_counts['supervised_tokens_per_k']
    assert counts==[4,4,4,6,6,6]
    # Different sequence lengths must not turn the six equal weights into token-proportional weights.
    assert not torch.isclose(torch.stack(per_k).mean(),sum(x*n for x,n in zip(per_k,counts))/sum(counts),rtol=1e-6,atol=1e-6)
    actual['loss'].backward();expected.backward()
    for a,b in zip(model.parameters(),reference.parameters()):
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=2e-5,rtol=2e-4)
    empty=model(**p,**pack_steps([[],[]],0))
    torch.testing.assert_close(empty['loss'],base['loss']);assert empty['continuation_loss']==0


def test_accumulation_scaling(tmp_path):
    trainer=Trainer(model=make_codi(AlignedStepCODI),args=TrainingArguments(output_dir=str(tmp_path),report_to='none',use_cpu=True,gradient_accumulation_steps=2))
    assert trainer.model_accepts_loss_kwargs is False
