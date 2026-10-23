import copy
import pytest
import torch
from transformers import Trainer,TrainingArguments
from src.window_suffix import WindowSuffixCODI,window_starts,pack_windows,weighted_suffix_loss
from src.model import CODI
from test_all_cuts import make_codi,tiny
from test_aligned_step import inputs


def test_window_mapping_and_equal_sample_candidate_weights():
    assert [window_starts(k,6) for k in range(1,7)]==[[1,2,3],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[4,5,6]]
    for m in range(1,13):
        last=0
        for k in range(1,7):
            js=window_starts(k,m)
            assert len(js)==min(3,m) and len(set(js))==len(js)
            assert min(js)>=1 and max(js)<=m and js==list(range(min(js),max(js)+1))
            assert min(js)>=last;last=min(js)
    rows=[[torch.tensor([31,4,1])],[],[torch.tensor([31,5,1]),torch.tensor([31,6,7,1])]]
    p=pack_windows(rows,0)
    for ids,labels,owners,weights in zip(p['window_input_ids'],p['window_labels'],p['window_owners'],p['window_weights']):
        assert owners.tolist()==[0,2,2]
        torch.testing.assert_close(weights,torch.tensor([.5,.25,.25]))
        assert labels[:,0].tolist()==[-100]*3
        assert labels[0].tolist()==[-100,4,1,-100]
    assert all(t.shape==(0,1) for t in pack_windows([[],[]],0)['window_input_ids'])


@pytest.mark.parametrize('recompute',[False,True])
def test_full_forward_matches_explicit_equal_weight_loss_and_gradient(recompute):
    torch.manual_seed(23)
    a=make_codi(WindowSuffixCODI);a.auxiliary_recompute=recompute;a.auxiliary_chunk_size=2
    b=make_codi(CODI);b.load_state_dict(a.state_dict())
    rows=[[torch.tensor([31,5,6,7,1]),torch.tensor([31,6,7,1]),torch.tensor([31,7,1]),torch.tensor([31,8,1])],[torch.tensor([31,9,1])]]
    result=a(**inputs(),**pack_windows(rows,0))
    caches=[]
    def hook(module,args,kw,out):
        if kw.get('inputs_embeds') is not None and kw['inputs_embeds'].shape[1]==1 and kw.get('output_hidden_states'):caches.append(out.past_key_values)
    h=b.codi.register_forward_hook(hook,with_kwargs=True);base=b(**inputs());h.remove()
    terms=[]
    for k,cache in enumerate(caches,1):
        for owner,row in enumerate(rows):
            js=window_starts(k,len(row))
            for j in js:
                ids=row[j-1].unsqueeze(0)
                out=b.codi(input_ids=ids,past_key_values=tuple(tuple(t[owner:owner+1] for t in layer) for layer in cache))
                terms.append(.3/6/len(rows)/len(js)*torch.nn.functional.cross_entropy(out.logits[:,:-1].reshape(-1,32),ids[:,1:].reshape(-1)))
    expected=base['loss']+sum(terms)
    torch.testing.assert_close(result['loss'],expected)
    torch.testing.assert_close(result['logits'],base['logits'])
    assert a.last_auxiliary_counts['latent_cuts']==6
    result['loss'].backward();expected.backward()
    for x,y in zip(a.parameters(),b.parameters()):
        if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,atol=2e-5,rtol=2e-4)
    empty=a(**inputs(),**pack_windows([[],[]],0))
    torch.testing.assert_close(empty['loss'],base['loss']);assert empty['continuation_loss']==0


def test_weighted_checkpoint_preserves_dropout_rng_and_cache():
    torch.manual_seed(12);a=tiny(.1);b=copy.deepcopy(a);p=torch.randn(2,3,16)
    packed=pack_windows([[torch.tensor([31,4,5,1]),torch.tensor([31,5,1])],[torch.tensor([31,6,1])]],0)
    reports=[]
    for model,recompute in [(a,False),(b,True)]:
        torch.manual_seed(17);prefix=p.clone().requires_grad_();cache=model(inputs_embeds=prefix,use_cache=True).past_key_values
        saved=tuple(tuple(t.detach().clone() for t in layer) for layer in cache)
        args=[packed['window_'+key][0] for key in ['input_ids','labels','owners','weights']]
        loss=weighted_suffix_loss(model,cache,*args,chunk_size=2,recompute=recompute);loss.backward()
        assert all(torch.equal(x,y) for l1,l2 in zip(cache,saved) for x,y in zip(l1,l2))
        assert prefix.grad.abs().sum()>0
        reports.append((loss.detach(),prefix.grad,[t.grad for t in model.parameters()],torch.get_rng_state()))
    for i in [0,1]:torch.testing.assert_close(reports[0][i],reports[1][i],rtol=0,atol=0)
    for x,y in zip(reports[0][2],reports[1][2]):torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert torch.equal(reports[0][3],reports[1][3])


def test_accumulation_scaling(tmp_path):
    trainer=Trainer(model=make_codi(WindowSuffixCODI),args=TrainingArguments(output_dir=str(tmp_path),report_to='none',use_cpu=True,gradient_accumulation_steps=2))
    assert trainer.model_accepts_loss_kwargs is False
