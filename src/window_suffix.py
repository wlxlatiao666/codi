"""Local ordered windows of <=3 suffix starts, equal k/sample/candidate weights."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .model import CODI


def window_starts(k, m, num_latent=6):
    if not 1 <= k <= num_latent or m < 1:
        raise ValueError((k,m,num_latent))
    width=min(3,m)
    center=(k*m+num_latent-1)//num_latent
    start=min(max(center-1,1),m-width+1)
    return list(range(start,start+width))


def pack_windows(rows,pad_id,num_latent=6):
    valid=sum(bool(row) for row in rows)
    result={f'window_{name}':[] for name in ['input_ids','labels','owners','weights']}
    for k in range(1,num_latent+1):
        seq=[];owners=[];weights=[]
        for owner,row in enumerate(rows):
            if not row:continue
            starts=window_starts(k,len(row),num_latent)
            for j in starts:
                assert len(row[j-1])>1
                seq.append(row[j-1]);owners.append(owner);weights.append(1/(valid*len(starts)))
        ids=torch.nn.utils.rnn.pad_sequence(seq,batch_first=True,padding_value=pad_id) if seq else torch.empty((0,1),dtype=torch.long)
        labels=torch.nn.utils.rnn.pad_sequence(seq,batch_first=True,padding_value=-100) if seq else ids.clone()
        if seq:labels[:,0]=-100
        for name,value in zip(['input_ids','labels','owners','weights'],[ids,labels,torch.tensor(owners,dtype=torch.long),torch.tensor(weights,dtype=torch.float32)]):
            result['window_'+name].append(value)
    return result


def weighted_suffix_loss(model,cache,ids,labels,owners,weights,chunk_size=64,recompute=True):
    assert isinstance(cache,tuple)
    flat=tuple(t for layer in cache for t in layer)
    total=next(model.parameters()).new_zeros((),dtype=torch.float32)
    for begin in range(0,len(ids),chunk_size):
        targets=labels[begin:begin+chunk_size]
        width=int(targets[:,1:].ne(-100).sum(-1).max().item())+1
        def branch(input_ids,targets,row_indices,row_weights,*cache_tensors):
            copied=tuple((cache_tensors[i].index_select(0,row_indices),cache_tensors[i+1].index_select(0,row_indices)) for i in range(0,len(cache_tensors),2))
            out=model(input_ids=input_ids,past_key_values=copied,attention_mask=None,use_cache=True,output_hidden_states=False)
            ce=F.cross_entropy(out.logits[:,:-1].float().reshape(-1,out.logits.size(-1)),targets[:,1:].reshape(-1),ignore_index=-100,reduction='none').reshape(len(targets),-1)
            lengths=targets[:,1:].ne(-100).sum(-1)
            return (ce.sum(-1)/lengths*row_weights).sum()
        args=(ids[begin:begin+chunk_size,:width],targets[:,:width],owners[begin:begin+chunk_size],weights[begin:begin+chunk_size],*flat)
        total=total+(checkpoint(branch,*args,use_reentrant=False,preserve_rng_state=True) if recompute and torch.is_grad_enabled() else branch(*args))
    return total


class WindowSuffixCODI(CODI):
    accepts_loss_kwargs=False
    auxiliary_chunk_size=64
    auxiliary_recompute=True

    def forward(self,*args,window_input_ids=None,window_labels=None,window_owners=None,window_weights=None,**kwargs):
        assert kwargs.get('continuation_checkpoint_override') is None
        assert not self.model_args.use_decoder
        if any(v is None for v in [window_input_ids,window_labels,window_owners,window_weights]):
            raise ValueError('Window continuation requires packed targets')
        assert all(len(v)==self.num_latent for v in [window_input_ids,window_labels,window_owners,window_weights])
        losses=[];lengths=[]
        def after_latent(module,positional,named,output):
            emb=named.get('inputs_embeds')
            if emb is None or emb.shape[1]!=1 or not named.get('output_hidden_states') or named.get('past_key_values') is None:return
            k=len(losses);assert k<self.num_latent
            cache=output.past_key_values;lengths.append(cache[0][0].shape[-2])
            losses.append(weighted_suffix_loss(self.codi,cache,window_input_ids[k],window_labels[k],window_owners[k],window_weights[k],self.auxiliary_chunk_size,self.auxiliary_recompute))
        handle=self.codi.register_forward_hook(after_latent,with_kwargs=True) if len(window_owners[0]) else None
        try:result=super().forward(*args,**kwargs)
        finally:
            if handle is not None:handle.remove()
        if len(window_owners[0]):
            assert len(losses)==self.num_latent
            assert lengths==list(range(lengths[0],lengths[0]+self.num_latent))
            auxiliary=torch.stack(losses).mean()
        else:auxiliary=result['loss'].new_zeros(())
        result['loss']=result['loss']+self.training_args.continuation_weight*auxiliary
        result['continuation_loss']=auxiliary.detach()
        self.last_auxiliary_counts=dict(latent_cuts=len(losses),branches_per_k=[len(x) for x in window_owners],per_k_loss=[float(x.detach()) for x in losses])
        return result
