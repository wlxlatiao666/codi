"""Each latent predicts exactly one ordered explicit step, with equal k weights."""
import torch
from .model import CODI
from .all_cuts import suffix_token_sum


def step_index(k, m, num_latent=6):
    """One-based j=ceil(k*m/K). No prefix text is supplied to the decoder."""
    if not 1 <= k <= num_latent or m < 1:
        raise ValueError((k, m, num_latent))
    return (k * m + num_latent - 1) // num_latent


def pack_steps(rows, pad_id, num_latent=6):
    owners = [i for i, steps in enumerate(rows) if steps]
    sequences = [[rows[i][step_index(k, len(rows[i]), num_latent)-1] for i in owners]
                 for k in range(1, num_latent+1)]
    width = max((len(s) for seq in sequences for s in seq), default=1)
    ids = torch.full((num_latent, len(owners), width), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    for k, seq in enumerate(sequences):
        for b, s in enumerate(seq):
            assert len(s) > 1
            ids[k,b,:len(s)] = s
            labels[k,b,1:len(s)] = s[1:]
    return dict(step_input_ids=ids, step_labels=labels,
                step_owners=torch.tensor(owners, dtype=torch.long))


class AlignedStepCODI(CODI):
    accepts_loss_kwargs = False
    auxiliary_chunk_size = 64
    auxiliary_recompute = True

    def forward(self, *args, step_input_ids=None, step_labels=None, step_owners=None, **kwargs):
        assert kwargs.get('continuation_checkpoint_override') is None
        assert not self.model_args.use_decoder
        if step_input_ids is None or step_labels is None or step_owners is None:
            raise ValueError('Aligned-step training requires per-k single-step targets')
        assert step_input_ids.shape == step_labels.shape
        assert step_input_ids.shape[0] == self.num_latent
        counts = step_labels[:,:,1:].ne(-100).sum(dim=(1,2))
        losses, lengths = [], []

        def after_latent(module, positional, named, output):
            emb = named.get('inputs_embeds')
            if (emb is None or emb.shape[1] != 1 or not named.get('output_hidden_states')
                    or named.get('past_key_values') is None):
                return
            k = len(losses)
            assert k < self.num_latent
            cache = output.past_key_values
            lengths.append(cache[0][0].shape[-2])
            value = suffix_token_sum(self.codi, cache, step_input_ids[k], step_labels[k],
                step_owners, self.auxiliary_chunk_size, self.auxiliary_recompute)
            losses.append(value / counts[k])

        handle = self.codi.register_forward_hook(after_latent, with_kwargs=True) if len(step_owners) else None
        try:
            result = super().forward(*args, **kwargs)
        finally:
            if handle is not None:
                handle.remove()
        if len(step_owners):
            assert len(losses) == self.num_latent
            assert lengths == list(range(lengths[0], lengths[0]+self.num_latent))
            auxiliary = torch.stack(losses).mean()
        else:
            auxiliary = result['loss'].new_zeros(())
        result['loss'] = result['loss'] + self.training_args.continuation_weight * auxiliary
        result['continuation_loss'] = auxiliary.detach()
        self.last_auxiliary_counts = dict(latent_cuts=len(losses), samples=len(step_owners),
            supervised_tokens_per_k=counts.detach().cpu().tolist(),
            per_k_loss=[float(v.detach()) for v in losses])
        return result
