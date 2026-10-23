"""Exhaustive continuation: all completed latent positions × all proper suffixes.

Auxiliary CE is the mean over all supervised (k, sample, suffix, token) entries.
The caller applies continuation_weight once. No new model parameters are added.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model import CODI


def pack_suffixes(rows, pad_id):
    """Keep each proper suffix once; retain its source sample's cache row."""
    ids, owners = [], []
    for owner, suffixes in enumerate(rows):
        for suffix in suffixes:
            if len(suffix) > 1:  # Historical ineligible rows contain only EOT.
                ids.append(suffix)
                owners.append(owner)
    if not ids:
        inputs = torch.empty((0, 1), dtype=torch.long)
        labels = inputs.clone()
    else:
        inputs = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=-100)
        labels[:, 0] = -100
    return dict(all_suffix_input_ids=inputs, all_suffix_labels=labels,
                all_suffix_owners=torch.tensor(owners, dtype=torch.long))


def suffix_token_sum(model, cache, ids, labels, owners, chunk_size=16, recompute=True):
    """Sum token CE at ONE k; differentiable cache row replication, no detach.

    Chunking bounds logits memory. Non-reentrant checkpointing recomputes only
    auxiliary forwards and preserves their dropout RNG during backward.
    """
    if not isinstance(cache, tuple):
        raise TypeError("This experiment is audited for GPT-2 legacy tuple caches only.")
    flat_cache = tuple(t for layer in cache for t in layer)
    if len(flat_cache) != 2 * len(cache):
        raise ValueError("Expected one key/value pair per layer")
    total = next(model.parameters()).new_zeros((), dtype=torch.float32)
    for begin in range(0, len(ids), chunk_size):
        sub_ids = ids[begin:begin + chunk_size]
        sub_labels = labels[begin:begin + chunk_size]
        sub_owners = owners[begin:begin + chunk_size]
        # Trimming ignored right padding leaves earlier causal predictions intact.
        width = int(sub_labels.ne(-100).sum(-1).max().item()) + 1
        sub_ids, sub_labels = sub_ids[:, :width], sub_labels[:, :width]

        def branch(input_ids, targets, row_indices, *cache_tensors):
            copied = tuple((cache_tensors[i].index_select(0, row_indices),
                            cache_tensors[i + 1].index_select(0, row_indices))
                           for i in range(0, len(cache_tensors), 2))
            outputs = model(input_ids=input_ids, past_key_values=copied,
                            attention_mask=None, use_cache=True, output_hidden_states=False)
            return F.cross_entropy(outputs.logits[:, :-1].float().reshape(-1, outputs.logits.size(-1)),
                                   targets[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")

        args = (sub_ids, sub_labels, sub_owners, *flat_cache)
        value = (checkpoint(branch, *args, use_reentrant=False, preserve_rng_state=True)
                 if recompute and torch.is_grad_enabled() else branch(*args))
        total = total + value
    return total


class AllCutsCODI(CODI):
    """Attach auxiliary losses without changing the historical main forward."""
    # The wrapper's **kwargs must not make HF Trainer infer batch-item-aware
    # loss normalization. Like CODI, we return a microbatch mean.
    accepts_loss_kwargs = False
    auxiliary_chunk_size = 64
    auxiliary_recompute = True

    def forward(self, *args, all_suffix_input_ids=None, all_suffix_labels=None,
                all_suffix_owners=None, **kwargs):
        assert kwargs.get("continuation_checkpoint_override") is None
        if all_suffix_input_ids is None:
            raise ValueError("Exhaustive training requires the full suffix bank")
        assert all_suffix_labels is not None and all_suffix_owners is not None
        assert not self.model_args.use_decoder
        token_count = all_suffix_labels[:, 1:].ne(-100).sum()
        sums, lengths = [], []

        def after_latent(module, positional, named, output):
            embeddings = named.get("inputs_embeds")
            if (embeddings is None or embeddings.shape[1] != 1
                    or not named.get("output_hidden_states")
                    or named.get("past_key_values") is None):
                return
            cache = output.past_key_values
            lengths.append(cache[0][0].shape[-2])
            sums.append(suffix_token_sum(
                self.codi, cache, all_suffix_input_ids, all_suffix_labels, all_suffix_owners,
                chunk_size=self.auxiliary_chunk_size, recompute=self.auxiliary_recompute))

        handle = (self.codi.register_forward_hook(after_latent, with_kwargs=True)
                  if len(all_suffix_input_ids) else None)
        try:
            result = super().forward(*args, **kwargs)
        finally:
            if handle is not None:
                handle.remove()
        if len(all_suffix_input_ids):
            assert len(sums) == self.num_latent, (len(sums), self.num_latent)
            assert lengths == list(range(lengths[0], lengths[0] + self.num_latent))
            auxiliary = torch.stack(sums).sum() / (self.num_latent * token_count)
        else:
            auxiliary = result["loss"].new_zeros(())
        result["loss"] = result["loss"] + self.training_args.continuation_weight * auxiliary
        result["continuation_loss"] = auxiliary.detach()
        # Useful for verification; no checkpoint state or trainable parameters.
        self.last_auxiliary_counts = dict(latent_cuts=len(sums), suffixes=len(all_suffix_input_ids),
                                         supervised_tokens=int(token_count.item()) * len(sums))
        return result
