"""Experiment B: single-path continuation from an intermediate latent cache."""

import math
import re

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def split_reasoning_steps(text):
    """Conservative splitter for GSM8k-Aug equations and GSM8k-Aug-NL.

    Unrecognized prose stays one step (and is ineligible), rather than being
    arbitrarily cut into words. This is a formatting heuristic, not a parser.
    """
    text = text.strip()
    if not text:
        return []
    equations = re.findall(r"<<[^<>]+>>", text)
    if equations and not re.sub(r"<<[^<>]+>>", "", text).strip():
        return equations
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1:
        return sentences
    words = text.split()
    if len(words) > 1 and all("=" in word for word in words):
        return words
    return [text]


def latent_checkpoint(num_latent, cut_ratio):
    if num_latent < 2 or not 0 < cut_ratio < 1:
        raise ValueError("Continuation requires num_latent >= 2 and 0 < continuation_cut_ratio < 1.")
    return min(num_latent - 1, max(1, math.ceil(num_latent * cut_ratio)))


def build_continuation(cot, answer, tokenizer, eot_id, remove_eos,
                       cut_ratio=0.5, max_tokens=256):
    """Build [EOT, optional EOS, suffix, answer, EOS], without the CoT prefix.

    Return an all-masked placeholder for ineligible examples. Never truncate a
    target, since truncation would change the supervised reasoning/answer.
    """
    steps = split_reasoning_steps(cot)
    empty = {"continuation_input_ids": torch.tensor([eot_id]),
             "continuation_labels": torch.tensor([IGNORE_INDEX])}
    if len(steps) < 2:
        return empty, "too_few_steps"
    cut = min(len(steps) - 1, max(1, math.ceil(len(steps) * cut_ratio)))
    suffix = "\n".join(steps[cut:]) + "\n" + answer
    markers = [eot_id] + ([] if remove_eos else [tokenizer.eos_token_id])
    body = tokenizer.encode(suffix, add_special_tokens=False)
    ids = markers + body + [tokenizer.eos_token_id]
    if len(ids) > max_tokens:
        return empty, "target_too_long"
    labels = [IGNORE_INDEX] * len(markers) + body + [tokenizer.eos_token_id]
    return {"continuation_input_ids": torch.tensor(ids, dtype=torch.long),
            "continuation_labels": torch.tensor(labels, dtype=torch.long)}, "eligible"


def collate_continuations(instances, pad_token_id):
    ids = torch.nn.utils.rnn.pad_sequence(
        [x["continuation_input_ids"] for x in instances],
        batch_first=True, padding_value=pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(
        [x["continuation_labels"] for x in instances],
        batch_first=True, padding_value=IGNORE_INDEX)
    return dict(continuation_input_ids=ids, continuation_labels=labels,
                continuation_attention_mask=ids.ne(pad_token_id))


def fork_cache(cache):
    """Copy cache storage WITHOUT detach: auxiliary gradients reach the rollout.

    DynamicCache.update mutates the container. deepcopy fails on non-leaf
    tensors; a shallow copy would leak the continuation into the main branch.
    Supports the legacy tuple and DynamicCache interfaces in transformers 4.52.
    """
    if isinstance(cache, (tuple, list)):
        return tuple(tuple(t.clone() for t in layer) for layer in cache)
    if hasattr(cache, "to_legacy_cache") and hasattr(type(cache), "from_legacy_cache"):
        return type(cache).from_legacy_cache(fork_cache(cache.to_legacy_cache()))
    raise TypeError(f"Unsupported cache type for continuation: {type(cache).__name__}")


def position_ids_from_mask(mask, current_length):
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(~mask.bool(), 0)
    return positions[:, -current_length:]


def continuation_loss(model, cache, prefix_mask, input_ids, labels, target_mask):
    """One teacher-forced suffix CE, normalized over valid next-token targets."""
    valid = labels[:, 1:].ne(IGNORE_INDEX)
    if not valid.any():
        return next(model.parameters()).new_zeros(())
    mask = torch.cat((prefix_mask.bool(), target_mask.bool()), dim=1)
    outputs = model(
        input_ids=input_ids, past_key_values=fork_cache(cache),
        attention_mask=mask,
        position_ids=position_ids_from_mask(mask, input_ids.size(1)),
        use_cache=True, output_hidden_states=False,
    )
    logits = outputs.logits[:, :-1].contiguous()
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels[:, 1:].contiguous().reshape(-1),
        ignore_index=IGNORE_INDEX, reduction="sum",
    ) / valid.sum()
