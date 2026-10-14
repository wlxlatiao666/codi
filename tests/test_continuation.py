"""Current cached-continuation helpers; GPU model parity is tested separately."""
from types import SimpleNamespace


import pytest


import torch


from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM


from src.continuation import (
    build_continuation, collate_continuations, continuation_loss,
    fork_cache, latent_checkpoint, split_reasoning_steps,
)


from src.model import CODI


class CharacterTokenizer:
    eos_token_id = 1
    bos_token_id = None
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 10 for c in text]

    def __call__(self, text, max_length=256, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([self.encode(text)[:max_length]]))


def test_splitters_and_suffix_labels():
    assert split_reasoning_steps("<<2*3=6>><<6+4=10>>") == ["<<2*3=6>>", "<<6+4=10>>"]
    assert split_reasoning_steps("2*3=6 6+4=10") == ["2*3=6", "6+4=10"]
    assert split_reasoning_steps("It costs 2.5 dollars. Add 3 dollars.") == ["It costs 2.5 dollars.", "Add 3 dollars."]
    tok = CharacterTokenizer()
    example, status = build_continuation("2*3=6 6+4=10", "The answer is: 10", tok, 2, True)
    expected = tok.encode("6+4=10\nThe answer is: 10") + [1]
    assert status == "eligible"
    assert example["continuation_input_ids"].tolist() == [2] + expected
    assert example["continuation_labels"].tolist() == [-100] + expected
    assert latent_checkpoint(6, 0.5) == 3


def test_skipping_and_padding():
    tok = CharacterTokenizer()
    skipped, reason = build_continuation("one unstructured thought", "4", tok, 2, True)
    assert reason == "too_few_steps"
    long, reason = build_continuation("a=b b=c", "4", tok, 2, True, max_tokens=2)
    assert reason == "target_too_long"
    assert long["continuation_labels"].tolist() == [-100]
    good, _ = build_continuation("a=b b=c", "4", tok, 2, False)
    assert good["continuation_labels"][:2].tolist() == [-100, -100]
    batch = collate_continuations([skipped, good], 0)
    assert batch["continuation_labels"][0].eq(-100).all()
    assert not batch["continuation_attention_mask"][0, 1:].any()


def make_model(kind="gpt2", weight=0.0, fix=True, cls=CODI):
    torch.manual_seed(7)
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    if kind == "gpt2":
        obj.codi = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=2,
            n_head=2, n_positions=128, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
        obj.model_name = "gpt2"
    else:
        obj.codi = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2, max_position_embeddings=128, attention_dropout=0))
        # Token positions below are explicit; avoid the legacy model-name offset.
        obj.model_name = "tiny"
    obj.num_latent = 4
    obj.use_prj = True
    obj.prj = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Tanh())
    obj.fix_attn_mask = fix
    obj.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    obj.distill_loss_fct = torch.nn.SmoothL1Loss()
    obj.distill_loss_div_std = False
    obj.distill_loss_type = "smooth_l1"
    obj.distill_loss_factor = 1.0
    obj.ref_loss_factor = 1.0
    obj.print_loss = False
    obj.continuation_loss_factor = weight
    obj.continuation_checkpoint = 2
    obj.training_args = SimpleNamespace(print_ref_model_stats=False)
    return obj


@pytest.mark.parametrize("kind", ["gpt2", "llama"])
def test_auxiliary_gradient_cache_and_mask(kind):
    obj = make_model(kind)
    prefix = torch.randn(1, 3, 16, requires_grad=True)
    cache = obj.codi(inputs_embeds=prefix, use_cache=True).past_key_values
    def cache_len(c):
        return c[0][0].shape[-2] if isinstance(c, tuple) else c.get_seq_length()
    before = cache_len(cache)
    args = (obj.codi, cache, torch.ones(1, 3, dtype=torch.bool),
            torch.tensor([[31, 11, 12, 1]]), torch.tensor([[-100, 11, 12, 1]]),
            torch.ones(1, 4, dtype=torch.bool))
    loss = continuation_loss(*args)
    loss.backward()
    assert prefix.grad.abs().sum() > 0  # branch did not detach the prefix
    assert cache_len(cache) == before  # DynamicCache branch did not mutate it
    padded = continuation_loss(obj.codi, cache, args[2],
        torch.tensor([[31, 11, 12, 1, 29, 29]]),
        torch.tensor([[-100, 11, 12, 1, -100, -100]]),
        torch.tensor([[True, True, True, True, False, False]]))
    torch.testing.assert_close(loss.detach(), padded.detach())


def test_legacy_cache_clone_is_differentiable():
    t = torch.randn(1, 2, 3, 4, requires_grad=True)
    original = ((t, t * 2),)
    cloned = fork_cache(original)
    assert cloned[0][0].data_ptr() != t.data_ptr()
    cloned[0][1].sum().backward()
    torch.testing.assert_close(t.grad, torch.full_like(t, 2))
