from types import SimpleNamespace
import copy

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

from src.all_cuts import AllCutsCODI, pack_suffixes, suffix_token_sum
from src.model import CODI


def tiny(dropout=0):
    return GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=2, n_head=2,
        n_positions=128, resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout))


def test_pack_suffixes_and_empty():
    out = pack_suffixes([[torch.tensor([31, 4, 5, 1]), torch.tensor([31, 5, 1])],
                        [torch.tensor([31])]], 0)
    assert out["all_suffix_owners"].tolist() == [0, 0]
    assert out["all_suffix_labels"].tolist() == [[-100, 4, 5, 1], [-100, 5, 1, -100]]
    assert pack_suffixes([[torch.tensor([31])]], 0)["all_suffix_input_ids"].shape == (0, 1)


@pytest.mark.parametrize("recompute", [False, True])
def test_all_suffixes_match_explicit_enumeration_and_gradients(recompute):
    torch.manual_seed(11)
    model = tiny()
    prefix = torch.randn(2, 4, 16, requires_grad=True)
    cache = model(inputs_embeds=prefix, use_cache=True).past_key_values
    saved = tuple(tuple(t.detach().clone() for t in layer) for layer in cache)
    packed = pack_suffixes([[torch.tensor([31, 4, 5, 1]), torch.tensor([31, 5, 1])],
                           [torch.tensor([31, 7, 1])]], 0)
    ids, labels, owners = [packed[k] for k in
                          ["all_suffix_input_ids", "all_suffix_labels", "all_suffix_owners"]]
    actual = suffix_token_sum(model, cache, ids, labels, owners, 2, recompute)
    expected = prefix.new_zeros(())
    for row, owner in enumerate(owners.tolist()):
        subcache = tuple(tuple(t[owner:owner + 1].clone() for t in layer) for layer in cache)
        out = model(input_ids=ids[row:row + 1], past_key_values=subcache)
        expected = expected + torch.nn.functional.cross_entropy(
            out.logits[:, :-1].reshape(-1, 32), labels[row, 1:], ignore_index=-100, reduction="sum")
    params = (prefix,) + tuple(model.parameters())
    ga = torch.autograd.grad(actual, params, retain_graph=True)
    gb = torch.autograd.grad(expected, params, retain_graph=True)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
    assert ga[0].abs().sum() > 0
    assert all(torch.equal(a, b) for la, lb in zip(cache, saved) for a, b in zip(la, lb))
    # Replicating all branches must not multiply the normalized loss.
    double = suffix_token_sum(model, cache, ids.repeat(2, 1), labels.repeat(2, 1),
                              owners.repeat(2), 2, recompute)
    torch.testing.assert_close(double / (2 * labels[:, 1:].ne(-100).sum()),
                               actual / labels[:, 1:].ne(-100).sum())


def test_checkpoint_preserves_dropout_gradient_and_rng():
    torch.manual_seed(3)
    a = tiny(.1)
    b = copy.deepcopy(a)
    prefix = torch.randn(2, 3, 16)
    packed = pack_suffixes([[torch.tensor([31, 4, 5, 1])], [torch.tensor([31, 7, 1])]], 0)
    records = []
    for model, recompute in [(a, False), (b, True)]:
        torch.manual_seed(17)
        p = prefix.clone().requires_grad_()
        cache = model(inputs_embeds=p, use_cache=True).past_key_values
        loss = suffix_token_sum(model, cache, packed["all_suffix_input_ids"],
            packed["all_suffix_labels"], packed["all_suffix_owners"], 1, recompute)
        loss.backward()
        records.append((loss.detach(), p.grad, [t.grad for t in model.parameters()], torch.get_rng_state()))
    torch.testing.assert_close(records[0][0], records[1][0], rtol=0, atol=0)
    torch.testing.assert_close(records[0][1], records[1][1], rtol=0, atol=0)
    for x, y in zip(records[0][2], records[1][2]):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert torch.equal(records[0][3], records[1][3])


def make_codi(cls):
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.codi = tiny()
    model.model_args = SimpleNamespace(use_decoder=False)
    model.training_args = SimpleNamespace(continuation_weight=.3, print_ref_model_stats=False)
    model.model_name = "gpt2"
    model.num_latent = 6
    model.use_prj = False
    model.fix_attn_mask = False
    model.print_loss = False
    model.ref_loss_factor = model.distill_loss_factor = 1.
    model.distill_loss_div_std = False
    model.distill_loss_type = "smooth_l1"
    model.distill_loss_fct = torch.nn.SmoothL1Loss()
    model.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    return model


def test_whole_forward_all_six_positions_and_loss_composition():
    torch.manual_seed(23)
    model = make_codi(AllCutsCODI)
    baseline = make_codi(CODI)
    baseline.load_state_dict(model.state_dict())
    inputs = dict(encoder_input_ids=torch.tensor([[2, 3, 30], [0, 4, 30]]),
        decoder_input_ids=torch.tensor([[31, 8, 1], [31, 9, 1]]),
        ref_input_ids=torch.tensor([[2, 3, 5, 8, 1], [4, 6, 7, 9, 1]]),
        labels=torch.tensor([[31, 8, 1], [31, 9, 1]]),
        ref_labels=torch.tensor([[-100, -100, 5, 8, 1], [-100, 6, 7, 9, 1]]),
        encoder_attention_mask=torch.tensor([[1, 1, 1], [0, 1, 1]]),
        ref_answer_position=torch.tensor([3, 3]), model_answer_position=torch.tensor([1, 1]))
    packed = pack_suffixes([[torch.tensor([31, 5, 8, 1]), torch.tensor([31, 8, 1])],
                           [torch.tensor([31, 7, 9, 1])]], 0)
    out = model(**inputs, **packed)
    base = baseline(**inputs)
    assert model.last_auxiliary_counts == dict(latent_cuts=6, suffixes=3, supervised_tokens=48)
    torch.testing.assert_close(out["logits"], base["logits"])
    torch.testing.assert_close(out["loss"], base["loss"] + .3 * out["continuation_loss"])
    out["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    empty = model(**inputs, **pack_suffixes([[torch.tensor([31])], [torch.tensor([31])]], 0))
    torch.testing.assert_close(empty["loss"], base["loss"])
    assert empty["continuation_loss"] == 0


def test_trainer_preserves_original_accumulation_scaling(tmp_path):
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none",
                             use_cpu=True, gradient_accumulation_steps=2)
    for cls in (CODI, AllCutsCODI):
        trainer = Trainer(model=make_codi(cls), args=args)
        assert trainer.model_accepts_loss_kwargs is False
