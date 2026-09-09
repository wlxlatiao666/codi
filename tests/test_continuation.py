"""CPU tests using random tiny GPT-2/Llama; no downloads or training data."""
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


def inputs():
    encoder = torch.tensor([[2, 3, 4, 30], [29, 2, 5, 30]])
    decoder = torch.tensor([[31, 6, 7, 8, 1], [31, 6, 7, 9, 1]])
    ref = torch.tensor([[2, 3, 4, 10, 11, 6, 7, 8, 1], [2, 5, 10, 11, 6, 7, 9, 1, 29]])
    ref_labels = ref.clone()
    ref_labels[0, :3] = -100
    ref_labels[1, :2] = -100
    ref_labels[ref == 29] = -100
    suffix = torch.tensor([[31, 11, 12, 8, 1], [31, 11, 12, 9, 1]])
    suffix_labels = suffix.clone()
    suffix_labels[:, 0] = -100
    return dict(encoder_input_ids=encoder, decoder_input_ids=decoder,
        labels=decoder, encoder_attention_mask=encoder.ne(29), ref_input_ids=ref,
        ref_labels=ref_labels, ref_attention_mask=ref.ne(29),
        ref_answer_position=torch.tensor([7, 6]), model_answer_position=torch.tensor([3, 3]),
        continuation_input_ids=suffix, continuation_labels=suffix_labels,
        continuation_attention_mask=torch.ones_like(suffix, dtype=torch.bool))


@pytest.mark.parametrize("kind", ["gpt2", "llama"])
def test_branch_isolation_loss_and_backward(kind):
    baseline, experiment = make_model(kind), make_model(kind, 0.3)
    a = baseline(**inputs())
    b = experiment(**inputs())
    # The correct teacher-forced suffix must never contaminate the answer cache.
    torch.testing.assert_close(a["logits"], b["logits"])
    assert b["continuation_loss"] > 0
    torch.testing.assert_close(b["loss"] - a["loss"], b["loss"].new_tensor(0.3 * b["continuation_loss"]))
    b["loss"].backward()
    assert experiment.prj[0].weight.grad.abs().sum() > 0


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


@pytest.mark.parametrize("kind", ["gpt2", "llama"])
def test_left_padding_invariance(kind):
    obj = make_model(kind, 0.3)
    batched = inputs()
    single = {k: v[1:2].clone() for k, v in batched.items()}
    for key in ("encoder_input_ids", "encoder_attention_mask"):
        single[key] = single[key][:, 1:]
    for key in ("ref_input_ids", "ref_attention_mask", "ref_labels"):
        single[key] = single[key][:, :-1]
    a, b = obj(**batched), obj(**single)
    torch.testing.assert_close(a["logits"][1:2], b["logits"], atol=1e-5, rtol=1e-4)


def test_all_ineligible_and_disabled_branch():
    obj = make_model(weight=0.3)
    batch = inputs()
    batch["continuation_labels"].fill_(-100)
    out = obj(**batch)
    assert out["continuation_loss"] == 0
    assert torch.isfinite(out["loss"])
    obj.continuation_loss_factor = 0
    assert obj(**{k: v for k, v in batch.items() if not k.startswith("continuation_")})["continuation_loss"] == 0


def test_legacy_cache_clone_is_differentiable():
    t = torch.randn(1, 2, 3, 4, requires_grad=True)
    original = ((t, t * 2),)
    cloned = fork_cache(original)
    assert cloned[0][0].data_ptr() != t.data_ptr()
    cloned[0][1].sum().backward()
    torch.testing.assert_close(t.grad, torch.full_like(t, 2))


def test_trainer_collator_accepts_continuation_columns():
    # The official collator is nested inside train(); extract the actual class
    # to exercise its integration without loading a pretrained model/dataset.
    import ast
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Dict, Sequence
    import transformers
    tree = ast.parse((Path(__file__).parents[1] / "train.py").read_text())
    train_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    cls = next(n for n in train_fn.body if isinstance(n, ast.ClassDef) and n.name == "DataCollatorForSupervisedDataset")
    env = dict(torch=torch, dataclass=dataclass, Dict=Dict, Sequence=Sequence,
               transformers=transformers, IGNORE_INDEX=-100,
               collate_continuations=collate_continuations)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "train.py", "exec"), env)
    data = inputs()
    instances = [{k: v[i] for k, v in data.items() if not k.endswith("attention_mask")} for i in range(2)]
    batch = env["DataCollatorForSupervisedDataset"](SimpleNamespace(pad_token_id=29))(instances)
    assert batch["continuation_labels"].shape == (2, 5)
    assert batch["encoder_attention_mask"][1, 0].item() is False


def test_real_dataset_constructs_suffixes_and_keeps_ineligible_rows():
    import ast
    from collections import Counter
    from pathlib import Path
    from typing import Dict, Sequence
    import logging
    import transformers
    from torch.utils.data import Dataset
    from train import _tokenize_fn
    tree = ast.parse((Path(__file__).parents[1] / "train.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    nodes = [n for n in fn.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and
             n.name in ("get_answer_token_position", "preprocess", "SupervisedDataset")]
    args = SimpleNamespace(exp_mode=False, include_last_cot=False, max_token_num=2000,
        remove_eos=True, continuation_loss_factor=0.3, continuation_cut_ratio=0.5,
        continuation_max_tokens=1024, model_max_length=1024, num_latent=4)
    model = SimpleNamespace(continuation_checkpoint=2,
        codi=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1024)))
    env = dict(torch=torch, logging=logging, Dataset=Dataset, Sequence=Sequence,
        Dict=Dict, transformers=transformers, training_args=args,
        data_args=SimpleNamespace(data_name="icot"), _tokenize_fn=_tokenize_fn,
        Counter=Counter, model=model, build_continuation=build_continuation,
        IGNORE_INDEX=-100)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "train.py", "exec"), env)
    tok = CharacterTokenizer()
    raw = [dict(question="Compute.", cot="2*3=6 6+4=10 10*2=20", answer="20"),
           dict(question="Compute.", cot="2*3=6 6+4=10", answer="10"),
           dict(question="Compute.", cot=" ".join(["1+1=2"] * 60), answer="2")]
    dataset = env["SupervisedDataset"]("icot", raw, tok, 3, 2)
    assert len(dataset) == 3  # eligibility never changes the baseline dataset
    assert dataset[0]["continuation_input_ids"].tolist() == [2] + tok.encode("6+4=10\nThe answer is: 20") + [1]
    assert dataset[1]["continuation_labels"].eq(-100).all()
    assert dataset[2]["continuation_labels"].eq(-100).all()


def test_two_optimizer_steps_with_real_custom_trainer(tmp_path):
    from train import CustomTrainer
    from transformers import TrainingArguments
    obj = make_model(weight=0.3)
    data = inputs()
    rows = [{k: v[i] for k, v in data.items()} for i in range(2)]
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, max_steps=2,
        per_device_train_batch_size=2, learning_rate=1e-3, save_strategy="no",
        report_to="none", logging_steps=1, disable_tqdm=True, dataloader_pin_memory=False)
    before = obj.prj[0].weight.detach().clone()
    trainer = CustomTrainer(model=obj, args=args, train_dataset=rows)
    trainer.train()
    assert trainer.state.global_step == 2
    assert not torch.equal(before, obj.prj[0].weight.detach())
    assert any(x.get("continuation_loss", 0) > 0 for x in trainer.state.log_history)
