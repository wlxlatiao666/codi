import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPTNeoXForCausalLM
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataclasses import dataclass, field
from typing import Optional
from peft import (
    get_peft_model,
    PeftModel,
    PeftConfig
)
from torch.nn.functional import gelu
from safetensors.torch import load_file
from transformers.modeling_outputs import ModelOutput
import random
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    separate_decoder_name: str = field(default="")
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    full_precision: bool = field(default=True, metadata={"help": "whether use int4 for the base model"})
    train: bool = field(
        default=True,
        metadata={
            "help": "if true, the model ckpt will be initialized for training; else, it's for inference"
        },
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    ckpt_dir: Optional[str] = field(default=None, metadata={"help": "checkpoint dir for inference."})

@dataclass
class DataArguments:
    data_name: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    data_path: Optional[str] = field(
        default=None, metadata={"help": "Local path to load dataset from. If set and valid, will override Hugging Face Hub loading."}
    )
    debug_data: bool = field(
        default=False,
        metadata={
            "help": "Enable debug dataset to quickly verify the training process"
        },
    )
    batch_size: int = field(default=1, metadata={"help": "batch size during inference"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    restore_from: str = field(
        default="",
        metadata={
            "help": "The checkpoint that should be restored from for fine-tuning"
        },
    )
    per_device_train_batch_size: int = field(
        default=1,
    )
    per_device_eval_batch_size: int = field(
        default=1,
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )
    icot_train_path: str = field(default="/users/k24020023/efficient_cot/icae/code/coconut/icot_gsm8k/train.txt", metadata={"help":"The training data path"})
    num_latent: int = field(default=5, metadata={"help": "The number of latent for training or inference."})
    use_lora: bool = field(default=True, metadata={"help": "Use lora or not."})
    greedy: bool = field(default=False, metadata={"help": "Greedy decoding during inference."})
    exp_mode: bool = field(default=False, metadata={"help": "Use partial number of data. for debugging."})
    exp_data_num: int = field(default=10000, metadata={"help": "The number of data used in exp mode"})
    use_prj: bool = field(default=False, metadata={"help": "Use a prj module after the llm for latent generation."})
    prj_dim: int = field(default=2048, metadata={"help": "The hidden dim of the projection module."})
    prj_dropout: float = field(default=0.0, metadata={"help": "Dropout ratio of the projection module."})
    prj_no_ln: bool = field(default=False, metadata={"help": "Remove the Layer Norm layer for the projection module."})
    distill_loss_div_std: bool = field(default=False, metadata={"help": "Divide the distillation loss by a std for normallisation."})
    distill_loss_type: str = field(default="smooth_l1", metadata={"help": "Specify the distillation loss. Use smoothL1 by default."})
    distill_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the distillation loss; set to 0 to skip it entirely."})
    ref_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the distillation loss."})
    inf_latent_iterations: int = field(default=1, metadata={"help": ""})
    inf_num_iterations: int = field(default=5, metadata={"help": "Run multiple times during inference"})
    remove_eos: bool = field(default=False, metadata={"help": "Do not add <eos> as a delimiter to split QA."})
    print_ref_model_stats: bool = field(default=False, metadata={"help": "Print some stats for the teacher task."})
    include_last_cot: bool = field(default=False, metadata={"help": "Include the last CoT step in the training data."})
    fix_attn_mask: bool = field(default=False, metadata={"help": "Correct a bug about attention mask."})
    log_full: bool = field(default=False, metadata={"help": "Log all losses."})
    print_loss: bool = field(default=True)
    max_token_num: int = field(default=1000, metadata={"help": "Limit the longest data to avoid OOM."})

    # ===== CoT 分段对比对齐参数 =====
    # 参数名保留 cross_attn 前缀以兼容现有训练/测试脚本。
    use_cross_attn_align: bool = field(default=False, metadata={"help": "Use fixed-segment CoT contrastive alignment."})
    cross_attn_rank: int = field(default=64, metadata={"help": "Legacy unused argument retained for CLI compatibility."})
    cross_attn_heads: int = field(default=4, metadata={"help": "Legacy unused argument retained for CLI compatibility."})
    cross_attn_layer_idx: int = field(default=-1, metadata={"help": "Which layer of student to align (use negative idx, e.g., -1 = 1rd last)."})
    align_loss_factor: float = field(default=0.3, metadata={"help": "Weight for fixed-segment contrastive alignment loss (lambda_a)."})
    align_loss_warmup_ratio: float = field(default=0.2, metadata={"help": "Warmup ratio for alignment loss (e.g., 0.1 = first 10% steps)."})
    align_loss_decay_start: float = field(default=0.8, metadata={"help": "Start decay at this ratio of total steps."})
    align_temperature: float = field(default=0.5, metadata={"help": "Temperature for contrastive InfoNCE alignment loss (higher = smoother, more numerically stable)."})
    align_warmup_steps: int = field(default=0, metadata={"help": "Phase 1: for the first N steps, detach student hiddens so align loss only trains projection heads (does NOT touch the backbone). 0 = disabled."})
    align_freeze_aligner_after_warmup: bool = field(default=False, metadata={"help": "Phase 2: after warmup, freeze alignment projection heads so align loss only updates the backbone."})

def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(
        f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}"
    )
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


class LearnableContrastiveAligner(nn.Module):
    """
    固定分段的对比对齐模块：
        - 将有效 teacher CoT 从前到后等分为 ``num_latent`` 个连续、不重叠的段
        - 对每段 teacher hidden states 做 mean pooling，作为对应 student latent 的固定 target
        - 使用对角 InfoNCE 强制 latent_i 对齐 CoT segment_i

    类名保留为 LearnableContrastiveAligner 以兼容现有调用方；可学习部分
    是 student/teacher projection heads，不再包含可学习 attention。
    """
    def __init__(self, dim: int, num_latent: int, temperature: float = 0.5):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"align temperature must be positive, got {temperature}")
        if num_latent < 2:
            raise ValueError(
                "fixed-segment contrastive alignment requires num_latent >= 2; "
                f"got {num_latent}"
            )

        self.dim = dim
        self.num_latent = num_latent

        # 固定温度参数（防止数值不稳定）
        self.temperature = temperature

        # 投影层：把student和teacher投影到同一个对比空间
        self.student_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.teacher_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    @staticmethod
    def _build_segment_weights(
        teacher_cot_mask: torch.Tensor,
        num_latent: int,
        dtype: torch.dtype,
    ):
        """构造固定的等分段 mean-pooling 权重。

        对有效长度 L，第 i 段的边界为::

            start = floor(i * L / num_latent)
            end   = floor((i + 1) * L / num_latent)

        mask 不必是严格右 padding：分段使用每个有效 token 在 mask 中的顺序。
        """
        if teacher_cot_mask.ndim != 2:
            raise ValueError(
                "teacher_cot_mask must have shape [batch, cot_len], "
                f"got {tuple(teacher_cot_mask.shape)}"
            )

        teacher_cot_mask = teacher_cot_mask.bool()
        batch_size, cot_len = teacher_cot_mask.shape
        device = teacher_cot_mask.device

        valid_len = teacher_cot_mask.sum(dim=-1)  # [batch]
        latent_idx = torch.arange(num_latent, device=device)
        segment_start = (valid_len[:, None] * latent_idx[None, :]) // num_latent
        segment_end = (valid_len[:, None] * (latent_idx[None, :] + 1)) // num_latent
        segment_len = segment_end - segment_start  # [batch, num_latent]

        # valid_rank 是 token 在各自样本有效 CoT 中的 0-based 序号。
        valid_rank = teacher_cot_mask.long().cumsum(dim=-1) - 1
        in_segment = (
            teacher_cot_mask[:, None, :]
            & (valid_rank[:, None, :] >= segment_start[:, :, None])
            & (valid_rank[:, None, :] < segment_end[:, :, None])
        )

        segment_weights = in_segment.to(dtype=dtype)
        segment_weights = segment_weights / segment_len.clamp_min(1).to(dtype)[:, :, None]

        # L < num_latent 时无法严格分成 num_latent 个非空且不重叠的段。
        # 这类样本在 loss 中整体跳过，避免重复 token 或伪造 target。
        valid_sample = valid_len >= num_latent
        return segment_weights, segment_len, valid_sample

    def forward(
        self,
        student_latents: torch.Tensor,
        teacher_cot_hiddens: torch.Tensor,
        teacher_cot_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            student_latents: [batch, num_latent, dim] - Student latent hiddens
            teacher_cot_hiddens: [batch, cot_len, dim] - Teacher CoT hiddens
            teacher_cot_mask: [batch, cot_len] - True = valid position

        Returns:
            loss: 对比损失
            segment_weights: 固定分段的 mean-pooling 权重
                [batch, num_latent, cot_len]
        """
        if student_latents.ndim != 3 or teacher_cot_hiddens.ndim != 3:
            raise ValueError(
                "student_latents and teacher_cot_hiddens must be rank-3 tensors; "
                f"got {tuple(student_latents.shape)} and {tuple(teacher_cot_hiddens.shape)}"
            )

        batch_size = student_latents.shape[0]
        num_latent = student_latents.shape[1]
        if batch_size != teacher_cot_hiddens.shape[0]:
            raise ValueError(
                "student/teacher batch sizes must match; "
                f"got {batch_size} and {teacher_cot_hiddens.shape[0]}"
            )
        if num_latent != self.num_latent:
            raise ValueError(
                f"expected {self.num_latent} student latents, got {num_latent}"
            )
        if student_latents.shape[-1] != self.dim or teacher_cot_hiddens.shape[-1] != self.dim:
            raise ValueError(
                f"student/teacher hidden size must be {self.dim}; got "
                f"{student_latents.shape[-1]} and {teacher_cot_hiddens.shape[-1]}"
            )

        cot_len = teacher_cot_hiddens.shape[1]
        if teacher_cot_mask is None:
            teacher_cot_mask = torch.ones(
                (batch_size, cot_len),
                dtype=torch.bool,
                device=teacher_cot_hiddens.device,
            )
        elif teacher_cot_mask.shape != (batch_size, cot_len):
            raise ValueError(
                "teacher_cot_mask must match [batch, cot_len]; "
                f"expected {(batch_size, cot_len)}, got {tuple(teacher_cot_mask.shape)}"
            )

        # ========== 1. 构造从前到后的固定 CoT 分段 ==========
        segment_weights, _, valid_sample = self._build_segment_weights(
            teacher_cot_mask=teacher_cot_mask,
            num_latent=num_latent,
            dtype=teacher_cot_hiddens.dtype,
        )
        # 分段均值在 fp32 中累加，避免长 CoT 在 bf16/fp16 下丢失精度；
        # 进入 teacher projection 前再转回模型 dtype。
        with torch.autocast(device_type=teacher_cot_hiddens.device.type, enabled=False):
            teacher_segments = torch.matmul(
                segment_weights.float(),
                teacher_cot_hiddens.float(),
            )
        teacher_segments = teacher_segments.to(teacher_cot_hiddens.dtype)
        # [batch, num_latent, dim]

        # ========== 2. 投影到对比空间 ==========
        student_proj = self.student_proj(student_latents)  # [batch, num_latent, dim]
        teacher_proj = self.teacher_proj(teacher_segments)  # [batch, num_latent, dim]

        # ========== 3. 计算固定对角 InfoNCE ==========
        with torch.autocast(device_type=student_proj.device.type, enabled=False):
            student_normalized = F.normalize(student_proj.float(), dim=-1, eps=1e-6)
            teacher_normalized = F.normalize(teacher_proj.float(), dim=-1, eps=1e-6)
            similarity = torch.matmul(
                student_normalized,
                teacher_normalized.transpose(-1, -2),
            )
            logits = similarity / self.temperature
        # [batch, num_latent, num_latent]

        # 仅保留能严格分成 num_latent 个非空段的样本。
        logits = logits[valid_sample]

        if logits.shape[0] == 0:
            # 整个 batch 都无法严格分段时返回 0，同时让投影层留在梯度图中，
            # 避免 DDP 把它们误判为 unused parameters。
            loss = (student_proj.sum() + teacher_proj.sum()) * 0.0
        else:
            targets = torch.arange(num_latent, dtype=torch.long, device=logits.device)
            targets = targets.unsqueeze(0).expand(logits.shape[0], -1).reshape(-1)
            loss = F.cross_entropy(logits.reshape(-1, num_latent), targets)

        return loss, segment_weights


def get_align_loss_weight(
    step_ratio: float,
    warmup_ratio: float,
    decay_start: float,
    max_weight: float
) -> float:
    """
    计算对齐损失的权重调度: warmup -> peak -> decay to 0
    完全基于训练进度比例 (0 ~ 1)

    Args:
        step_ratio: 当前训练进度 (0 ~ 1)
        warmup_ratio: warmup 结束的比例 (e.g., 0.1 = first 10% steps)
        decay_start: 开始 decay 的训练进度比例 (0 ~ 1)
        max_weight: 最大权重

    Returns:
        weight: 当前步的权重
    """
    if not 0.0 <= warmup_ratio <= decay_start <= 1.0:
        raise ValueError(
            "align schedule must satisfy 0 <= warmup_ratio <= decay_start <= 1, "
            f"got warmup_ratio={warmup_ratio}, decay_start={decay_start}"
        )
    if max_weight < 0.0:
        raise ValueError(f"align max_weight must be non-negative, got {max_weight}")

    # 自定义 total-step 估算、恢复训练或 max_steps 配置可能让比例略微越界。
    # clamp 可避免 decay 权重变负；decay_start=1 表示不做末段衰减。
    step_ratio = min(max(float(step_ratio), 0.0), 1.0)

    if warmup_ratio > 0 and step_ratio < warmup_ratio:
        # Warmup: 从 0 线性增加
        return max_weight * (step_ratio / warmup_ratio)
    elif step_ratio < decay_start or decay_start == 1.0:
        # 保持峰值 (warmup_ratio=0 时从第 0 步即为峰值, ease-in 交给两阶段机制负责)
        return max_weight
    else:
        # Decay to 0
        decay_progress = (step_ratio - decay_start) / (1.0 - decay_start)
        return max_weight * (1.0 - decay_progress)


class CODI(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        model_wrapper_class = AutoModelForCausalLM
        if model_args.full_precision:
            self.codi = model_wrapper_class.from_pretrained(
                    self.model_name,
                    torch_dtype=(
                        torch.float16 if training_args.bf16 is False else torch.bfloat16
                    ),
                    resume_download=True,
                )
        else:
            self.codi = model_wrapper_class.from_pretrained(
                    self.model_name,
                    torch_dtype=(
                        torch.float16 if training_args.bf16 is False else torch.bfloat16
                    ),
                    resume_download=True,
                    quantization_config=transformers.BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=False,
                        bnb_4bit_quant_type='nf4',
                    )
                )


        ori_vocab_size = self.codi.config.vocab_size
        self.training = self.model_args.train

        # special tokens to enclose the latent embeddings
        self.pad_token_id = ori_vocab_size
        self.bot_id = ori_vocab_size + 1
        self.eot_id = ori_vocab_size + 2

        self.codi.resize_token_embeddings(
            ori_vocab_size + 3
        )  # dummy values for mem tokens

        self.dim = self.codi.config.hidden_size
        self.num_latent = training_args.num_latent
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        # LoRA
        if training_args.use_lora:
            self.codi = get_peft_model(self.codi, lora_config)

        # Projection Layer
        self.use_prj = training_args.use_prj
        self.prj_no_ln = training_args.prj_no_ln
        if training_args.use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(training_args.prj_dropout),
                nn.Linear(self.dim, training_args.prj_dim),
                nn.GELU(),
                nn.Linear(training_args.prj_dim, self.dim),
            )
            if not self.prj_no_ln:
                self.prj.add_module("ln", nn.LayerNorm(self.dim))
            # Convert to same dtype as main model
            self.prj.to(dtype=(torch.float16 if training_args.bf16 is False else torch.bfloat16))

        # ===== 固定 CoT 分段的对比对齐模块 =====
        self.use_cross_attn_align = training_args.use_cross_attn_align
        if self.use_cross_attn_align:
            self.contrastive_aligner = LearnableContrastiveAligner(
                dim=self.dim,
                num_latent=self.num_latent,
                temperature=training_args.align_temperature
            )
            self.cross_attn_layer_idx = training_args.cross_attn_layer_idx
            self.align_loss_factor = training_args.align_loss_factor
            self.align_loss_warmup_ratio = training_args.align_loss_warmup_ratio
            self.align_loss_decay_start = training_args.align_loss_decay_start
            # 两阶段对齐调度参数
            self.align_warmup_steps = training_args.align_warmup_steps
            self.align_freeze_aligner_after_warmup = training_args.align_freeze_aligner_after_warmup
            # Convert to same dtype as main model
            self.contrastive_aligner.to(dtype=(torch.float16 if training_args.bf16 is False else torch.bfloat16))

        # Losses
        self.print_loss = training_args.print_loss
        self.ref_loss_factor = training_args.ref_loss_factor

        # Cross Entropy Loss
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        # Distillation Loss
        self.distill_loss_div_std = training_args.distill_loss_div_std
        self.distill_loss_type = training_args.distill_loss_type
        self.distill_loss_factor = training_args.distill_loss_factor
        if self.distill_loss_type == "smooth_l1":
            self.distill_loss_fct = nn.SmoothL1Loss()
        elif self.distill_loss_type == "l2":
            self.distill_loss_fct = nn.MSELoss()
        else:
            raise NotImplementedError

        # general
        self.fix_attn_mask = training_args.fix_attn_mask

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token_id = self.pad_token_id

        if self.training:
            self.init()

    def get_embd(self, model, model_name):
        try:
            if "pythia" in model_name:
                return model.get_base_model().gpt_neox.embed_in
            elif "gpt2" in model_name:
                try:
                    return model.get_base_model().transformer.wte
                except Exception: # no lora
                    return model.transformer.wte
            else:
                try:
                    return model.get_base_model().model.embed_tokens
                except Exception: # no lora
                    return model.model.embed_tokens
        except AttributeError:
            if "pythia" in model_name:
                return model.gpt_neox.embed_in
            raise NotImplementedError

    def init(self):
        print_trainable_parameters(self)
        if (
            self.training_args.restore_from is not None
            and self.training_args.restore_from != ""
        ):
            print(
                f"Loading from the pretrained checkpoint: {self.training_args.restore_from}..."
            )
            state_dict = dict(load_file(self.training_args.restore_from))
            # 旧版 aligner 包含可学习 attention query/key。固定分段版不再
            # 使用这些参数；精确移除旧 key，其余权重仍保持 strict load。
            legacy_attention_keys = [
                key for key in state_dict
                if ".contrastive_aligner.attn_query." in key
                or ".contrastive_aligner.attn_key." in key
                or key.startswith("contrastive_aligner.attn_query.")
                or key.startswith("contrastive_aligner.attn_key.")
            ]
            for key in legacy_attention_keys:
                state_dict.pop(key)
            if legacy_attention_keys:
                print(
                    "Ignoring legacy learnable-attention parameters: "
                    f"{len(legacy_attention_keys)} tensors."
                )
            self.load_state_dict(state_dict)
            print(f"Finished loading from {self.training_args.restore_from}")

    def forward(
        self,
        encoder_input_ids: torch.LongTensor = None,
        decoder_input_ids: torch.LongTensor = None,
        ref_input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        ref_answer_position: Optional[torch.LongTensor] = None,
        model_answer_position: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.LongTensor] = None,
        ref_labels: torch.LongTensor = None,
        ref_cot_start: Optional[torch.LongTensor] = None,
        ref_cot_end: Optional[torch.LongTensor] = None,
        step: int = None,
        step_ratio: float = None
    ):
        if not self.fix_attn_mask:
            ref_attention_mask = None

        # Encode the question
        past_key_values = None
        outputs = self.codi(input_ids=encoder_input_ids, use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=encoder_attention_mask)
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1) # as the next input
        if self.use_prj:
            latent_embd = self.prj(latent_embd)

        len_pred_loss = 0
        dynamic_mask = None
        if self.fix_attn_mask:
            dynamic_mask = torch.ones((encoder_attention_mask.size(0), self.num_latent), device=ref_labels.device)

        # Iterate over the latent embeddings
        distill_loss_total = 0
        ce_loss_total = 0
        align_loss_total = 0
        align_loss_raw = 0
        align_weight = 0.0

        # ===== Teacher 前向传播 (获取 CoT 隐状态) =====
        with torch.no_grad():
            ref_outputs = self.codi(input_ids=ref_input_ids, output_hidden_states=True, attention_mask=ref_attention_mask)
        ref_outputs_with_grad = self.codi(input_ids=ref_input_ids, output_hidden_states=True, attention_mask=ref_attention_mask)

        # Formatting for deprecated exps
        ref_outputs_list = [ref_outputs]
        ref_input_ids_list = [ref_input_ids]

        # Process the position tensor
        # Normalise the position definition
        if "llama" in self.model_name.lower() or "qwen" in self.model_name.lower(): # there is one more token standing for " "
            if model_answer_position is not None:
                model_answer_position = model_answer_position + 1
            if ref_answer_position is not None:
                ref_answer_position = ref_answer_position + 1

        # For DEBUG: Print the probability of the teacher task to predict the correct answer
        if self.training_args.print_ref_model_stats:
            for i, (ref_inputs, ref_outputs) in enumerate(zip(ref_input_ids_list, ref_outputs_list)):
                # evalutae the reference model
                if len(ref_outputs_list) > 1:
                    pos = ref_answer_position[i]
                else:
                    pos = ref_answer_position
                ref_probs = torch.nn.functional.softmax(ref_outputs.logits, dim=-1)
                input_positions = (pos-1).unsqueeze(1).unsqueeze(1).expand(-1, -1, ref_probs.size(2))
                ref_probs_at_positions = ref_probs.gather(1, input_positions)
                probe_positions_positions = pos.unsqueeze(1)
                probe_positions = ref_inputs.gather(1, probe_positions_positions).unsqueeze(1)
                ref_probs_of_target = ref_probs_at_positions.gather(2, probe_positions)
                print(f'stage{i}: mean of the prob of the target token: {ref_probs_of_target.mean()}')

        # the model answer position is the position of the eot token to predict the first token of the response
        if model_answer_position is not None:
            model_answer_position = model_answer_position - 1
        if ref_answer_position is not None:
            ref_answer_position = ref_answer_position - 1

        # ===== 收集 Student 隐层状态用于对齐 =====
        student_latent_hiddens = []  # h^S_{z_i}: 指定层的隐状态

        num_latent = self.num_latent
        if self.num_latent != 0:
            for i in range(num_latent):
                # Implicit CoT generation
                outputs = self.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values

                # 保存指定层的隐状态 (用于对齐)
                if self.use_cross_attn_align:
                    layer_hidden = outputs.hidden_states[self.cross_attn_layer_idx]  # [batch, 1, dim]
                    student_latent_hiddens.append(layer_hidden.squeeze(1))  # [batch, dim]

                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if self.use_prj:
                    latent_embd = self.prj(latent_embd)

                # Calculate the distillation loss
                if i == num_latent - 1: # the last latent embedding
                    # Decode the final answer in natural language
                    embds = self.get_embd(self.codi, self.model_name)(decoder_input_ids)

                    if dynamic_mask is not None: # Prevent attending the paddings
                        decoder_mask = torch.ones((embds.size(0), embds.size(1)), dtype=torch.bool).to(dynamic_mask)
                        dynamic_mask = torch.cat((encoder_attention_mask, dynamic_mask, decoder_mask), dim=1)
                        dynamic_mask = dynamic_mask.bool()
                    # Student task's output
                    outputs = self.codi(inputs_embeds=embds, use_cache=True, output_hidden_states=True, past_key_values=past_key_values, attention_mask=dynamic_mask)
                    # Teacher task's output
                    ref_outputs = ref_outputs_list[0]

                    # factor=0 表示真正禁用 distillation。不能只在总 loss 中乘 0，
                    # 因为一旦这里产生 NaN，NaN * 0 仍会污染总 loss。
                    if self.distill_loss_factor != 0.0:
                        distill_loss = 0
                        # Calculate distillation loss between the teacher's and
                        # student's answer-position hidden states for every layer.
                        for out, ref_out in zip(outputs.hidden_states, ref_outputs.hidden_states):
                            ref_selected = ref_out.gather(1, ref_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ref_out.size(-1)))
                            out_selected = out.gather(1, model_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out.size(-1)))

                            distill_loss_tmp = self.distill_loss_fct(out_selected, ref_selected.detach())

                            if self.distill_loss_div_std:
                                # 在 fp32 中计算并 clamp，避免很小的 teacher std
                                # 在 bf16/fp16 训练中放大成 inf/NaN。
                                ref_std = ref_selected.float().std().clamp_min(1e-6)
                                distill_loss_tmp = distill_loss_tmp / ref_std.to(distill_loss_tmp.dtype)
                            distill_loss += distill_loss_tmp

                        distill_loss /= len(outputs.hidden_states)

                        if self.print_loss:
                            print(f'latent{i}: distill_loss={distill_loss}')

                        distill_loss_total += distill_loss

                    # Calculate the CE loss for the student task
                    if i == num_latent - 1:
                        logits = outputs.logits
                        effective_logits = logits[:, :-1, :]
                        effective_logits = effective_logits.reshape(-1, logits.size(-1))
                        target_ids = labels[:, 1:].reshape(-1)
                        ce_loss = self.loss_fct(effective_logits, target_ids)
                        ce_loss_total += ce_loss

        # ===== 固定 CoT 分段的对比对齐损失 =====
        if self.use_cross_attn_align and ref_cot_start is not None and len(student_latent_hiddens) > 0:
            # ===== 两阶段对齐调度 =====
            # Phase 1 (step < align_warmup_steps): detach 主干，align loss 只训练
            #          student/teacher projection heads，不扰动主干。
            # Phase 2 (step >= align_warmup_steps): 接通主干（不 detach），让 align loss 塑形主干；
            #          可选地冻结 projection heads，使对比压力只传导到主干。
            in_warmup = (
                self.align_warmup_steps > 0
                and step is not None
                and step < self.align_warmup_steps
            )
            detach_backbone = in_warmup
            freeze_aligner = (
                (not in_warmup)
                and self.align_freeze_aligner_after_warmup
            )
            # 冻结/解冻 aligner 参数（幂等，可安全每步调用）
            for p in self.contrastive_aligner.parameters():
                p.requires_grad = not freeze_aligner

            # 1. 准备 Student 侧：使用经过transformer后的hidden states
            student_hiddens = torch.stack(student_latent_hiddens, dim=1)  # [batch, num_latent, dim]
            if detach_backbone:
                # Phase 1: 切断到主干的梯度，align loss 的梯度只会落到 aligner 上
                student_hiddens = student_hiddens.detach()

            # 2. 准备 Teacher CoT 侧: 提取 CoT 区间的隐状态
            batch_size = ref_input_ids.shape[0]
            teacher_cot_hiddens = []
            teacher_cot_mask = []

            # 使用teacher的hidden states（用与student相同的层，或者最后一层）
            ref_hidden = ref_outputs.hidden_states[self.cross_attn_layer_idx]  # [batch, seq_len, dim]

            for b in range(batch_size):
                cot_start = ref_cot_start[b].item()
                cot_end = ref_cot_end[b].item() if ref_cot_end is not None else ref_answer_position[b].item()

                # 提取 CoT 区间的隐状态
                cot_hidden = ref_hidden[b, cot_start:cot_end, :]  # [cot_len, dim]
                cot_len = cot_hidden.shape[0]

                teacher_cot_hiddens.append(cot_hidden)
                teacher_cot_mask.append(torch.ones(cot_len, dtype=torch.bool, device=cot_hidden.device))

            # Padding 到相同长度
            max_cot_len = max([h.shape[0] for h in teacher_cot_hiddens])
            padded_teacher_hiddens = []
            padded_teacher_mask = []

            for h, m in zip(teacher_cot_hiddens, teacher_cot_mask):
                pad_len = max_cot_len - h.shape[0]
                # new_zeros 同时继承 device 和 dtype；默认 torch.zeros(float32)
                # 会把 bf16/fp16 teacher hidden 静默提升到 fp32。
                padded_h = torch.cat([h, h.new_zeros((pad_len, h.shape[1]))], dim=0)
                padded_m = torch.cat([m, torch.zeros(pad_len, dtype=torch.bool, device=m.device)], dim=0)
                padded_teacher_hiddens.append(padded_h)
                padded_teacher_mask.append(padded_m)

            teacher_cot_hiddens = torch.stack(padded_teacher_hiddens, dim=0)  # [batch, max_cot_len, dim]
            teacher_cot_mask = torch.stack(padded_teacher_mask, dim=0)  # [batch, max_cot_len]

            # 3. 固定分段的对比对齐
            align_loss, segment_weights = self.contrastive_aligner(
                student_latents=student_hiddens,
                teacher_cot_hiddens=teacher_cot_hiddens.detach(),
                teacher_cot_mask=teacher_cot_mask
            )
            align_loss_raw = align_loss

            # ===== 诊断: 每 500 步打印第一个样本的确定性分段 =====
            if step is not None and step % 500 == 0:
                with torch.no_grad():
                    sw = segment_weights[0].float()  # [num_latent, cot_len]
                    valid_len = int(teacher_cot_mask[0].sum().item())
                    sample0_used = valid_len >= self.num_latent
                    skipped_samples = int(
                        (teacher_cot_mask.sum(dim=-1) < self.num_latent).sum().item()
                    )

                    print(
                        f"[SEGMENT_ALIGN] step={step}, valid_cot_len={valid_len}, "
                        f"sample0_used={sample0_used}, "
                        f"skipped_short_cot={skipped_samples}/{batch_size}"
                    )
                    if not sample0_used:
                        print(
                            "[SEGMENT_ALIGN]   sample0 excluded from alignment loss: "
                            "valid_cot_len < num_latent"
                        )
                    else:
                        for li in range(sw.shape[0]):
                            positions = torch.nonzero(sw[li] > 0, as_tuple=False).flatten()
                            start = int(positions[0].item())
                            end = int(positions[-1].item()) + 1
                            token_weight = float(sw[li, positions[0]].item())
                            print(
                                f"[SEGMENT_ALIGN]   latent{li}: cot_pos=[{start}, {end}), "
                                f"segment_len={positions.numel()}, token_weight={token_weight:.6f}"
                            )

            # 4. 权重调度
            if step_ratio is not None:
                align_weight = get_align_loss_weight(
                    step_ratio=step_ratio,
                    warmup_ratio=self.align_loss_warmup_ratio,
                    decay_start=self.align_loss_decay_start,
                    max_weight=self.align_loss_factor
                )
            else:
                align_weight = self.align_loss_factor
            phase = "P1(detach-backbone,train-projections)" if detach_backbone else (
                "P2(train-backbone,freeze-projections)" if freeze_aligner else "P2(train-backbone+projections)"
            )
            print(f"step={step}, align_weight={align_weight}, align_phase={phase}")

            align_loss_total = align_loss * align_weight

        # Calculate the CE loss for the teacher task
        ref_ce_loss = 0
        ref_logits = ref_outputs_with_grad.logits
        effective_ref_logits = ref_logits[:, :-1, :]
        effective_ref_logits = effective_ref_logits.reshape(-1, ref_logits.size(-1))
        ref_target_ids = ref_labels[:, 1:].reshape(-1)
        ref_ce_loss = self.loss_fct(effective_ref_logits, ref_target_ids)
        ref_ce_loss *= self.ref_loss_factor

        # DataParallel/DDP 的输出聚合要求各 loss 指标保持 tensor。禁用某项
        # loss 时也返回同 device/dtype 的 0-d tensor，而不是 Python 0/0.0。
        if not torch.is_tensor(ce_loss_total):
            ce_loss_total = ref_ce_loss.new_zeros(())
        if not torch.is_tensor(distill_loss_total):
            distill_loss_total = ref_ce_loss.new_zeros(())
        if not torch.is_tensor(align_loss_total):
            align_loss_total = ref_ce_loss.new_zeros(())
        if not torch.is_tensor(align_loss_raw):
            align_loss_raw = ref_ce_loss.new_zeros(())

        # Weigh the distillation loss
        distill_loss_total_scaled = distill_loss_total * self.distill_loss_factor

        # Total loss
        loss = ce_loss_total + distill_loss_total_scaled + ref_ce_loss
        if self.use_cross_attn_align:
            loss = loss + align_loss_total

        if self.print_loss:
            loss_str = f'loss={loss.item()}, ce_loss={ce_loss_total if isinstance(ce_loss_total, float) else ce_loss_total.item()}, distill_loss={distill_loss_total if isinstance(distill_loss_total, float) else distill_loss_total.item()}, ref_ce_loss={ref_ce_loss if isinstance(ref_ce_loss, float) else ref_ce_loss.item()}'
            if self.use_cross_attn_align:
                loss_str += f', align_loss={align_loss_total if isinstance(align_loss_total, float) else align_loss_total.item()}, align_weight={align_weight}'
            print(loss_str)

        # Keep as tensors for DataParallel gather (detach but don't convert to Python float)
        ce_loss_total = ce_loss_total.detach()
        distill_loss_total = distill_loss_total.detach()
        ref_ce_loss = ref_ce_loss.detach()

        return_dict = {
            "loss": loss,
            "logits": logits,
            "ce_loss": ce_loss_total,
            "distill_loss": distill_loss_total,
            "distill_loss_scaled": distill_loss_total_scaled.detach() if torch.is_tensor(distill_loss_total_scaled) else distill_loss_total_scaled,
            "ref_ce_loss": ref_ce_loss,
        }

        if self.use_cross_attn_align:
            return_dict["align_loss"] = align_loss_total.detach()
            return_dict["align_loss_raw"] = align_loss_raw.detach()
            # Convert align_weight to a tensor for DataParallel
            return_dict["align_weight"] = torch.tensor(align_weight, device=loss.device, dtype=loss.dtype)

        return return_dict
