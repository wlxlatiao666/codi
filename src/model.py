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
import math
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

    # ===== Cross-Attention 软对齐蒸馏新增参数 =====
    use_cross_attn_align: bool = field(default=False, metadata={"help": "Use cross-attention soft alignment distillation."})
    cross_attn_rank: int = field(default=64, metadata={"help": "Low-rank dimension for cross-attention Q/K/V projections."})
    cross_attn_heads: int = field(default=4, metadata={"help": "Number of attention heads for cross-attention."})
    cross_attn_layer_idx: int = field(default=-1, metadata={"help": "Which layer of student to align (use negative idx, e.g., -1 = 1rd last)."})
    align_loss_factor: float = field(default=0.3, metadata={"help": "Weight for cross-attention soft alignment loss (lambda_a)."})
    align_loss_warmup_ratio: float = field(default=0.2, metadata={"help": "Warmup ratio for alignment loss (e.g., 0.1 = first 10% steps)."})
    align_loss_decay_start: float = field(default=0.8, metadata={"help": "Start decay at this ratio of total steps."})
    align_temperature: float = field(default=0.5, metadata={"help": "Temperature for contrastive InfoNCE alignment loss (higher = smoother, more numerically stable)."})
    align_warmup_steps: int = field(default=0, metadata={"help": "Phase 1: for the first N steps, detach student hiddens so align loss only trains the aligner (does NOT touch the backbone). 0 = disabled (gradient always flows to backbone)."})
    align_freeze_aligner_after_warmup: bool = field(default=False, metadata={"help": "Phase 2: after warmup, freeze the aligner so align loss gradient flows only into the backbone (prevents the aligner from taking the attention-collapse shortcut)."})

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
    可学习的对比对齐模块：
        - 让student latent自动学习该attend to teacher CoT的哪些位置
        - 使用对比学习而非强制MSE
    """
    def __init__(self, dim: int, num_latent: int, temperature: float = 0.5, sparsity_factor: float = 0.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"align temperature must be positive, got {temperature}")
        if num_latent < 1:
            raise ValueError(f"contrastive alignment requires num_latent >= 1, got {num_latent}")

        self.dim = dim
        self.num_latent = num_latent

        # 固定温度参数（防止数值不稳定）
        self.temperature = temperature
        # attention 稀疏性正则系数（>0 时鼓励更 sharp 的注意力，缓解均匀塌缩）
        self.sparsity_factor = sparsity_factor

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

        # 注意力投影层（用于计算student对teacher的attention）
        self.attn_query = nn.Linear(dim, dim)
        self.attn_key = nn.Linear(dim, dim)

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
            attn_weights: attention权重 [batch, num_latent, cot_len]
        """
        batch_size = student_latents.shape[0]
        num_latent = student_latents.shape[1]

        # ========== 1. 投影到对比空间 ==========
        student_proj = self.student_proj(student_latents)  # [batch, num_latent, dim]
        teacher_proj = self.teacher_proj(teacher_cot_hiddens)  # [batch, cot_len, dim]

        # ========== 2. 计算attention权重 ==========
        # 用student作为query，teacher作为key
        q = self.attn_query(student_proj)  # [batch, num_latent, dim]
        k = self.attn_key(teacher_proj)  # [batch, cot_len, dim]

        # attention logits/softmax 使用 fp32，避免 bf16/fp16 下 q·k 或 softmax
        # 精度不足。输出权重在与 value 相乘前再转回模型 dtype。
        with torch.autocast(device_type=q.device.type, enabled=False):
            attn_scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / (self.dim ** 0.5)
        # [batch, num_latent, cot_len]

        # mask padding
        if teacher_cot_mask is not None:
            mask = teacher_cot_mask.unsqueeze(1)  # [batch, 1, cot_len]
            valid_rows = teacher_cot_mask.any(dim=-1).view(batch_size, 1, 1)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
            # softmax(all -inf) 会产生 NaN；空 CoT 行先置零，稍后其权重和
            # logits 会被显式清零/过滤。
            attn_scores = torch.where(valid_rows, attn_scores, torch.zeros_like(attn_scores))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch, num_latent, cot_len]
        if teacher_cot_mask is not None:
            attn_weights = attn_weights * mask

        # ========== 3. 得到对齐后的teacher表示 ==========
        aligned_teacher = torch.matmul(attn_weights.to(teacher_proj.dtype), teacher_proj)
        # [batch, num_latent, dim]

        # ========== 4. 计算对比损失 ==========
        # 归一化特征
        with torch.autocast(device_type=student_proj.device.type, enabled=False):
            student_normalized = F.normalize(student_proj.float(), dim=-1, eps=1e-6)
            aligned_normalized = F.normalize(aligned_teacher.float(), dim=-1, eps=1e-6)

            # --- 正例相似度：(student_i, aligned_teacher_i)
            pos_sims = (student_normalized * aligned_normalized).sum(dim=-1)
            # [batch, num_latent]

            # --- 负例相似度：
            # 负例1: student_i 和其他位置的aligned_teacher_j (j != i)
            # 计算所有pair的相似度
            all_sim = torch.matmul(student_normalized, aligned_normalized.transpose(-1, -2))
        # [batch, num_latent, num_latent]

        # 去掉对角线（正例）
        mask = ~torch.eye(num_latent, dtype=torch.bool, device=all_sim.device)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        neg_sims_cross = all_sim.masked_select(mask).view(batch_size, num_latent, num_latent - 1)

        # 负例2: student_i 和teacher的其他位置
        # 重新计算student对teacher各个位置的相似度
        with torch.autocast(device_type=teacher_proj.device.type, enabled=False):
            teacher_normalized = F.normalize(teacher_proj.float(), dim=-1, eps=1e-6)
            teacher_sim = torch.matmul(student_normalized, teacher_normalized.transpose(-1, -2))
        # [batch, num_latent, cot_len]

        # 采样一些低 attention 的有效 CoT 位置作为负例。
        # 不能直接对 -attn_weights 做 topk：padding 的 attention 恒为 0，
        # 会被优先选中，导致模型把伪造的 padding hidden 当成 teacher 负例。
        num_neg_teacher = min(5, teacher_sim.shape[-1] - 1)
        if num_neg_teacher > 0:
            if teacher_cot_mask is not None:
                valid_teacher_mask = teacher_cot_mask.unsqueeze(1).expand(-1, num_latent, -1)
                neg_rank_scores = attn_weights.masked_fill(
                    ~valid_teacher_mask,
                    torch.finfo(attn_weights.dtype).max,
                )
                _, neg_indices = torch.topk(
                    neg_rank_scores,
                    k=num_neg_teacher,
                    dim=-1,
                    largest=False,
                )
            else:
                _, neg_indices = torch.topk(-attn_weights, k=num_neg_teacher, dim=-1)

            neg_sims_teacher = teacher_sim.gather(-1, neg_indices)

            if teacher_cot_mask is not None:
                # 每个样本至少保留一个有效 teacher 位置作为非负例；当有效
                # CoT 长度不足时，用 -inf 屏蔽 topk 补齐出来的槽位。
                valid_neg_count = (teacher_cot_mask.sum(dim=-1) - 1).clamp(
                    min=0,
                    max=num_neg_teacher,
                )
                neg_rank = torch.arange(num_neg_teacher, device=teacher_sim.device)
                neg_valid_mask = neg_rank.view(1, 1, -1) < valid_neg_count.view(-1, 1, 1)
                neg_sims_teacher = neg_sims_teacher.masked_fill(~neg_valid_mask, float("-inf"))

            # 合并两类负例
            neg_sims = torch.cat([neg_sims_cross, neg_sims_teacher], dim=-1)
        else:
            neg_sims = neg_sims_cross

        # --- InfoNCE loss (log-sum-exp 稳定版) ---
        # 把正例放在 index 0，负例接在后面，构造 logits，目标标签恒为 0。
        # F.cross_entropy 内部使用 log_softmax(=log-sum-exp)，天然数值稳定，
        # 避免手写 exp()/log() 在低温或 bf16 下溢出为 inf/nan。
        logits = torch.cat([pos_sims.unsqueeze(-1), neg_sims], dim=-1) / self.temperature
        # [batch, num_latent, 1 + num_neg]
        logits = logits.reshape(-1, logits.shape[-1])  # [(batch*num_latent), 1+num_neg]

        # 保护：过滤掉 teacher CoT 全为 padding 的样本（valid_cot_len=0），
        # 它们的 aligned_teacher 来自全 mask 行的伪均匀分布，会污染对比损失。
        if teacher_cot_mask is not None:
            valid_rows = teacher_cot_mask.any(dim=-1)  # [batch] True=该样本有有效 CoT
            row_keep = valid_rows.unsqueeze(-1).expand(batch_size, num_latent).reshape(-1)
            logits = logits[row_keep]

        if logits.shape[0] == 0:
            # 整个 batch 都没有有效 CoT（极端情况），返回 0 损失并让全部
            # aligner 参数留在梯度图中，避免 DDP 把它们误判为 unused。
            loss = (student_proj.sum() + teacher_proj.sum() + q.sum() + k.sum()) * 0.0
        else:
            targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, targets)

        # ========== 5. 稀疏性损失：让attention更集中 ==========
        # 注意方向：sparsity_loss = entropy.mean()，最小化它会让 attention 更 sharp。
        # 当前问题是 attention 快速塌缩(entropy->0)，因此默认不启用(sparsity_factor=0)。
        # 若需缓解“均匀塌缩”可设正系数；若需缓解“单点塌缩”应改用负系数(奖励熵)。
        total_loss = loss
        if self.sparsity_factor != 0.0:
            entropy = -(attn_weights * torch.log(attn_weights + 1e-9)).sum(dim=-1)
            sparsity_loss = entropy.mean()
            total_loss = loss + self.sparsity_factor * sparsity_loss

        return total_loss, attn_weights


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

        # ===== Learnable Contrastive 对齐模块 =====
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
            state_dict = load_file(self.training_args.restore_from)
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
        student_latent_inputs = []  # z_i: latent 输入 embedding
        student_latent_hiddens = []  # h^S_{z_i}: 指定层的隐状态

        num_latent = self.num_latent
        if self.num_latent != 0:
            for i in range(num_latent):
                # Implicit CoT generation
                outputs = self.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values

                # 保存 latent 输入 embedding (用于 cross-attention query)
                student_latent_inputs.append(latent_embd.squeeze(1))  # [batch, dim]

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

        # ===== Learnable Contrastive 对齐损失 =====
        if self.use_cross_attn_align and ref_cot_start is not None and len(student_latent_hiddens) > 0:
            # ===== 两阶段对齐调度 =====
            # Phase 1 (step < align_warmup_steps): detach 主干，align loss 只训练 aligner，
            #          让 aligner 先学出稳定、非塌缩的对齐目标，且完全不扰动主干（不会崩）。
            # Phase 2 (step >= align_warmup_steps): 接通主干（不 detach），让 align loss 塑形主干；
            #          可选地冻结 aligner，堵死“塌缩 attention 走捷径”这条路，
            #          迫使对比压力真正传导到主干（推理态受益的关键）。
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

            # 3. Learnable Contrastive 对齐
            align_loss, attn_weights = self.contrastive_aligner(
                student_latents=student_hiddens,
                teacher_cot_hiddens=teacher_cot_hiddens.detach(),
                teacher_cot_mask=teacher_cot_mask
            )
            align_loss_raw = align_loss

            # ===== 诊断: 每 500 步打印 attention 权重的熵和 top-3 关注位置 =====
            # 用于判断 attention 是否退化 (模式A: 都关注同一位置 / 模式B: 均匀分布)
            if step is not None and step % 500 == 0:
                with torch.no_grad():
                    # 取 batch 中第一个样本的 attention: [num_latent, cot_len]
                    aw = attn_weights[0].float()
                    valid_len = int(teacher_cot_mask[0].sum().item())  # 该样本有效 CoT 长度

                    # 每个 latent 的注意力熵 (越大越接近均匀分布)
                    entropy = -(aw * torch.log(aw + 1e-9)).sum(dim=-1)  # [num_latent]
                    # 归一化熵 (除以 log(valid_len), 便于跨样本比较, 范围 ~[0,1])
                    norm_factor = math.log(valid_len) if valid_len > 1 else 1.0
                    norm_entropy = entropy / norm_factor

                    # 每个 latent 最关注的 top-3 CoT 位置及其权重
                    k = min(3, aw.shape[-1])
                    topk_vals, topk_idx = torch.topk(aw, k=k, dim=-1)  # [num_latent, k]

                    print(f"[ATTN] step={step}, valid_cot_len={valid_len}")
                    print(f"[ATTN]   entropy(per-latent)      = {[round(x, 3) for x in entropy.tolist()]}")
                    print(f"[ATTN]   norm_entropy(per-latent)  = {[round(x, 3) for x in norm_entropy.tolist()]}")
                    for li in range(aw.shape[0]):
                        idx_str = topk_idx[li].tolist()
                        val_str = [round(x, 3) for x in topk_vals[li].tolist()]
                        print(f"[ATTN]   latent{li}: top{k}_pos={idx_str}, top{k}_weight={val_str}")

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
            phase = "P1(detach-backbone,train-aligner)" if detach_backbone else (
                "P2(train-backbone,freeze-aligner)" if freeze_aligner else "P2(train-backbone+aligner)"
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
