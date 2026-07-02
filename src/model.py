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
    distill_loss_factor: float = field(default=1.0, metadata={"help": "A multiplier of the distillation loss."})
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
    cross_attn_layer_idx: int = field(default=-3, metadata={"help": "Which layer of student to align (use negative idx, e.g., -3 = 3rd last)."})
    align_loss_factor: float = field(default=0.3, metadata={"help": "Weight for cross-attention soft alignment loss (lambda_a)."})
    align_loss_warmup_ratio: float = field(default=0.2, metadata={"help": "Warmup ratio for alignment loss (e.g., 0.1 = first 10% steps)."})
    align_loss_decay_start: float = field(default=0.8, metadata={"help": "Start decay at this ratio of total steps."})

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
    def __init__(self, dim: int, num_latent: int, temperature: float = 0.1):
        super().__init__()
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

        # 计算attention分数
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.dim ** 0.5)
        # [batch, num_latent, cot_len]

        # mask padding
        if teacher_cot_mask is not None:
            mask = teacher_cot_mask.unsqueeze(1)  # [batch, 1, cot_len]
            attn_scores = attn_scores.masked_fill(~mask, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch, num_latent, cot_len]

        # ========== 3. 得到对齐后的teacher表示 ==========
        aligned_teacher = torch.matmul(attn_weights, teacher_proj)
        # [batch, num_latent, dim]

        # ========== 4. 计算对比损失 ==========
        # 归一化特征
        student_normalized = F.normalize(student_proj, dim=-1)
        aligned_normalized = F.normalize(aligned_teacher, dim=-1)

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
        teacher_normalized = F.normalize(teacher_proj, dim=-1)
        teacher_sim = torch.matmul(student_normalized, teacher_normalized.transpose(-1, -2))
        # [batch, num_latent, cot_len]

        # 采样一些低attention的位置作为负例
        num_neg_teacher = min(4, teacher_sim.shape[-1] - 1)
        if num_neg_teacher > 0:
            _, neg_indices = torch.topk(-attn_weights, k=num_neg_teacher, dim=-1)
            neg_sims_teacher = teacher_sim.gather(-1, neg_indices)
            # 合并两类负例
            neg_sims = torch.cat([neg_sims_cross, neg_sims_teacher], dim=-1)
        else:
            neg_sims = neg_sims_cross

        # --- InfoNCE loss
        exp_pos = torch.exp(pos_sims / self.temperature)
        exp_neg = torch.exp(neg_sims / self.temperature).sum(dim=-1)

        loss = -torch.log(exp_pos / (exp_pos + exp_neg + 1e-9)).mean()

        # ========== 5. 稀疏性损失：让attention更集中 ==========
        # 计算attention的熵，鼓励更sharp的分布
        entropy = -(attn_weights * torch.log(attn_weights + 1e-9)).sum(dim=-1)
        sparsity_loss = entropy.mean()

        total_loss = loss + 0.01 * sparsity_loss

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
    if step_ratio < warmup_ratio:
        # Warmup: 从 0 线性增加
        return max_weight * (step_ratio / warmup_ratio)
    elif step_ratio < decay_start:
        # 保持峰值
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
                temperature=0.1
            )
            self.cross_attn_layer_idx = training_args.cross_attn_layer_idx
            self.align_loss_factor = training_args.align_loss_factor
            self.align_loss_warmup_ratio = training_args.align_loss_warmup_ratio
            self.align_loss_decay_start = training_args.align_loss_decay_start
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

                    distill_loss = 0
                    # Calculate distillation loss between the teacher's logits and the student's logits for every layer
                    for j, (out, ref_out) in enumerate(zip(outputs.hidden_states, ref_outputs.hidden_states)):
                        ref_selected = ref_out.gather(1, ref_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ref_out.size(-1)))
                        out_selected = out.gather(1, model_answer_position.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out.size(-1)))

                        distill_loss_tmp = self.distill_loss_fct(out_selected, ref_selected.detach())

                        if self.distill_loss_div_std:
                            if self.distill_loss_type == 'l2':
                                distill_loss_tmp /= ref_selected.std()
                            distill_loss_tmp /= ref_selected.std()
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
            # 1. 准备 Student 侧：使用经过transformer后的hidden states
            student_hiddens = torch.stack(student_latent_hiddens, dim=1)  # [batch, num_latent, dim]

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
                padded_h = torch.cat([h, torch.zeros(pad_len, h.shape[1], device=h.device)], dim=0)
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
            print(f"step={step}, align_weight={align_weight}")

            align_loss_total = align_loss * align_weight

        # Calculate the CE loss for the teacher task
        ref_ce_loss = 0
        ref_logits = ref_outputs_with_grad.logits
        effective_ref_logits = ref_logits[:, :-1, :]
        effective_ref_logits = effective_ref_logits.reshape(-1, ref_logits.size(-1))
        ref_target_ids = ref_labels[:, 1:].reshape(-1)
        ref_ce_loss = self.loss_fct(effective_ref_logits, ref_target_ids)
        ref_ce_loss *= self.ref_loss_factor

        # Weigh the distillation loss
        distill_loss_total_scaled = distill_loss_total * self.distill_loss_factor

        # Total loss
        loss = ce_loss_total + distill_loss_total_scaled + ref_ce_loss
        if self.use_cross_attn_align and align_loss_total != 0:
            loss = loss + align_loss_total

        if self.print_loss:
            loss_str = f'loss={loss.item()}, ce_loss={ce_loss_total if isinstance(ce_loss_total, float) else ce_loss_total.item()}, distill_loss={distill_loss_total if isinstance(distill_loss_total, float) else distill_loss_total.item()}, ref_ce_loss={ref_ce_loss if isinstance(ref_ce_loss, float) else ref_ce_loss.item()}'
            if self.use_cross_attn_align:
                loss_str += f', align_loss={align_loss_total if isinstance(align_loss_total, float) else align_loss_total.item()}, align_weight={align_weight}'
            print(loss_str)

        # Keep as tensors for DataParallel gather (detach but don't convert to Python float)
        if ce_loss_total != 0 and not isinstance(ce_loss_total, float):
            ce_loss_total = ce_loss_total.detach()
        if distill_loss_total != 0 and not isinstance(distill_loss_total, float):
            distill_loss_total = distill_loss_total.detach()
        if ref_ce_loss != 0 and not isinstance(ref_ce_loss, float):
            ref_ce_loss = ref_ce_loss.detach()

        return_dict = {
            "loss": loss,
            "logits": logits,
            "ce_loss": ce_loss_total,
            "distill_loss": distill_loss_total,
            "ref_ce_loss": ref_ce_loss,
        }

        if self.use_cross_attn_align:
            if not isinstance(align_loss_total, float):
                return_dict["align_loss"] = align_loss_total.detach()
            else:
                return_dict["align_loss"] = align_loss_total
            # Convert align_weight to a tensor for DataParallel
            return_dict["align_weight"] = torch.tensor(align_weight, device=loss.device, dtype=loss.dtype)

        return return_dict
