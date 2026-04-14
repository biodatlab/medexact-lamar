from unsloth import FastModel
from datasets import load_dataset, Dataset
import re
import difflib
from collections import defaultdict, Counter
from trl import GRPOConfig, GRPOTrainer
import pandas as pd
from vllm import SamplingParams
import math
from torch.nn.parallel import DistributedDataParallel
DistributedDataParallel.config = property(lambda self: getattr(self.module, "config", None))

import argparse
from utils.rewards import *

parser = argparse.ArgumentParser(description="GRPO for Clinical Decision Extraction")
parser.add_argument("--model_name", type=str, default="Keetawan/Qwen3.5-4B_LoRA_Exact_BF16_Rank256_Alpha32_DFT")
parser.add_argument("--data_path", type=str, default="./dataset/train.csv")
parser.add_argument("--output_dir", type=str, default="/workspace/qwen3_5-4b_dft_lora_r2_alpha2_grpo_dapo_reward")
parser.add_argument("--lora_rank", type=int, default=2)
parser.add_argument("--lora_alpha", type=int, default=2)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--grad_accumulation", type=int, default=4)
parser.add_argument("--max_seq_length", type=int, default=24576)
args = parser.parse_args()


def apply_qwen35_rope_delta_patch() -> None:
    """
    Work around shape mismatches in Qwen3.5 multimodal RoPE deltas during mixed
    train/eval/generation flows where effective batch sizes can change.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
    except Exception as exc:
        print(f"Skipping Qwen3.5 RoPE patch import: {exc}", flush=True)
        return

    qwen_model_cls = getattr(modeling_qwen3_5, "Qwen3_5Model", None)
    if qwen_model_cls is None or not hasattr(qwen_model_cls, "compute_3d_position_ids"):
        return

    if getattr(qwen_model_cls, "_rope_delta_guard_patched", False):
        return

    original_compute = qwen_model_cls.compute_3d_position_ids

    def _patched_compute_3d_position_ids(
        self,
        input_ids,
        inputs_embeds,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        past_key_values=None,
    ):
        # If stale deltas cannot align to this batch, clear and let the model infer
        # from cache_position / attention mask to avoid zero-length delta tensors.
        if getattr(self, "rope_deltas", None) is not None and inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
            delta_rows = self.rope_deltas.shape[0] if self.rope_deltas.ndim > 0 else 0

            if delta_rows == 0:
                self.rope_deltas = None
            elif delta_rows != batch_size:
                if batch_size % delta_rows == 0:
                    pass
                elif delta_rows > batch_size:
                    self.rope_deltas = self.rope_deltas[:batch_size]
                else:
                    repeats = math.ceil(batch_size / delta_rows)
                    self.rope_deltas = self.rope_deltas.repeat_interleave(repeats, dim=0)[:batch_size]

        return original_compute(
            self,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

    qwen_model_cls.compute_3d_position_ids = _patched_compute_3d_position_ids
    qwen_model_cls._rope_delta_guard_patched = True
    print("Applied Qwen3.5 RoPE-delta guard patch.", flush=True)


apply_qwen35_rope_delta_patch()

max_seq_length = args.max_seq_length

model, tokenizer = FastModel.from_pretrained(
    model_name = args.model_name,
    max_seq_length = args.max_seq_length,
    load_in_4bit = False,     # MoE QLoRA not recommended, dense 27B is fine
    load_in_16bit = True,     # bf16/16-bit LoRA
    full_finetuning = False,
    gpu_memory_utilization=0.9,
    fast_inference=False,             # GRPO training doesn't need fast inference optimizations
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers
    use_gradient_checkpointing = "unsloth",
    r = args.lora_rank,
    lora_alpha = args.lora_alpha,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407
)



# Dataset Loading
train = pd.read_csv(args.data_path)


USER_PROMPT_TEMPLATE = """"You are an expert specializes in extracting clinical decisions from a patient's discharge summary.

### YOUR TASK ###
Given an input discharge summary, return the EXACT SAME text, but with specific phrases wrapped in inline tags to mark clinical decisions.

IMPORTANT:
- Do NOT add, remove, or rephrase any text outside the tags.
- Preserve all original punctuation, line breaks, and spacing.
- EVERY opening tag MUST have a corresponding closing tag (e.g., <drug_decision>Aspirin 81 mg daily</drug_decision>).
- These tags CAN overlap or nest in one another, as long as they are VALID TAGS.

### DECISION CATEGORIES & TAGS ###
Use the following tags exactly as defined:
1. <define_problem> : diagnostic conclusions, health state evaluations, etiological inference, or prognostic judgment.
2. <drug_decision> : decisions to start, stop, continue, withhold, or modify medications.
3. <evaluate_result> : interpretation of clinical findings or test results.
4. <contact_related> : admissions, discharges, follow-ups, or referrals to other hospitals.
5. <therapeutic_procedure> : decisions to perform, plan, or refrain from procedures.
6. <advice_and_precaution> : patient instructions, advice, or precautions.
7. <gather_info> : decisions to order tests and investigations or consult another colleague.
8. <treatment_goal> : therapeutic goals, aims, or treatment objectives.
9. <defer_decision> : delaying judgment or action for now.

### ANNOTATION RULES ###
1. Boundary: Annotate spans that capture the full clinical decision. Prefer longer spans than short words.
2. Comprehensiveness: The output should be comprehensively annotated. Extract as many valid decisions as possible.
3. Exclusions: DO NOT annotate document headers or labels (e.g., "Admission Date:", "Discharge Date:", "Physical Exam:").
4. Overlapping Spans: Spans may overlap or belong to multiple categories. Wrap each span independently with all applicable tags.
Example (nested): <drug_decision>continue warfarin for <treatment_goal>stroke prevention</treatment_goal></drug_decision>
Example (partial overlap): <define_problem>The next previous examination suggested <evaluate_result>atelectasis</define_problem> - density in the left base cannot be evaluated</evaluate_result>

### OUTPUT FORMAT ###
Return ONLY the fully annotated text. Ensure all tags are properly closed. Do not include any explanations.

### INPUT TEXT ###
{raw_text}"""

# Format
formatted_data = []
for index, row in train.iterrows():
    raw_text = str(row['raw_text'])
    gold_token = str(row['gold_span'])


    prompt_messages = [
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(raw_text=raw_text)}
    ]


    formatted_data.append({
        "prompt": prompt_messages,   # GRPO จะดึงคอลัมน์นี้ไปให้โมเดลสร้างคำตอบ
        "gold": gold_token,          # ส่งไปให้ Reward Function ตรวจความถูกต้อง
        "raw_text": raw_text         # ส่งไปให้ Reward Function ตรวจการเปลี่ยนคำ (Reconstruction)
    })


dataset = Dataset.from_list(formatted_data)


max_prompt_length = 10029
max_completion_length = max_seq_length - max_prompt_length

vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=42,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
        max_tokens=max_completion_length,
        temperature=1.0)


training_args = GRPOConfig(
    learning_rate = args.learning_rate,
    lr_scheduler_type = "constant",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = args.batch_size,
    gradient_accumulation_steps = args.grad_accumulation,
    num_generations = 8,
    max_completion_length = max_completion_length,
    num_train_epochs = args.num_epochs,
    #DAPO
    save_steps = 10,
    max_grad_norm = 0.2,
    loss_type = 'dapo',
    importance_sampling_level = "token", # "sequence" or "token"
    scale_rewards = "group",
    multi_objective_aggregation = 'normalize_then_sum',
    mask_truncated_completions = True,
    report_to = "none", #
    output_dir = args.output_dir,
    epsilon_high=0.28,
    epsilon = 0.2,
    beta=0.0,
    remove_unused_columns = False,
    vllm_sampling_params = vllm_sampling_params

)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer.tokenizer,
    reward_funcs=[
        reward_fidelity,
        reward_fast_token_f1,
        reward_exact_match_f1

    ],
    args = training_args,
    train_dataset = dataset,
)



trainer.train()
# trainer.train(resume_from_checkpoint=True)
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)