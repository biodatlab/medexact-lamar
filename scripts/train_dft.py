from unsloth import FastLanguageModel, FastModel
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
import pandas as pd
from unsloth.chat_templates import standardize_data_formats, train_on_responses_only
import argparse

parser = argparse.ArgumentParser(description="Dynamic Finetuning (DFT) for Clinical Decision Extraction")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-4B", help="Base model name")
parser.add_argument("--data_path", type=str, default="./dataset/train.csv", help="Path to training data")
parser.add_argument("--output_dir", type=str, default="./qwen3_5-4b_lora256_alpha32_dft", help="Output directory")
parser.add_argument("--lora_rank", type=int, default=256, help="LoRA rank")
parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=1, help="Per device train batch size")
parser.add_argument("--grad_accumulation", type=int, default=8, help="Gradient accumulation steps")
parser.add_argument("--max_seq_length", type=int, default=24576, help="Maximum sequence length")
args = parser.parse_args()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_name,
    max_seq_length = args.max_seq_length,
    load_in_4bit = False,     # MoE QLoRA not recommended, dense 27B is fine
    load_in_16bit = True,     # bf16/16-bit LoRA
    full_finetuning = False,
    unsloth_tiled_mlp = True,
    use_gradient_checkpointing = "unsloth"
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
    random_state = 3407,
    use_rslora = True,
    loftq_config = None,
)
train = pd.read_csv(args.data_path)

USER_PROMPT_TEMPLATE = """You are an expert specializes in extracting clinical decisions from a patient's discharge summary.

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

    conversation = [
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(raw_text=raw_text)},
        {"role": "assistant", "content": gold_token}
    ]

    formatted_data.append({"messages": conversation})

# Change to Hugging Face Dataset
dataset = Dataset.from_list(formatted_data)

# Unsloth standardize_data_formats
dataset = standardize_data_formats(dataset)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=["messages"])

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    tokenizer = tokenizer.tokenizer,
    args = SFTConfig(
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accumulation,
        warmup_ratio = 0.05,
        num_train_epochs = args.num_epochs,
        learning_rate = args.learning_rate,
        logging_steps = 1,
        optim = "adamw_8bit",
        save_strategy = "epoch",
        output_dir = args.output_dir,
        weight_decay = 0.001,
        lr_scheduler_type = "cosine",
        bf16 = True,
        seed = 3407,
        remove_unused_columns = True,
        report_to = "tensorboard",
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False}
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
    tokenizer = tokenizer.tokenizer,
)

trainer.train()