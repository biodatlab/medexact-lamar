import argparse
from unsloth import FastLanguageModel
from peft import PeftModel
import torch

parser = argparse.ArgumentParser(description="Merge LoRA adapter and push to Hugging Face Hub")
parser.add_argument("--base_model", type=str, default="Keetawan/Qwen3.5-4B_LoRA_Exact_BF16_Rank256_Alpha32_DFT")
parser.add_argument("--adapter_path", type=str, default="./weights/qwen3_5-4b_dft_lora_r2_grpo_dapo_f1_reward")
parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace repo id to push to (e.g. Keetawan/my-model)")
parser.add_argument("--token", type=str, required=True, help="HuggingFace API token")
parser.add_argument("--private", action="store_true", default=True)
args = parser.parse_args()

# Load base model + adapter with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.base_model,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    load_in_16bit=True,
)

# Load adapter into model
model = PeftModel.from_pretrained(
    model,
    args.adapter_path,
    torch_dtype=torch.bfloat16,
)

# Merge
model = model.merge_and_unload()

model.push_to_hub(args.repo_id, token=args.token, private=args.private)
tokenizer.push_to_hub(args.repo_id, token=args.token, private=args.private)
