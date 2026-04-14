import argparse
import numpy as np
from vllm import LLM, SamplingParams
import pandas as pd

parser = argparse.ArgumentParser(description="Run vLLM inference on val and test sets")
parser.add_argument("--model", type=str, default="Keetawan/Qwen3.5-4B_LoRA_Exact_BF16_Rank256_Alpha32_DFT")
parser.add_argument("--val_csv", type=str, default="./dataset/val.csv")
parser.add_argument("--test_csv", type=str, default="./dataset/test.csv")
parser.add_argument("--val_out", type=str, default="val_model1_dft.csv")
parser.add_argument("--test_out", type=str, default="test_model1_dft.csv")
parser.add_argument("--tensor_parallel_size", type=int, default=1)
parser.add_argument("--max_model_len", type=int, default=32768)
parser.add_argument("--max_tokens", type=int, default=24576)
parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
parser.add_argument("--max_num_seqs", type=int, default=8)
args = parser.parse_args()

# Initialize LLM
llm = LLM(
    model=args.model,
    tensor_parallel_size=args.tensor_parallel_size,
    max_model_len=args.max_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
    trust_remote_code=True,
    dtype="bfloat16",
    max_num_seqs=args.max_num_seqs,
)

tokenizer = llm.get_tokenizer()

# Load Datasets
val_df = pd.read_csv(args.val_csv)
test_df = pd.read_csv(args.test_csv)

system_prompt = """You are an expert specializes in extracting clinical decisions from a patient's discharge summary.

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


# Prepare Prompts
def create_prompts(df):
    prompts = []
    for raw_text in df["raw_text"]:
        messages = [{"role": "user", "content": system_prompt.format(raw_text=raw_text)}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False
        )
        prompts.append(prompt_text)
    return prompts

val_prompts = create_prompts(val_df)
test_prompts = create_prompts(test_df)

# combine prompts of val and test
all_prompts = val_prompts + test_prompts
num_val = len(val_prompts)

# Generate (Inference)
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    max_tokens=args.max_tokens,
    logprobs=1
)

outputs = llm.generate(all_prompts, sampling_params)

# Process Outputs
def process_outputs(outputs_subset):
    results = []
    entropies = []
    for o in outputs_subset:
        generated_text = o.outputs[0].text
        results.append(generated_text)

        token_ids = o.outputs[0].token_ids
        logprobs_list = o.outputs[0].logprobs

        seq_logprobs = []
        if logprobs_list is not None:
            for token_id, logprob_dict in zip(token_ids, logprobs_list):
                if logprob_dict and token_id in logprob_dict:
                    lp_obj = logprob_dict[token_id]
                    if hasattr(lp_obj, "logprob"):
                        seq_logprobs.append(lp_obj.logprob)
                    else:
                        seq_logprobs.append(lp_obj)

        if len(seq_logprobs) > 0:
            entropy = -np.sum(seq_logprobs) / len(seq_logprobs)
        else:
            entropy = 0.0

        entropies.append(entropy)

    return results, entropies

# separate outputs of val and test
val_outputs = outputs[:num_val]
test_outputs = outputs[num_val:]

val_results, val_entropies = process_outputs(val_outputs)
test_results, test_entropies = process_outputs(test_outputs)

# Save Separate CSVs
# save val
val_df["predicted"] = val_results
val_df["entropy"] = val_entropies
val_df.to_csv(args.val_out, index=False)

# save test
test_df["predicted"] = test_results
test_df["entropy"] = test_entropies
test_df.to_csv(args.test_out, index=False)

print("Inference completed and saved successfully!")