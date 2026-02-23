#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.46.0",
#     "accelerate>=0.24.0",
#     "peft>=0.7.0",
#     "datasets>=3.0.0",
#     "huggingface_hub>=0.20.0",
#     "torch",
# ]
# ///

"""
=============================================================================
  MedMCQA Evaluation — Compare Base / SFT / GRPO on Held-Out Data
=============================================================================

  Eval set:  200 MedMCQA samples  (indices 1500-1699, disjoint from train)
  Models:    Qwen3-0.6B (base), SFT checkpoint, GRPO checkpoint

  Metrics reported:
  ──────────────────────────────────────────────────────────────────────────
  accuracy          = correct / total
                      Did the model pick the right answer (A/B/C/D)?

  valid_json_rate   = json_parseable / total
                      Can json.loads() parse the raw output?

  schema_pass_rate  = schema_valid / total
                      JSON has all 3 keys with correct types?
                        - predicted_category in {A, B, C, D}
                        - confidence is float in [0, 1]
                        - reason is non-empty string

  accuracy_on_schema = correct_and_schema / schema_valid
                       Accuracy only among well-formed outputs.
                       This isolates "does the model know medicine?"
                       from "can the model follow the format?"
  ──────────────────────────────────────────────────────────────────────────

  Cost:  ~10 min on t4-small,  ~$0.15
=============================================================================
"""

import json
import os
import time

import torch
from datasets import load_dataset
from huggingface_hub import list_repo_files
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════

HF_USERNAME = os.environ.get("HF_USERNAME", "wei25")

MODELS = {
    "base": "Qwen/Qwen3-0.6B",
    "sft":  f"{HF_USERNAME}/qwen3-0.6b-medmcqa-sft",
    "grpo": f"{HF_USERNAME}/qwen3-0.6b-medmcqa-grpo",
}

EVAL_SIZE = 200          # held-out samples
EVAL_START_IDX = 1500    # disjoint from SFT (0-999) and GRPO (1000-1499)
MAX_NEW_TOKENS = 128     # JSON output is short; disable thinking to keep this tight
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COP_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

SYSTEM_PROMPT = (
    "You are a medical MCQ classifier. "
    "Given a medical multiple-choice question with 4 options (A, B, C, D), "
    "analyze the question and respond with ONLY valid JSON in this exact format:\n"
    '{"predicted_category": "<A|B|C|D>", "confidence": <float 0-1>, "reason": "<concise explanation>"}\n'
    "Output nothing else — no markdown, no extra text, just the JSON object."
)


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  MedMCQA Evaluation — Loading Held-Out Set")
print("=" * 70)

raw = load_dataset(
    "openlifescienceai/medmcqa",
    split="train",
    trust_remote_code=True,
)
raw_eval = raw.shuffle(seed=42).select(
    range(EVAL_START_IDX, EVAL_START_IDX + EVAL_SIZE)
)
print(f"✅  Loaded {len(raw_eval)} eval samples (indices {EVAL_START_IDX}-{EVAL_START_IDX + EVAL_SIZE - 1})")

# Pre-format prompts and ground truth
eval_prompts = []
eval_ground_truths = []
for ex in raw_eval:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {ex['question']}\n"
                f"A) {ex['opa']}\n"
                f"B) {ex['opb']}\n"
                f"C) {ex['opc']}\n"
                f"D) {ex['opd']}"
            ),
        },
    ]
    eval_prompts.append(messages)
    eval_ground_truths.append(COP_MAP[ex["cop"]])


# ═══════════════════════════════════════════════════════════════════════════
#  METRIC COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(outputs, ground_truths):
    """Compute all 4 evaluation metrics.

    Returns dict with:
      accuracy, valid_json_rate, schema_pass_rate, accuracy_on_schema
    """
    total = len(outputs)
    correct = 0
    json_parseable = 0
    schema_valid = 0
    correct_and_schema = 0

    per_sample = []

    for output, gt in zip(outputs, ground_truths):
        sample = {
            "raw_output": output,
            "ground_truth": gt,
            "json_valid": False,
            "schema_valid": False,
            "predicted": None,
            "correct": False,
        }

        # 1. JSON validity
        try:
            obj = json.loads(output)
            sample["json_valid"] = True
            json_parseable += 1
        except (json.JSONDecodeError, TypeError):
            per_sample.append(sample)
            continue

        # 2. Schema compliance
        cat = obj.get("predicted_category")
        conf = obj.get("confidence")
        reason = obj.get("reason")

        has_cat = cat in ("A", "B", "C", "D")
        has_conf = isinstance(conf, (int, float)) and 0 <= conf <= 1
        has_reason = isinstance(reason, str) and len(reason.strip()) > 0

        if has_cat and has_conf and has_reason:
            sample["schema_valid"] = True
            schema_valid += 1

        # 3. Accuracy
        sample["predicted"] = cat
        if cat == gt:
            sample["correct"] = True
            correct += 1
            if sample["schema_valid"]:
                correct_and_schema += 1

        per_sample.append(sample)

    metrics = {
        "accuracy": correct / total if total > 0 else 0.0,
        "valid_json_rate": json_parseable / total if total > 0 else 0.0,
        "schema_pass_rate": schema_valid / total if total > 0 else 0.0,
        "accuracy_on_schema": (
            correct_and_schema / schema_valid if schema_valid > 0 else 0.0
        ),
        "total": total,
        "correct": correct,
        "json_parseable": json_parseable,
        "schema_valid": schema_valid,
        "correct_and_schema": correct_and_schema,
    }

    return metrics, per_sample


# ═══════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def _is_peft_adapter(model_id: str) -> bool:
    """Return True if the hub repo contains an adapter_config.json (LoRA adapter)."""
    try:
        repo_files = [f for f in list_repo_files(model_id)]
        return "adapter_config.json" in repo_files
    except Exception:
        return False


def run_inference(model_id, prompts, label):
    """Run greedy inference on all prompts, return list of raw outputs."""
    print(f"\n{'─' * 70}")
    print(f"  [{label.upper()}]  Loading: {model_id}")
    print(f"{'─' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA adapter repos need AutoPeftModelForCausalLM + merge_and_unload()
    if _is_peft_adapter(model_id):
        print(f"  ℹ️   Detected LoRA adapter — loading base + merging weights …")
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    outputs = []
    start = time.time()

    for i, messages in enumerate(prompts):
        # Apply chat template — disable Qwen3 thinking mode for fast JSON output
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,   # Qwen3: suppress <think> blocks
            )
        except TypeError:
            # Tokenizer doesn't support enable_thinking kwarg (non-Qwen3 model)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,          # greedy
                temperature=None,
                top_p=None,
            )

        # Decode only the generated tokens (exclude prompt)
        prompt_len = inputs["input_ids"].shape[1]
        output_ids = generated[0][prompt_len:]
        raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # Strip any residual <think>...</think> blocks (safety net)
        import re as _re
        raw_output = _re.sub(r"<think>.*?</think>", "", raw_output, flags=_re.DOTALL).strip()
        # If the model wrapped JSON in markdown code fences, unwrap them
        if raw_output.startswith("```"):
            raw_output = _re.sub(r"^```[a-z]*\n?", "", raw_output)
            raw_output = _re.sub(r"\n?```$", "", raw_output).strip()

        outputs.append(raw_output)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"    {i+1}/{len(prompts)}  ({elapsed:.1f}s)")

    elapsed = time.time() - start
    print(f"  ✅  Generated {len(outputs)} outputs in {elapsed:.1f}s")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return outputs


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN — Run all models and compare
# ═══════════════════════════════════════════════════════════════════════════

all_results = {}

for label, model_id in MODELS.items():
    try:
        outputs = run_inference(model_id, eval_prompts, label)
        metrics, per_sample = compute_metrics(outputs, eval_ground_truths)
        all_results[label] = {
            "model_id": model_id,
            "metrics": metrics,
            "per_sample": per_sample,
        }
    except Exception as e:
        print(f"\n  ⚠️  [{label}] Failed: {e}")
        all_results[label] = {
            "model_id": model_id,
            "metrics": None,
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════════════════════

print("\n")
print("=" * 70)
print("  EVALUATION RESULTS  —  MedMCQA Structured-JSON Classifier")
print("=" * 70)

# Header
header = f"{'Metric':<25}"
for label in MODELS:
    header += f"  {label:>10}"
print(header)
print("─" * 70)

metric_names = ["accuracy", "valid_json_rate", "schema_pass_rate", "accuracy_on_schema"]

for metric_name in metric_names:
    row = f"{metric_name:<25}"
    for label in MODELS:
        result = all_results.get(label, {})
        metrics = result.get("metrics")
        if metrics:
            val = metrics[metric_name]
            row += f"  {val:>9.1%}"
        else:
            row += f"  {'ERR':>10}"
    print(row)

# Raw counts
print("─" * 70)
for count_name in ["correct", "json_parseable", "schema_valid", "total"]:
    row = f"{count_name:<25}"
    for label in MODELS:
        result = all_results.get(label, {})
        metrics = result.get("metrics")
        if metrics:
            row += f"  {metrics[count_name]:>10}"
        else:
            row += f"  {'—':>10}"
    print(row)

print("=" * 70)

# Save to JSON
output_path = os.path.join(os.path.dirname(__file__) or ".", "eval_results.json")
# Serialise — strip per_sample for the summary file to keep it small
summary = {}
for label, result in all_results.items():
    summary[label] = {
        "model_id": result["model_id"],
        "metrics": result.get("metrics"),
        "error": result.get("error"),
    }

with open(output_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n📄  Results saved to: {output_path}")

# Also save full per-sample predictions for error analysis
full_path = output_path.replace(".json", "_full.json")
with open(full_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"📄  Per-sample details saved to: {full_path}")

print("\n✅  Evaluation complete.")
