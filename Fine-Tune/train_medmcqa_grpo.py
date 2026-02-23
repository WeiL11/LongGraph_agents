#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.46.0",
#     "accelerate>=0.24.0",
#     "datasets>=3.0.0",
#     "trackio",
# ]
# ///

"""
=============================================================================
  MedMCQA Structured-JSON Classifier — GRPO Fine-Tuning (Phase 2)
=============================================================================

  Base model:  wei25/qwen3-0.6b-medmcqa-sft   (Phase 1 SFT output)
  Data:        openlifescienceai/medmcqa        (500-sample GRPO subset)
  Method:      Group Relative Policy Optimization (GRPO)
  Hardware:    t4-small  (1x NVIDIA T4 16 GB)

  ── Why GRPO after SFT? ────────────────────────────────────────────────
  SFT teaches format + domain vocabulary from fixed "perfect" answers.
  GRPO lets the model explore and self-improve:
    1.  Generate G completions per prompt
    2.  Score each with rule-based rewards (no reward model)
    3.  Reinforce completions that are correct + well-formatted

  ── GRPO Objective ──────────────────────────────────────────────────────
  For prompt q, generate G completions {o_1, ..., o_G}.
  Compute group-relative advantage:

      A_hat_i = ( r_i - mean(r) ) / std(r)

  Optimise the clipped surrogate:

      L_GRPO = -(1 / sum|o_i|) * sum_i sum_t
          min( rho_{i,t} * A_hat_i ,
               clip(rho_{i,t}, 1-eps, 1+eps) * A_hat_i )

  where rho_{i,t} = pi_theta / pi_ref  is the importance ratio.

  ── Reward Functions (Rule-Based, No Reward Model) ──────────────────────
  R = R_json(0/1) + R_schema(0/1) + R_accuracy(0/1)    range [0, 3]

  R_json     :  output parses as valid JSON
  R_schema   :  JSON has predicted_category in {A,B,C,D},
                confidence in [0,1], reason is non-empty string
  R_accuracy :  predicted_category matches ground-truth label

  ── Cost Estimate ───────────────────────────────────────────────────────
  ~30-50 min on t4-small  →  ~$0.50
=============================================================================
"""

import json
import os

import torch
import trackio
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig


# ═══════════════════════════════════════════════════════════════════════════
#  1.  REWARD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _extract_text(completion):
    """Extract text from a completion (handles both str and chat formats)."""
    if isinstance(completion, list):
        # Chat format: list of message dicts
        return completion[-1]["content"] if completion else ""
    return str(completion)


def json_validity_reward(completions, **kwargs):
    """R1: Can the output be parsed as valid JSON?  (0 or 1)"""
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        try:
            json.loads(text)
            rewards.append(1.0)
        except (json.JSONDecodeError, TypeError):
            rewards.append(0.0)
    return rewards


def schema_compliance_reward(completions, **kwargs):
    """R2: Does the JSON have the 3 required keys with correct types?  (0 or 1)

    Schema:
      - predicted_category: str in {"A", "B", "C", "D"}
      - confidence: float in [0, 1]
      - reason: non-empty string
    """
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        try:
            obj = json.loads(text)
            has_cat = obj.get("predicted_category") in ("A", "B", "C", "D")
            conf = obj.get("confidence")
            has_conf = isinstance(conf, (int, float)) and 0 <= conf <= 1
            reason = obj.get("reason")
            has_reason = isinstance(reason, str) and len(reason.strip()) > 0
            rewards.append(1.0 if (has_cat and has_conf and has_reason) else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def accuracy_reward(completions, ground_truth, **kwargs):
    """R3: Does predicted_category match the gold label?  (0 or 1)

    The `ground_truth` column is automatically passed from the dataset.
    """
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        text = _extract_text(completion)
        try:
            obj = json.loads(text)
            rewards.append(1.0 if obj.get("predicted_category") == gt else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


# ═══════════════════════════════════════════════════════════════════════════
#  2.  DATASET PREPARATION  — 500 MedMCQA questions → prompt-only format
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a medical MCQ classifier. "
    "Given a medical multiple-choice question with 4 options (A, B, C, D), "
    "analyze the question and respond with ONLY valid JSON in this exact format:\n"
    '{"predicted_category": "<A|B|C|D>", "confidence": <float 0-1>, "reason": "<concise explanation>"}\n'
    "Output nothing else — no markdown, no extra text, just the JSON object."
)

COP_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def format_grpo_prompt(example):
    """Convert MedMCQA row to GRPO format: prompt + ground_truth.

    GRPO datasets are prompt-only — the model generates completions online.
    The `ground_truth` column is passed to reward functions automatically.
    """
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {example['question']}\n"
                f"A) {example['opa']}\n"
                f"B) {example['opb']}\n"
                f"C) {example['opc']}\n"
                f"D) {example['opd']}"
            ),
        },
    ]
    return {
        "prompt": prompt,
        "ground_truth": COP_MAP[example["cop"]],
    }


print("=" * 60)
print("  MedMCQA GRPO — Loading & Formatting Dataset")
print("=" * 60)

# Load 500 samples DISJOINT from SFT set (indices 1000-1499)
raw = load_dataset(
    "openlifescienceai/medmcqa",
    split="train",
    trust_remote_code=True,
)
raw_grpo = raw.shuffle(seed=42).select(range(1000, 1500))
print(f"✅  Loaded {len(raw_grpo)} samples for GRPO (indices 1000-1499)")

dataset = raw_grpo.map(
    format_grpo_prompt,
    remove_columns=raw_grpo.column_names,
    desc="Formatting → GRPO prompt-only",
)

# Sanity check
print("\n── Sample GRPO prompt ──")
print(json.dumps(dataset[0]["prompt"], indent=2, ensure_ascii=False)[:500])
print(f"Ground truth: {dataset[0]['ground_truth']}")
print("─" * 60)


# ═══════════════════════════════════════════════════════════════════════════
#  3.  TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

HF_USERNAME = os.environ.get("HF_USERNAME", "wei25")
MODEL_NAME = "qwen3-0.6b-medmcqa-grpo"
HUB_REPO = f"{HF_USERNAME}/{MODEL_NAME}"

# The SFT model from Phase 1
SFT_MODEL = os.environ.get("SFT_MODEL", f"{HF_USERNAME}/qwen3-0.6b-medmcqa-sft")

config = GRPOConfig(
    # Output & Hub
    output_dir=MODEL_NAME,
    push_to_hub=True,
    hub_model_id=HUB_REPO,
    hub_strategy="every_save",
    hub_private_repo=False,

    # ── Core GRPO hyper-parameters ──
    max_steps=400,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,        # effective batch = 8 prompts
    learning_rate=5e-6,                   # much lower than SFT for RL stability
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",

    # ── Generation settings ──
    num_generations=4,                    # G=4 completions per prompt
    max_completion_length=256,            # JSON output is short

    # ── Logging & checkpoints ──
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,

    # ── Memory optimisation ──
    gradient_checkpointing=True,
    bf16=True,

    # ── Monitoring (Trackio) ──
    report_to="trackio",
    project="medmcqa-classifier",
    run_name="grpo-qwen3-0.6b-400steps",
)


# ═══════════════════════════════════════════════════════════════════════════
#  4.  LoRA CONFIG  (same as SFT for consistency)
# ═══════════════════════════════════════════════════════════════════════════

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)


# ═══════════════════════════════════════════════════════════════════════════
#  5.  TRAIN
# ═══════════════════════════════════════════════════════════════════════════

BASE_MODEL = "Qwen/Qwen3-0.6B"

print(f"\n🎯  Initialising GRPO Trainer …")
print(f"    SFT adapter  : {SFT_MODEL}")
print(f"    Base model   : {BASE_MODEL}")
print(f"    Dataset      : {len(dataset)} prompts (GRPO set)")
print(f"    Max steps    : 400")
print(f"    LR           : 5e-6   (cosine, warmup 0.05)")
print(f"    Generations  : G=4 per prompt")
print(f"    Rewards      : json_validity + schema_compliance + accuracy")
print(f"    Effective BS : 8 prompts")
print(f"    LoRA rank    : 16  (alpha=32)")
print(f"    Hub target   : {HUB_REPO}")
print()

# ── Load base model + SFT LoRA adapter, then merge ──
# The SFT checkpoint is a LoRA adapter, not a full model.
# We load base → apply adapter → merge → use as starting point for GRPO.
print("📦  Loading base model + SFT adapter …")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("🔗  Applying SFT LoRA adapter …")
model = PeftModel.from_pretrained(base_model, SFT_MODEL)
print("🔀  Merging adapter into base weights …")
model = model.merge_and_unload()
print("✅  Merged model ready for GRPO\n")

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        json_validity_reward,
        schema_compliance_reward,
        accuracy_reward,
    ],
    train_dataset=dataset,
    args=config,
    peft_config=peft_config,
)

print("🚀  Starting GRPO training …\n")
trainer.train()

print("\n💾  Pushing final model to Hub …")
trainer.push_to_hub()

trackio.finish()

print("\n" + "=" * 60)
print(f"  ✅  DONE — Model saved to: https://huggingface.co/{HUB_REPO}")
print(f"  📊  Metrics at: https://huggingface.co/spaces/{HF_USERNAME}/trackio")
print("=" * 60)
