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
  MedMCQA Structured-JSON Classifier — SFT Fine-Tuning Pipeline
=============================================================================

  Model:    Qwen/Qwen3-0.6B  (0.6 B params, Apache-2.0)
  Data:     openlifescienceai/medmcqa  (1 000-sample subset)
  Method:   Supervised Fine-Tuning (SFT) with LoRA
  Hardware: t4-small  (1× NVIDIA T4 16 GB)
  Output:   Structured JSON  { predicted_category, confidence, reason }

  ── Training Objective ──────────────────────────────────────────────────
  Standard causal-LM cross-entropy over the assistant response tokens:

      L_SFT = - (1/T) Σ_{t=1}^{T}  log  π_θ( y_t | y_{<t}, x )

  where:
      x   = system prompt ⊕ user question (with MCQ options)
      y   = target JSON string
      T   = number of tokens in y
      π_θ = model parameterised by θ (base weights + LoRA adapters)

  ── LoRA Adapter Math ───────────────────────────────────────────────────
  For a pretrained weight matrix  W₀ ∈ ℝ^{d×k} ,  LoRA decomposes the
  update as a low-rank product:

      W = W₀ + (α / r) · B A        B ∈ ℝ^{d×r},   A ∈ ℝ^{r×k}

  Trainable params  ≈  2 · r · d · n_layers   ≪   total params

  Config used here:  r = 16,  α = 32,  dropout = 0.05
  → ~0.6 % of total params are trainable (~3.6 M / 600 M)

  ── Hyperparameters ─────────────────────────────────────────────────────
  max_steps        = 800       (≈ 3.2 effective epochs over 1 000 samples)
  learning_rate    = 2 × 10⁻⁴
  warmup_ratio     = 0.03      (24 warmup steps)
  batch_size       = 4         (per device)
  grad_accum       = 4         (effective batch = 16)
  max_length       = 1024      (tokens)
  scheduler        = cosine
  precision        = bf16 / fp16  (auto)

  ── Cost Estimate ───────────────────────────────────────────────────────
  ~15-25 min on t4-small  →  ~$0.20-$0.30
=============================================================================
"""

import json
import os

import trackio
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# ═══════════════════════════════════════════════════════════════════════════
#  1.  DATASET PREPARATION  — 1 000 MedMCQA questions → chat format
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a medical MCQ classifier. "
    "Given a medical multiple-choice question with 4 options (A, B, C, D), "
    "analyze the question and respond with ONLY valid JSON in this exact format:\n"
    '{"predicted_category": "<A|B|C|D>", "confidence": <float 0-1>, "reason": "<concise explanation>"}\n'
    "Output nothing else — no markdown, no extra text, just the JSON object."
)

COP_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def format_to_chat(example):
    """Convert a MedMCQA row into a chat-messages list for SFT.

    Input columns:  question, opa, opb, opc, opd, cop (0-3), exp
    Output column:  messages  (list of {role, content} dicts)
    """
    user_msg = (
        f"Question: {example['question']}\n"
        f"A) {example['opa']}\n"
        f"B) {example['opb']}\n"
        f"C) {example['opc']}\n"
        f"D) {example['opd']}"
    )

    # Build ground-truth JSON response
    answer_letter = COP_MAP[example["cop"]]
    reason = (example.get("exp") or "").strip()
    # Truncate long explanations to keep within context budget
    if len(reason) > 250:
        reason = reason[:247] + "..."
    if not reason:
        reason = "Based on medical knowledge and clinical reasoning."

    assistant_msg = json.dumps(
        {
            "predicted_category": answer_letter,
            "confidence": 0.95,
            "reason": reason,
        },
        ensure_ascii=False,
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


print("=" * 60)
print("  MedMCQA SFT — Loading & Formatting Dataset")
print("=" * 60)

# Load 1 000 samples (deterministic seed for reproducibility)
raw = load_dataset(
    "openlifescienceai/medmcqa",
    split="train",
    trust_remote_code=True,
)
raw_subset = raw.shuffle(seed=42).select(range(1000))
print(f"✅  Loaded {len(raw_subset)} samples from MedMCQA train split")

# Convert to chat format
dataset = raw_subset.map(
    format_to_chat,
    remove_columns=raw_subset.column_names,
    desc="Formatting → chat JSON",
)

# Sanity check — print one formatted example
print("\n── Sample formatted example ──")
print(json.dumps(dataset[0]["messages"], indent=2, ensure_ascii=False)[:600])
print("─" * 60)


# ═══════════════════════════════════════════════════════════════════════════
#  2.  TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# --- Hub / output settings ---
HF_USERNAME = os.environ.get("HF_USERNAME", "wei25")
MODEL_NAME = "qwen3-0.6b-medmcqa-sft"
HUB_REPO = f"{HF_USERNAME}/{MODEL_NAME}"

config = SFTConfig(
    # Output & Hub
    output_dir=MODEL_NAME,
    push_to_hub=True,
    hub_model_id=HUB_REPO,
    hub_strategy="every_save",
    hub_private_repo=False,

    # ── Core training hyper-parameters ──
    max_steps=800,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,       # effective batch = 16
    learning_rate=2e-4,
    warmup_ratio=0.03,                   # ~24 warmup steps
    lr_scheduler_type="cosine",
    max_length=1024,

    # ── Logging & checkpoints ──
    logging_steps=20,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,

    # ── Memory optimisation ──
    gradient_checkpointing=True,
    bf16=True,

    # ── Monitoring (Trackio) ──
    report_to="trackio",
    project="medmcqa-classifier",
    run_name="sft-qwen3-0.6b-800steps",
)


# ═══════════════════════════════════════════════════════════════════════════
#  3.  LoRA ADAPTER CONFIG
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
#  4.  TRAIN
# ═══════════════════════════════════════════════════════════════════════════

print("\n🎯  Initialising SFTTrainer …")
print(f"    Model        : Qwen/Qwen3-0.6B")
print(f"    Dataset      : {len(dataset)} chat examples")
print(f"    Max steps    : 800")
print(f"    LR           : 2e-4   (cosine, warmup 0.03)")
print(f"    Effective BS : 16")
print(f"    LoRA rank    : 16  (α=32)")
print(f"    Hub target   : {HUB_REPO}")
print()

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=dataset,
    args=config,
    peft_config=peft_config,
)

print("🚀  Starting training …\n")
trainer.train()

print("\n💾  Pushing final model to Hub …")
trainer.push_to_hub()

trackio.finish()

print("\n" + "=" * 60)
print(f"  ✅  DONE — Model saved to: https://huggingface.co/{HUB_REPO}")
print(f"  📊  Metrics at: https://huggingface.co/spaces/{HF_USERNAME}/trackio")
print("=" * 60)
