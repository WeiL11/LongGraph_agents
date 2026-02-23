# MedMCQA Structured-JSON Classifier — Full Training & Eval Pipeline

> **Goal**: Demo an end-to-end healthcare ML pipeline — from data curation
> through SFT, GRPO reinforcement learning, to rigorous evaluation — using a
> tiny model on a real medical exam dataset.

---

## 0. Architecture Overview

```
openlifescienceai/medmcqa  (194 k MCQs)
            │
    ┌───────┴──────────────────────┐
    │  shuffle(seed=42)            │
    │  split into 3 disjoint sets  │
    └───────┬──────────────────────┘
            │
   ┌────────┼────────────┐
   ▼        ▼            ▼
 1 000    500          200
 SFT    GRPO          EVAL
 set    set           set
   │        │            │
   ▼        ▼            │
┌──────┐ ┌──────┐       │
│Phase1│ │Phase2│       │
│ SFT  │→│ GRPO │       │
└──┬───┘ └──┬───┘       │
   │        │            │
   ▼        ▼            ▼
  wei25/    wei25/    ┌──────────┐
  qwen3-   qwen3-    │ Phase 3  │
  0.6b-    0.6b-     │   EVAL   │
  medmcqa  medmcqa   │ compare: │
  -sft     -grpo     │ base/sft │
                      │ /grpo   │
                      └──────────┘
```

### Model

| Property | Value |
|----------|-------|
| Base model | `Qwen/Qwen3-0.6B` |
| Parameters | 0.6 B |
| License | Apache-2.0 |
| Why | Tiny, fast to train, Qwen3 arch is well-supported by TRL GRPO |

### Dataset

| Property | Value |
|----------|-------|
| Source | `openlifescienceai/medmcqa` |
| Domain | AIIMS & NEET PG medical entrance exams |
| Columns | `question`, `opa..opd`, `cop` (0-3 int), `exp`, `subject_name` |
| License | Apache-2.0 |

### Structured Output Format

Every model response must be **exactly** this JSON — nothing else:

```json
{
  "predicted_category": "B",
  "confidence": 0.85,
  "reason": "Vitamin B12 is exclusively found in animal products..."
}
```

---

## 1. Phase 1 — Supervised Fine-Tuning (SFT)

**Script**: [`Fine-Tune/train_medmcqa_sft.py`](./train_medmcqa_sft.py)

### 1.1 Objective

Standard causal-LM cross-entropy over the assistant response tokens:

```
L_SFT = -(1/T) * sum_{t=1}^{T} log pi_theta( y_t | y_{<t}, x )
```

Where:
- `x = system_prompt + user_question` (with MCQ options A-D)
- `y = target JSON string`
- `T = number of tokens in y`
- `pi_theta` = model parameterised by `theta` (frozen base + LoRA adapters)

### 1.2 LoRA Adapter Decomposition

For a pre-trained weight matrix `W_0` in `R^{d x k}`, LoRA decomposes the
update as a low-rank product:

```
W = W_0 + (alpha / r) * B @ A       B in R^{d x r},  A in R^{r x k}
```

Trainable params ~ `2 * r * d * n_layers` << total params.

| LoRA param | Value |
|------------|-------|
| `r` | 16 |
| `alpha` | 32 |
| `dropout` | 0.05 |
| `target_modules` | q_proj, k_proj, v_proj, o_proj |
| Trainable params | ~3.6 M (0.6% of 600 M) |

### 1.3 Data Preparation

- Source: first 1 000 samples (after `shuffle(seed=42)`)
- Format: **chat-messages** (TRL SFT standard)
  ```
  [
    { "role": "system",    "content": "<classifier instructions>" },
    { "role": "user",      "content": "Question: ...\nA) ...\nB) ..." },
    { "role": "assistant", "content": "{\"predicted_category\": \"C\", ...}" }
  ]
  ```
- Ground-truth confidence fixed at `0.95`
- Explanations (`exp`) truncated to 250 chars

### 1.4 Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `max_steps` | 800 | ~3.2 effective epochs over 1 000 samples |
| `learning_rate` | 2e-4 | |
| `warmup_ratio` | 0.03 | ~24 warmup steps |
| `per_device_train_batch_size` | 4 | |
| `gradient_accumulation_steps` | 4 | effective batch = 16 |
| `max_length` | 1024 | tokens |
| `lr_scheduler_type` | cosine | |
| `bf16` | True | |
| `gradient_checkpointing` | True | saves ~40% VRAM |

### 1.5 Infrastructure

| Setting | Value |
|---------|-------|
| Hardware | `t4-small` (1x NVIDIA T4 16 GB) |
| Estimated time | 15-25 min |
| Estimated cost | ~$0.25 |
| Monitoring | Trackio (`medmcqa-classifier` / `sft-qwen3-0.6b-800steps`) |
| Hub target | `wei25/qwen3-0.6b-medmcqa-sft` |

### 1.6 Expected Outcome

The SFT model should:
- Always produce valid JSON (high `valid_json_rate`)
- Learn the schema structure (high `schema_pass_rate`)
- Pick up basic medical knowledge from the explanations
- Serve as a strong starting point for GRPO RL refinement

---

## 2. Phase 2 — Group Relative Policy Optimization (GRPO)

**Script**: `Fine-Tune/train_medmcqa_grpo.py` *(to be created)*

### 2.1 Why GRPO After SFT?

SFT teaches the model *format* and *domain vocabulary*, but it learns from a
fixed set of "perfect" answers. GRPO lets the model:
- **Explore** — generate multiple candidate answers per question
- **Self-improve** — learn from its own correct answers (no reward model needed)
- **Sharpen** — the rule-based reward directly optimises accuracy + JSON compliance

This is the same approach used in **DeepSeek-R1** for math reasoning.

### 2.2 GRPO Objective

At each step, for a prompt `q`, generate `G` completions `{o_1, ..., o_G}`.
Score each with reward `r_i`. Compute group-relative advantage:

```
A_hat_i = ( r_i - mean(r_1, ..., r_G) ) / std(r_1, ..., r_G)
```

Then optimise the clipped surrogate objective:

```
L_GRPO(theta) = -(1 / sum|o_i|) * sum_{i=1}^{G} sum_{t=1}^{|o_i|}
    min(
        rho_{i,t} * A_hat_i,
        clip( rho_{i,t}, 1-eps, 1+eps ) * A_hat_i
    )
```

Where `rho_{i,t} = pi_theta(o_{i,t} | q, o_{i,<t}) / pi_ref(o_{i,t} | q, o_{i,<t})`
is the importance sampling ratio.

**Key insight**: The advantage is *relative within the group* — no learned
reward model needed. We only need a rule-based reward function.

### 2.3 Reward Functions (Rule-Based)

Three separate reward signals, summed by the trainer:

```python
def json_validity_reward(completions, **kwargs):
    """R1: Can the output be parsed as valid JSON?"""
    rewards = []
    for completion in completions:
        try:
            json.loads(completion)
            rewards.append(1.0)
        except (json.JSONDecodeError, TypeError):
            rewards.append(0.0)
    return rewards


def schema_compliance_reward(completions, **kwargs):
    """R2: Does the JSON have the 3 required keys with correct types?"""
    rewards = []
    for completion in completions:
        try:
            obj = json.loads(completion)
            has_cat   = obj.get("predicted_category") in ("A","B","C","D")
            has_conf  = isinstance(obj.get("confidence"), (int, float)) and 0 <= obj["confidence"] <= 1
            has_reason = isinstance(obj.get("reason"), str) and len(obj["reason"]) > 0
            rewards.append(1.0 if (has_cat and has_conf and has_reason) else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def accuracy_reward(completions, ground_truth, **kwargs):
    """R3: Does predicted_category match the gold label?"""
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        try:
            obj = json.loads(completion)
            rewards.append(1.0 if obj.get("predicted_category") == gt else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards
```

**Total reward per completion**: `R = R1 + R2 + R3` (range 0-3).
- Perfect answer: `3.0`  (valid JSON + correct schema + correct answer)
- Right format, wrong answer: `2.0`
- Broken JSON: `0.0`

### 2.4 Data Preparation (Prompt-Only)

GRPO uses a **prompt-only** dataset — the model generates completions online.
The dataset must have:
- `prompt` column: the system+user message (without assistant response)
- `ground_truth` column: the correct answer letter ("A"/"B"/"C"/"D") for the
  accuracy reward function

```python
# 500 samples, disjoint from SFT set (indices 1000-1499)
raw_grpo = raw.shuffle(seed=42).select(range(1000, 1500))

def format_grpo_prompt(example):
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Question: {example['question']}\n"
                                      f"A) {example['opa']}\n..."
        }
    ]
    return {
        "prompt": prompt,
        "ground_truth": COP_MAP[example["cop"]],
    }
```

### 2.5 Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Starting model | `wei25/qwen3-0.6b-medmcqa-sft` | Phase 1 output |
| `max_steps` | 400 | smaller than SFT — RL is more sample-efficient |
| `learning_rate` | 5e-6 | 40x lower than SFT — RL needs small steps |
| `warmup_ratio` | 0.05 | |
| `per_device_train_batch_size` | 2 | smaller — each prompt generates G completions |
| `gradient_accumulation_steps` | 4 | effective batch = 8 prompts |
| `num_generations` | 4 | G=4 completions per prompt per step |
| `max_completion_length` | 256 | JSON output is short |
| `max_prompt_length` | 768 | |
| `lr_scheduler_type` | cosine | |
| `bf16` | True | |
| `gradient_checkpointing` | True | |
| LoRA | same config as SFT | r=16, alpha=32 |

### 2.6 Infrastructure

| Setting | Value |
|---------|-------|
| Hardware | `t4-small` (1x T4 16 GB) |
| Estimated time | 30-50 min (generation is slower than SFT) |
| Estimated cost | ~$0.50 |
| Monitoring | Trackio (`medmcqa-classifier` / `grpo-qwen3-0.6b-400steps`) |
| Hub target | `wei25/qwen3-0.6b-medmcqa-grpo` |

### 2.7 Expected Outcome

Compared to SFT alone, GRPO should:
- Maintain or improve `valid_json_rate` and `schema_pass_rate`
- **Improve `accuracy`** — the model learns from its own correct reasoning
- Show increasing reward curves on the Trackio dashboard

---

## 3. Phase 3 — Evaluation

**Script**: `Fine-Tune/eval_medmcqa.py` *(to be created)*

### 3.1 Held-Out Eval Set

- **200 samples**, disjoint from both SFT and GRPO training data
- Indices 1500-1699 after `shuffle(seed=42)`

### 3.2 Models Compared

| Label | Model | Description |
|-------|-------|-------------|
| `base` | `Qwen/Qwen3-0.6B` | Zero-shot, no fine-tuning |
| `sft` | `wei25/qwen3-0.6b-medmcqa-sft` | After Phase 1 |
| `grpo` | `wei25/qwen3-0.6b-medmcqa-grpo` | After Phase 2 |

### 3.3 Metrics

For each model, run greedy inference on all 200 eval prompts and compute:

| Metric | Formula | What it measures |
|--------|---------|------------------|
| **`accuracy`** | `correct / total` | Overall medical QA accuracy (parsed or not) |
| **`valid_json_rate`** | `json_parseable / total` | Can the output be parsed by `json.loads()`? |
| **`schema_pass_rate`** | `schema_valid / total` | JSON has all 3 keys with correct types? |
| **`accuracy_on_schema`** | `correct / schema_valid` | Accuracy *only* among well-formed outputs |

Where:
```
correct      = predicted_category matches ground truth
json_parseable = json.loads(output) does not raise
schema_valid   = json_parseable AND has "predicted_category" in {A,B,C,D}
                 AND "confidence" is float in [0,1]
                 AND "reason" is non-empty string
```

### 3.4 Actual Results (200 held-out samples, greedy decoding)

| Metric | base | sft | grpo | Notes |
|--------|------|-----|------|-------|
| `accuracy` | 27.5% | **34.0%** | 27.5% | SFT: +6.5pp lift |
| `valid_json_rate` | 98.5% | **98.5%** | 98.5% | All models output valid JSON |
| `schema_pass_rate` | 95.5% | **98.5%** | 95.5% | SFT best schema compliance |
| `accuracy_on_schema` | 28.8% | **34.5%** | 28.8% | Accuracy among well-formed outputs |

**HF Job IDs (for reference):**
| Phase | Job ID | Status |
|-------|--------|--------|
| SFT | `699b9aa152d1c53b7df7d387` | COMPLETED (step ~512/800, timeout at 1h) |
| GRPO | `699bacd052d1c53b7df7d39e` | COMPLETED (400 steps) |
| Eval | `699be2881aad19adb8aad4d9` | COMPLETED |

**Hub Models:**
- SFT: https://huggingface.co/wei25/qwen3-0.6b-medmcqa-sft
- GRPO: https://huggingface.co/wei25/qwen3-0.6b-medmcqa-grpo

### 3.5 Key Findings

1. **Qwen3-0.6B is already a strong JSON outputter** — base model achieves 98.5%
   `valid_json_rate` with zero fine-tuning; the structured-output format came for
   free from pre-training.

2. **SFT (+6.5 pp accuracy)** — training on 1000 MedMCQA samples with the JSON
   classifier format meaningfully improved answer accuracy, even though the SFT
   run was cut short at ~512/800 steps due to a 1h wall-clock timeout.

3. **GRPO reward collapse** — all reward values were `0.0` throughout training
   (`frac_reward_zero_std = 1.0`). Root cause: the GRPO LoRA adapter was
   initialised on top of the merged SFT model, but its `adapter_config.json`
   records `Qwen/Qwen3-0.6B` as the base (not the SFT model). When the eval
   loads the GRPO adapter via `AutoPeftModelForCausalLM`, it applies near-zero
   LoRA weights to the base model — recovering base performance rather than SFT
   performance.

4. **Lesson**: For GRPO → eval to chain correctly, the GRPO training script
   should either (a) push a fully-merged model, or (b) record the SFT model as
   its base in adapter_config. The next iteration would fix this and re-run GRPO
   with a longer SFT (full 800 steps) so the starting rewards are non-zero.

### 3.5 Eval Output

The eval script produces:
1. **Console table** — metrics side-by-side
2. **JSON results file** — `eval_results.json` for programmatic use
3. **Per-sample predictions** — for error analysis

---

## 4. File Layout

```
Fine-Tune/
├── PLAN.md                        ← this file
├── train_medmcqa_sft.py           ← Phase 1: SFT training script  (DONE ✅)
├── train_medmcqa_grpo.py          ← Phase 2: GRPO training script (DONE ✅)
├── eval_medmcqa.py                ← Phase 3: evaluation script    (DONE ✅)
└── eval_results.json              ← Phase 3: output               (generated)
```

---

## 5. Execution Order

```bash
# Step 1: Validate dataset format (~30 sec, ~$0.01)
#         → catch format issues before GPU spend

# Step 2: Submit SFT job to HF Jobs
#         → ~20 min on t4-small, ~$0.25
#         → produces: wei25/qwen3-0.6b-medmcqa-sft

# Step 3: Submit GRPO job to HF Jobs (depends on Step 2)
#         → ~40 min on t4-small, ~$0.50
#         → produces: wei25/qwen3-0.6b-medmcqa-grpo

# Step 4: Run evaluation on all 3 models
#         → ~10 min on t4-small, ~$0.15
#         → produces: eval_results.json + console report

# Total estimated cost: ~$0.90
# Total estimated time: ~70 min (sequential)
```

---

## 6. Key Equations Summary (for Portfolio)

### SFT Loss
```
L_SFT = -(1/T) * sum_{t=1}^{T} log pi_theta( y_t | y_{<t}, x )
```

### LoRA Update
```
W = W_0 + (alpha / r) * B @ A
```

### GRPO Group-Relative Advantage
```
A_hat_i = ( r_i - mean(r) ) / std(r)
```

### GRPO Clipped Surrogate Objective
```
L_GRPO = -(1/sum|o_i|) * sum_i sum_t min(
    rho_{i,t} * A_hat_i,
    clip(rho_{i,t}, 1-eps, 1+eps) * A_hat_i
)
```

### Rule-Based Reward (no reward model needed)
```
R = R_json(0/1) + R_schema(0/1) + R_accuracy(0/1)    range: [0, 3]
```

### Evaluation Metrics
```
accuracy          = correct / total
valid_json_rate   = json_parseable / total
schema_pass_rate  = schema_valid / total
accuracy_on_schema = correct_and_schema_valid / schema_valid
```
