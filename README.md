# LongGraph_agents

## Project Highlights

- Fine-tuned Qwen3-0.6B on 1,000 MedMCQA questions to emit structured JSON predictions, improving accuracy from 27.5% → 34.0% (+6.5 pp), then applied GRPO with rule-based rewards and debugged reward-collapse/LoRA-chaining issues (HF Jobs, <$1).
- Implemented an educational notebook explaining PPO vs GRPO from first principles and demonstrated GRPO by training a small policy to sort 3 numbers to 100% accuracy using group-normalized rewards (no critic).
- Built a LongGraph agent structure to orchestrate multi-model, multi-step tasks with clear state, routing, and tool-interaction scaffolding.
