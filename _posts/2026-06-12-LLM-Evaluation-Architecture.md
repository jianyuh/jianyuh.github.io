---
layout: post
title: "Metrics and Benchmarks Across LLM Training Stage"
date: 2026-06-12
categories: [LLM, Evaluation, Systems]
tags: [Evaluation, Benchmarks, Pre-training, Mid-training, Post-training, RLHF, MoE, Monitoring]
---

Evaluating LLMs is not a single test you run at the end. It is a layered system that runs from the very first training step all the way to the production release gate. To build competitive models, teams need an evaluation architecture that operates across three distinct operational layers:

1. **Training Health Monitoring (step / hour level):** real-time signals to catch anomalies, data-pipeline issues, or hardware degradation early.
2. **Capability Regression Evaluation (checkpoint level):** scheduled validation to measure capability growth and flag catastrophic forgetting.
3. **Release Gating Evaluation (pre-release level):** the final comprehensive evaluation that decides whether a model is production-ready.

Each of these layers shows up differently in the three major phases of model development: **[Pre-training, Mid-training, and Post-training]({% post_url 2025-12-14-Pre-Mid-Post-Train %})**. Below is the full breakdown of the monitoring metrics and evaluation benchmarks that matter at each phase.

---

## Phase 1: Pre-training Evaluation Architecture

The primary focus of pre-training is ensuring **training stability**, analyzing **data ingestion quality**, optimizing **hardware efficiency**, and tracking the **emergence of foundational capabilities**.

### 1.1 Real-Time Training Health Monitoring

These metrics are tracked continuously at the step or hour level to keep the training run healthy:

| Category | Specific Monitoring Metrics |
| --- | --- |
| **Loss & Optimization** | Train Loss; Validation Loss (by bucket / domain); Perplexity (PPL); Loss Spikes; Learning Rate (LR) schedule; Gradient Norms; Update-to-Weight Ratio |
| **Data & Contamination** | Data Deduplication Rates; Toxicity Ratios; Language & Domain Distribution; Benchmark Overlap (contamination checking); N-gram Overlap |
| **System Efficiency** | Tokens/sec/GPU (Throughput); Model FLOPs Utilization (MFU); GPU Utilization & Memory Footprint; Fault Recovery / Checkpointing Times |
| **MoE Specifics** *(if applicable)* | Expert Load Balance / Routing Distribution; Router Entropy; Dropped Token Rates |

A few of these deserve emphasis. The **Update-to-Weight Ratio** is one of the most underrated early-warning signals — if updates are too large relative to weights, the run is on the edge of divergence. For [MoE models]({% post_url 2026-03-11-MoE-Megatron %}), **Router Entropy** and **Dropped Token Rates** tell you whether your experts are actually specializing or collapsing into a degenerate routing pattern.

### 1.2 Checkpoint Capability Evaluation

As the model digests more tokens, it is periodically evaluated against a core benchmark suite to track the emergence of capabilities:

- **Language Modeling:** held-out validation sets across diverse domains.
- **General Knowledge & Reasoning:**
  - **MMLU** & **MMLU-Pro** (Massive Multitask Language Understanding)
  - **ARC** (AI2 Reasoning Challenge — Easy & Challenge splits)
  - **HellaSwag** (commonsense reasoning)
  - **OpenBookQA** (question answering)
- **Math & Code:**
  - **GSM8K** (grade-school math)
  - **MATH** (harder competition problems)
  - **HumanEval** (Python coding tasks)
  - **MBPP** (Mostly Basic Python Problems)
- **Long Context & Safety Baseline:**
  - **Needle-in-a-Haystack (NIAH)** (retrieval accuracy across context lengths)
  - **RULER** (effective context-window evaluation)
  - **RealToxicityPrompts** (baseline generation safety)

> **Pre-training Minimum Evaluation Suite (the Baseline Checkpoint):**
> Total Validation Loss + Bucket Losses (Language, Code, Math, Web, Books, Papers) + an MMLU-Pro subset + GSM8K + HumanEval + Toxicity/Memorization Canaries.

The bucket losses are what let you debug a data problem instead of just observing one. A flat total loss can hide a quietly regressing code bucket — only the per-domain breakdown surfaces it.

---

## Phase 2: Mid-training (Domain Adaptation & Continuous Pre-training)
<a id="catastrophic-forgetting"></a>

Mid-training builds specialized capabilities — enhanced reasoning, additional languages, vertical domains — through continuous pre-training on a base model. The dominant risk in this phase is **catastrophic forgetting**: gaining new skills while silently losing old ones.

### 2.1 Mid-Training Monitoring Metrics

- **Target vs. General Loss:** track domain-specific validation loss directly against general validation loss (e.g., MMLU / HumanEval subsets) so regression shows up the moment it begins.
- **Data Mixtures:** monitor and tune the **target-to-general replay ratio** and the **synthetic-to-human data ratio**.
- **Synthetic Data Quality:** if you use synthetic pipelines, watch **format error rates**, **repetition rates**, and **answer verifiability** (via programmatic checkers or separate judge models).
- **Reasoning & Code Specifics:** track **final-answer accuracy**, **Chain-of-Thought (CoT) token length**, **invalid reasoning-path ratios**, and code **compile rates**.

### 2.2 The Fixed Retention Suite

To safely gate checkpoints during mid-training, evaluate against a rigid suite that balances new expertise against baseline retention:

```
                  ┌────────────────────────────────────────┐
                  │        MID-TRAINING GATEWAY SUITE       │
                  └───────────────────┬────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
┌─────────────────┐          ┌─────────────────┐          ┌──────────────────┐
│  TARGET DOMAIN  │          │ GENERAL BASELINE│          │ ENHANCEMENT CHECKS│
├─────────────────┤          ├─────────────────┤          ├──────────────────┤
│ • FinanceBench  │          │ • C-Eval        │          │ • MATH-500       │
│ • MedQA         │          │ • MMLU          │          │ • LiveCodeBench  │
│ • LegalBench    │          │ • CMMLU         │          │ • RULER /        │
│                 │          │                 │          │   LongBench      │
└─────────────────┘          └─────────────────┘          └──────────────────┘
```

The key discipline here: the **General Baseline** column must stay *frozen* across runs. If you keep changing it, you can no longer tell whether a regression came from your model or from your eval.

---

## Phase 3: Post-training (SFT & Alignment)

Post-training converts the raw capabilities of a base model into a helpful, obedient, and safe user-facing assistant — through Supervised Fine-Tuning (SFT) and [Reinforcement Learning (RLHF / DPO)]({% post_url 2025-11-23-RL-Scaling %}).

### 3.1 SFT Stage Monitoring Metrics

- **Instruction Following:** adherence pass rates for hard constraints (negative constraints, specific word lengths, language enforcement).
- **Format Capabilities:** **JSON validity rates**, **schema validation compliance**, and Markdown/XML formatting accuracy.
- **Factuality & Style:** baseline hallucination rates, unsupported-claim tracking, **verbosity drift** (unwarranted lengthening of responses), and robotic/repetitive structural templates.

### 3.2 RLHF / DPO Stage Monitoring Metrics

- **Preference Optimization:** policy and reference model log-probabilities, **preference loss**, and the exact margin between **chosen vs. rejected** responses.
- **Reward Hacking Signals:** watch for exponential reward-model score growth against static proxy metrics — over-apologetic behavior, fluff, and formatting hacks that game the reward model without improving the response.
- **RL Stability:** monitor **KL divergence** relative to the reference policy to prevent policy collapse or severe degradation of baseline capabilities.

The single most important post-training instinct: **reward going up is not the goal — reward going up *while proxy metrics hold* is.** A reward curve that climbs while your held-out quality metrics flatten or drop is reward hacking, not progress.

### 3.3 Release Gating Benchmarks (the Final Evaluation Layer)

Before production release, the model must pass a matrix of automated, LLM-as-a-judge, and human benchmarks:

| Evaluation Dimension | Industry-Standard Benchmarks | Focus Areas |
| --- | --- | --- |
| **Instruction Compliance** | **IFEval** | Verifiable constraints (formatting, length, punctuation, specific keywords) |
| **Dialog & User Preference** | **MT-Bench**, **Arena-Hard**, **AlpacaEval 2.0**, **Human Pairwise Evaluation** | Conversational fluency, complex multi-turn reasoning, human-preference alignment |
| **Factuality & Truthfulness** | **SimpleQA**, **TruthfulQA** | Short-form fact retrieval accuracy, hallucination reduction, fewer false assertions |
| **Safety & Robustness** | **HarmBench**, **JailbreakBench** | Red-teaming resilience, refusal consistency, safety-alignment robustness |

---

## Strategic Takeaway

Two principles tie the whole architecture together:

- **Automated vs. Human Judges.** Public automated benchmarks are essential for tracking your progress relative to the broader market — but they are highly susceptible to data contamination. Treat a rising public-benchmark score with the same suspicion as a rising reward-model score: verify it isn't leaking into your training set.
- **The "Golden Set."** The final release gate should lean heavily on a *proprietary, continually updated* internal **Golden Set**, evaluated with a combination of high-tier LLM-as-a-judge patterns and rigorous human blind side-by-side (A/B) testing. Public benchmarks tell you where you stand; the Golden Set tells you whether you should ship.

The throughline across all three phases is the same: **measure continuously, gate against a frozen baseline, and never trust a single number you can't decompose.** Loss spikes, forgetting, and reward hacking all share one property — they hide inside aggregate metrics and only reveal themselves when you break the signal apart.
