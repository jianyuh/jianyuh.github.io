---
layout: post
title: "The Mechanics of Reasoning Effort and Inference Scaling"
date: 2026-07-19
categories: [LLM, RL]
tags: [Reasoning Effort, Inference Scaling, RLVR, Thinking Mode, Length Penalty, Test-Time Compute]
---

Reading notes on:
- [Controlling Reasoning Effort in LLMs (How LLMs Learn Low-, Medium-, and High-Effort Reasoning Modes)](https://magazine.sebastianraschka.com/p/controlling-reasoning-effort-in-llms)

As LLMs pivot from plain autoregressive generation to explicit "reasoning," inference-time scaling has taken center stage. Flagship models — OpenAI's GPT-5.6 family through open-weight models like DeepSeek-R1, Qwen3, and [Inkling]({% post_url 2026-07-17-Inkling-975B-MoE %}) — now embed controllable **reasoning effort**. This note synthesizes the mechanisms, training recipes, and math that calibrate computational depth. It pairs naturally with the throughput view in [The Economics of a Token]({% post_url 2026-05-17-token-economics %}) and the training-compute view in [The Architecture of Scaling Laws]({% post_url 2026-06-25-The-Architecture-of-Scaling-Laws %}).

---

## 1. RLVR and the "Illusion" of `<think>` Tokens

State-of-the-art reasoning is trained mostly via **Reinforcement Learning with Verifiable Rewards (RLVR)** — domains where correctness is computationally checkable (math via symbolic solvers, code via compilers/tests). A key insight: the intermediate reasoning trace itself is *not* scored for accuracy — only the final answer is. Self-corrections mid-trace are the famed "Aha" moments.

The `<think>`/`</think>` tags don't grant reasoning ability; they're cosmetic delimiters, incentivized by a formatting reward added to the RLVR objective:

$$R_{\text{total}} = R_{\text{accuracy}} + R_{\text{format}}$$

---

## 2. The On/Off Switch: Thinking Mode Fusion

Early reasoning models couldn't turn *off* their verbose traces. Hybrid architectures like Qwen3 add a post-training SFT phase — **Thinking Mode Fusion** — pairing prompts with two target formats:
- `/think`: `<think>{reasoning}</think>{answer}`
- `/no_think`: `<think></think>{answer}`

At inference, a hard switch at the tokenizer level (`enable_thinking=False`) injects an empty `<think></think>` block into the assistant prefill, forcing the model to skip reasoning and answer immediately.

---

## 3. Calibrating Depth: The Math of Reasoning Budgets

Beyond binary switches, models like GPT-5.6 and Inkling let you dial effort (Low/Medium/High, or a continuous scalar) — inference-time compute scaling via the token budget. The most sophisticated approach modifies the RLVR objective with a **dynamic length penalty** conditioned on a requested effort $e \in [0.0, 1.0]$ passed in the system prompt:

$$R(e) = R_{\text{task}} - \lambda(e)\, N_{\text{tokens}}$$

- $R_{\text{task}}$: baseline accuracy reward.
- $N_{\text{tokens}}$: reasoning-trace length.
- $\lambda(e)$: penalty coefficient mapped from $e$. Low $e$ → large $\lambda$ (heavy per-token cost → short, efficient traces); high $e$ → small $\lambda$ (large budget to explore deeper reasoning trees).

---

## 4. Recipes in SOTA Open-Weight Models

- **DeepSeek-V4 (distilled specialists):** on-policy distillation from a pool of specialized teachers; "Non-think", "Think High", "Think Max" modes, each with tuned context and length penalty. "Think Max" is triggered by a strict system prompt. (See [DeepSeek-V4 Architecture & Training]({% post_url 2026-04-26-DeepSeek-V4-Arch-Train %}).)
- **Nemotron 3 Ultra (truncation robustness):** during SFT, traces are randomly truncated at token limits, `</think>` is masked from the loss, and the original final answer is appended — teaching graceful close-out under hard budgets. (See [Deep Dive into Nemotron 3 Ultra]({% post_url 2026-06-06-Nemotron-3-Ultra %}).)
- **Kimi K2.5 (the toggle method):** alternate two RL phases; in the "budgeted phase," correct solutions must fit a strict token budget (dynamic percentiles of successful rollouts), enforced *only after* the model crosses an accuracy threshold — preventing premature reasoning collapse.
- **GLM-5 (granular turn-level thinking):** multi-turn controls via SFT — "interleaved thinking" (reasoning before tool calls) and "preserved thinking" (keeping earlier reasoning in history). (See [GLM-5.2]({% post_url 2026-06-21-GLM-5.2 %}).)

---

![Training-scaling and inference-effort curves overlap: small@High ≈ mid@Low, with diminishing returns at max effort](/assets/images/reasoning_effort_frontier.svg)

## 5. The Training vs. Inference Scaling Frontier

A profound observation: model scaling (training compute) and effort scaling (inference compute) **overlap**. A smaller model (e.g. GPT-5.6 Luna) at High effort can match a larger model (Terra) at Low effort. But the highest limits hit diminishing returns — GPT-5.6 Sol at max effort spends enormous token counts for sub-proportional accuracy gains, the same efficiency ceiling explored in [Efficiency in LLMs]({% post_url 2026-06-26-Efficiency-in-LLMs-Fast-Inference-Memory-Bandwidth %}).

**Implication:** today's paradigm relies on user-dictated system prompts to set effort. Going forward, the optimal accuracy-per-dollar intersection will be **routed dynamically** — agentic wrappers computing the best model-size / token-penalty ($\lambda(e)$) tradeoff from task complexity, failure cost, and token limits.
