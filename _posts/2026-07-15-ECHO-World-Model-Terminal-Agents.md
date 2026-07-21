---
layout: post
title: "ECHO: Training Terminal Agents via World-Model Objectives"
date: 2026-07-15
categories: [RL, Agents]
tags: [ECHO, GRPO, RL, World Model, Terminal Agents, TerminalBench, Self-Improvement]
---

Reading notes on:
- [ECHO: Terminal Agents Learn World Models for Free](https://x.com/DimitrisPapail/status/2056368948870811746)

## The Problem: Wasted Supervision in Agent RL

In standard RL for CLI agents — e.g. Group Relative Policy Optimization (GRPO) — the agent's interaction with the environment is heavily underutilized. Terminal tasks have sparse, delayed, binary rewards, so small models see low pass rates and get almost no learning signal from failed trajectories.

Crucially, GRPO applies loss **only** on the agent's action tokens and **masks out** the environment's response (stdout, stderr, file listings, stack traces). This is wasteful: the model is *already* conditioned on those observation tokens, attends to them, and computes distributions over them during the forward pass. Ignoring them discards ground-truth information about how the agent's actions changed the system state. The trainer/generator inefficiencies this creates echo the throughput-matching problems in [RL Systems Mind the Gap]({% post_url 2026-06-19-RL-Mind-The-Gap %}) and the wasted-rollout critique in [Experience Replay for LLM RL]({% post_url 2026-04-16-RL-Experience-Replay %}).

![GRPO masks observation tokens; ECHO adds a cross-entropy loss on them for free](/assets/images/echo_loss_mask.svg)

## The ECHO Objective

ECHO (**E**nvironment-aware **C**ommand-**H**istory **O**bjective) trains jointly on both sides of the interaction — the agent's commands *and* the terminal's responses — by adding a length-normalized cross-entropy loss on environment-observation tokens alongside standard GRPO:

$$L_{\text{ECHO}} = L_{\text{GRPO}}(\text{Actions}) + \lambda \cdot L_{\text{env}}(\text{Observations})$$

- **Actions:** positions where the agent writes commands.
- **Observations:** raw terminal output tokens (tracebacks, filenames) — deliberately excluding easily-memorized harness warnings.
- **$\lambda$:** a balance coefficient. Too small and the env loss fails to shape representations; too large and the policy degenerates into emitting actions that yield *predictable* terminal output rather than task progress.

## The "Free" Compute

A major advantage: ECHO learns a world model essentially **for free**. The added cross-entropy loss needs no extra rollouts, no teacher, no extra forward passes. The backward cost is practically identical to GRPO — the expensive attention/MLP matmuls run over the same sequence length regardless of the loss mask. ECHO simply changes which already-computed logits are gathered for each loss term.

## Results

ECHO is fully on-policy, learning from the current model's own transitions — better policies induce better feedback, and better feedback prediction improves action priors.
- **Performance:** TerminalBench-2.0 pass@1 nearly doubled — Qwen3-8B (2.7 → 5.2) and Qwen3-14B (5.2 → 10.8).
- **Efficiency:** reached GRPO's 500-step performance 280 steps faster — a **2.3× training speedup**.

## Three Deeper Insights

**1. Implicit world modeling via token compression.** Forcing the agent to predict exact terminal output makes it track file creations, kernel state, and directory structure in its hidden states. On held-out trajectories from a stronger 32B teacher, ECHO models show a sharp drop in environment-token cross-entropy vs. GRPO — informationally, the ECHO policy becomes a better *compressor* of terminal dynamics, i.e. it internalizes a mental model of the system.

**2. Redefining the value of expert SFT.** Starting from a base model with no expert behavior cloning, ECHO recovers up to **104%** of the gains usually provided by expert SFT on some benchmarks (50% on the harder TerminalBench-2.0). The implication: much of what expert SFT buys is an *interaction prior* (how to read errors, follow tracebacks, inspect files) rather than a *strategy prior* — and ECHO learns that interaction prior directly from the environment, no demonstrations needed. This reframes the SFT-then-RL pipeline behind agents like [SWE-1.7]({% post_url 2026-07-10-SWE-1.7 %}), and complements the local-support view in [Revisiting On-Policy Distillation]({% post_url 2026-04-19-On-Policy-Distillation %}).

**3. Verifier-free self-improvement.** The radical finding: drop the GRPO term and the verifier entirely, leaving only

$$L = L_{\text{env}}(\text{Observations})$$

The model acts, observes, and updates purely by predicting the consequences of its own actions. With *no* reward labeling good vs. bad actions, task performance still improved out-of-distribution (**+10.0 points** on Python-heavy PyTerm). Once a policy can explore at all, learning to predict dense, action-linked terminal feedback (like tracebacks) is enough to reshape its action priors — a striking data point for the self-improving-harness thesis in [Harness Engineering for Self-Improvement]({% post_url 2026-07-09-Harness-Engineering-Self-Improvement %}).
