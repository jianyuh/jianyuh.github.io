---
layout: post
title: "Revisiting On-Policy Distillation: Failure Modes and Local Support Matching"
date: 2026-04-19
categories: [LLM]
tags: [LLM, Distillation, OPD, Post-Training, Reasoning]
---

Paper: [Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes](https://arxiv.org/pdf/2603.25562).

The [Thinking Machines Lab blog]({% post_url 2025-11-21-TML %}) covered OPD's per-token reverse KL objective and its dramatic efficiency gains (50–100× over RL). This paper takes a closer look at that token-level estimator and shows it is surprisingly brittle in long-horizon settings—then proposes a principled fix.

On-policy distillation (OPD) is an increasingly popular technique for post-training LLMs on reasoning and agentic tasks. Unlike offline distillation (which uses fixed teacher traces), OPD evaluates a stronger teacher's feedback directly on student-generated rollouts. This is crucial for long-horizon tasks because students quickly encounter states (prefixes) that are rare or entirely absent in static teacher datasets.

However, the standard implementation of OPD—evaluating the log-ratio on a single **sampled token**—is notoriously brittle. This paper dissects exactly *why* this token-level approximation fails in practice and proposes a mathematically grounded, easy-to-implement fix: **Teacher Top-K Local Support Matching**.

---

## 1. Theoretical Foundations: The Bias-Variance Tradeoff

The core issue in OPD is managing the estimator trade-off between sequence-level coupling and token-level variance.

### The Sequence-Level Estimator

The ideal sequence-level objective for a prompt $x$ is the reverse-KL divergence between the student $\pi_\theta$ and the teacher $q$:

$$J_{OPD}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} [D_{KL}(\pi_\theta(\cdot \mid x) \parallel q(\cdot \mid x))]$$

Using the score-function identity, the gradient over an autoregressive factorization can be written as:

$$\nabla_\theta J_{OPD}(\theta) = \mathbb{E}_{x, y \sim \pi_\theta(\cdot \mid x)} \left[ \left(\sum_{t'=1}^T r_{t'}\right) \sum_{t=1}^T g_t \right]$$

where the immediate reward is $r_t = \log \pi_\theta(y_t \mid c_t) - \log q(y_t \mid c_t)$, and the score function is $g_t = \nabla_\theta \log \pi_\theta(y_t \mid c_t)$.

Because $r_{t'}$ only depends on the prefix before step $t$, causality implies $\mathbb{E}[r_{t'} g_t] = 0$ for $t' < t$. This yields the **causal return-to-go sequence-level estimator**:

$$\hat{g}_{seq} = \sum_{t=1}^T \left( \sum_{t'=t}^T r_{t'} \right) g_t$$

**The Variance Problem:** While unbiased, $\hat{g}_{seq}$ couples each token update to all future rewards. Under bounded rewards and gradients, the worst-case variance scales as $\mathcal{O}(T^4)$. In long-horizon tasks (where sequence length $T$ is large), this quartic scaling leads to unstable optimization and diverging policies.

### The Token-Level Estimator

To fix the variance explosion, standard LLM pipelines drop future-reward coupling entirely, resulting in the **token-level OPD estimator**:

$$\hat{g}_{tok} = \sum_{t=1}^T r_t g_t$$

**The Bias Problem:** This creates a heavily biased estimator. The worst-case variance bound tightens drastically to $\mathcal{O}(T^2)$, making it cheap and stable to compute, but it reduces trajectory-level quality matching to an isolated, point-estimate comparison.

---

## 2. Empirical Failure Modes of Sampled-Token OPD

While mathematically attractive for its low variance, the $\hat{g}_{tok}$ formulation suffers from three severe practical failure modes:

1.  **Highly Imbalanced Signal:** The token-level signal is driven by $\log q(y_t \mid c_t) - \log \pi_\theta(y_t \mid c_t)$. Because the teacher distribution is often sharp, most tokens sampled by the student receive negative rewards. The positive learning signal is concentrated on a tiny subset of tokens, making the optimization sensitive to idiosyncratic, short-term teacher preferences (like hesitation markers or fillers).
2.  **Unreliable Teacher Guidance on Student Prefixes:** When a student rollout drifts out-of-distribution (OOD) for the teacher, the teacher's next-token probabilities cease to be a valid proxy for whole-trajectory quality. For example, if the student enters a repetitive loop (e.g., generating "Wait, Wait, Wait"), the teacher might still assign high probability to generating another "Wait", actively rewarding a degenerate trajectory.
3.  **Tokenizer and Special-Token Mismatch:** If the teacher and student use different tokenizers, semantically identical outputs can trigger spurious penalties. For example, if a student generates `<think>` as `<, think, >` but the teacher expects `<th, ink, >`, the teacher assigns near-zero probability to `<`, heavily penalizing the student for a mere formatting artifact.

---

## 3. The Fix: Teacher Top-K Local Support Matching

To solve these failure modes, the key idea is to move away from a one-token point estimate without regressing to the high-variance sequence-level estimator. The goal is to perform a **distribution-level comparison within a stable local region**.

### The Derivation

A full-vocabulary reverse-KL at prefix $c_t$ is:

$$L_{full}(c_t) = \sum_{v \in V} \pi_\theta(v \mid c_t) \log \frac{\pi_\theta(v \mid c_t)}{q(v \mid c_t)}$$

Sampled-token OPD is just a one-sample Monte Carlo approximation of this equation.

Instead of the full vocabulary or a single sample, the proposed method truncates the comparison to the **teacher's top-K local support set**:

$$S(c_{i,t}) = \text{TopK}_q(c_{i,t})$$

To make this mathematically sound, the probabilities of both the teacher and student must be **renormalized** within this subset:

$$\hat{\pi}_\theta(v \mid c_{i,t}) = \frac{\pi_\theta(v \mid c_{i,t})}{\sum_{u \in S} \pi_\theta(u \mid c_{i,t})}, \quad \hat{q}(v \mid c_{i,t}) = \frac{q(v \mid c_{i,t})}{\sum_{u \in S} q(u \mid c_{i,t})}$$

The final training objective becomes the truncated reverse-KL averaged over rollout positions:

$$L_{LSM} = \mathbb{E}_{x, \{o_i\}} \left[ \frac{1}{\sum \lvert o_i \rvert} \sum_{i,t} \sum_{v \in S} \hat{\pi}_\theta(v \mid c_{i,t}) \log \frac{\hat{\pi}_\theta(v \mid c_{i,t})}{\hat{q}(v \mid c_{i,t})} \right]$$

### Practical Stabilization Tricks

Two crucial engineering components to make this work:

1.  **Top-p Rollout Sampling:** The student must generate rollouts using top-p sampling (e.g., $p=0.9$) to prevent trajectories from drifting into extremely low-probability, OOD regions where the teacher's signal becomes meaningless.
2.  **Special-Token Masking:** Masking out special tokens (like EOS markers) during the loss computation prevents the model from suffering false negatives due to tokenizer incompatibility.

---

## 4. Empirical Insights & Results

This was validated on Qwen2.5-7B-Instruct (student) using OpenThinker3-7B (teacher) across math and ALFWorld (agentic) tasks.

*   **Performance:** Local Support Matching consistently outperformed baseline sampled-token OPD. On math reasoning (Math500), it raised the unmasked baseline from 80.0 to 82.0.
*   **Stability:** The proposed method yielded significantly smaller gradient norms, lower clipping-boundary fractions, and a smaller teacher-student log-probability gap compared to baseline OPD.
*   **Ablations (The "Why"):** Simply applying a Top-K loss isn't enough. Top-p rollout sampling is absolutely required to keep the student in a space where the teacher can accurately guide it. Furthermore, omitting the renormalization step leads to rapid optimization collapse.

---

## 5. Takeaways

1.  **Stop optimizing point-estimates:** A single sampled token carries too much noise and too little structural information. Redistributing updates across a Top-K support set transforms a noisy penalization regime into a balanced, distributional learning signal.
2.  **Teacher matching ≠ Task Success:** A profound limitation of OPD is *teacher distribution sharpness*. On OOD prefixes, teachers hallucinate confident probabilities for garbage tokens (e.g., endless repetition). OPD assumes teacher probability correlates with task success, but this assumption breaks down outside the teacher's comfort zone.
3.  **The Future is Hybrid:** While local support matching patches the local objective, solving the global problem will likely require hybridizing OPD with outcome-based RL (verifiable rewards) or tighter rollout drift control (like EMA anchors or logit-fusion) to guarantee that locally sound tokens actually lead to a correct final answer.
