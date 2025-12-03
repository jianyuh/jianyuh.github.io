---
layout: post
title: "First-Order Approximation for Stable LLM-RL Training"
date: 2025-12-02
categories: [RL]
tags: [RL]
---

Read on DeepSeek-V3.2:
- [Stabilizing Reinforcement Learning with LLMs: Formulation and Practices](https://www.arxiv.org/pdf/2512.01374).

Instability of Reinforcement Learning (RL) in LLMs, specifically the mismatch between **sequence-level rewards** (outcome-based) and **token-level optimization** (policy gradient). The authors propose a formulation where the token-level objective acts as a **first-order approximation** of the true sequence-level reward.

**Key Takeaway:** The validity of this approximation hinges on minimizing two gaps: (1) **Training–Inference Discrepancy** (numerical/infra differences) and (2) **Policy Staleness** (lag between rollout and update). This framework explains why techniques like Importance Sampling (IS), Clipping, and Routing Replay (for MoEs) are mathematically necessary for stability.

---

### 2. Theoretical Formulation: The First-Order Approximation
The paper argues that we rarely optimize the true expected sequence-level reward directly due to high variance. Instead, we use a surrogate token-level objective (similar to REINFORCE).

**The Approximation:**
The gradient of the surrogate token-level objective ($J_{token}$) equals the gradient of the true sequence-level objective ($J_{seq}$) **if and only if** the target policy $\pi_\theta$ is identical to the rollout policy $\mu_{\theta_{old}}$.

**The Decomposition of Instability:**
To maintain this validity, the authors decompose the Importance Sampling (IS) weight for a token $y_t$ into two components:

$$ \frac{\pi_\theta(y_t | \dots)}{\mu_{\theta_{old}}(y_t | \dots)} = \underbrace{\frac{\pi_{\theta_{old}}(y_t | \dots)}{\mu_{\theta_{old}}(y_t | \dots)}}_{\text{Training-Inference Discrepancy}} \times \underbrace{\frac{\pi_\theta(y_t | \dots)}{\pi_{\theta_{old}}(y_t | \dots)}}_{\text{Policy Staleness}} $$

1.  **Training-Inference Discrepancy:** Differences caused by engine mismatches (e.g., vLLM vs. Megatron) or non-deterministic kernels.
2.  **Policy Staleness:** Caused by off-policy updates (splitting large batches into mini-batches) or async training.

**Implication:** If either term deviates significantly from 1, the first-order approximation breaks, and the optimization direction no longer maximizes the sequence reward.

---

### 3. The Mixture-of-Experts (MoE) Challenge
MoE models introduce a third variable: **Expert Routing**. The dynamic selection of experts ($e_t$) exacerbates both discrepancies:
*   **Inconsistent Routing:** The same input might route to different experts in training vs. inference engines.
*   **Routing Shift:** As $\theta$ updates, the router selects different experts, causing massive shifts in the active parameters (policy staleness).

#### Solution: Routing Replay
To stabilize MoEs, we must fix the experts during optimization to match the rollout. The paper compares two approaches:
*   **Vanilla Routing Replay (R2):** Replays experts selected by the *training* rollout policy ($\pi_{\theta_{old}}$).
*   **Rollout Routing Replay (R3):** Replays experts selected by the *inference* rollout policy ($\mu_{\theta_{old}}$).

**Trade-off:** Routing Replay restores the validity of the first-order approximation but introduces bias by altering the target policy (forcing it to use "old" experts).

---

### 4. Practical Recipes (Empirical Results)
The authors developed "MiniRL," a minimalist baseline using token-level IS weights, PPO-style clipping, and group-normalized advantages.

#### A. On-Policy Training (Global Batch = Mini Batch)
*   **Best Recipe:** Basic Policy Gradient + **IS Correction** for training-inference discrepancy.
*   **Avoid:**
    *   **Length Normalization:** Common in GRPO/CISPO, but the authors find it invalidates the first-order approximation and degrades performance.
    *   **Routing Replay (R3):** In strict on-policy settings, R3 biases the objective without providing necessary stability benefits, leading to worse results.

#### B. Off-Policy Training (Global Batch > Mini Batch)
When splitting large batches for multiple updates (accelerating convergence), stability becomes the bottleneck.
*   **Requirement:** Both **Clipping** and **Routing Replay** are essential. Removing either causes collapse.
*   **R2 vs. R3 Selection:**
    *   **Low Off-Policiness ($2\times$):** **R2 is superior.** It preserves the original target policy in the first mini-batch.
    *   **High Off-Policiness ($4\times, 8\times$):** **R3 is superior.** Under high staleness, the benefit of matching the inference routing (R3) outweighs the bias it introduces.

#### C. Cold-Start Initialization
*   **Finding:** Once the RL process is stabilized (using the recipes above), the specific cold-start model (e.g., distilled from Qwen vs. DeepSeek vs. GPT) matters less. Prolonged training allows different initializations to converge to comparable final performance.

---

### 5. Technical Nuances & Critique
*   **Token-Level IS Weights are Critical:** Unlike standard PPO which often ignores the denominator difference between inference/training engines, this paper explicitly corrects it ($P_{train} / P_{inference}$). Omitting this leads to rapid entropy collapse.
*   **Critique of Length Normalization:** The paper challenges the standard practice (used in GRPO) of dividing rewards by response length. They argue this biases the objective away from the true expected reward.
*   **FP8 Stress Test:** Experiments were conducted with FP8 inference and BF16 training to intentionally maximize the training-inference discrepancy, validating the robustness of the IS correction.
