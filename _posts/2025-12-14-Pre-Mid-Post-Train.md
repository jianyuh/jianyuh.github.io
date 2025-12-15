---
layout: post
title: "Interplay of Training Stages"
date: 2025-12-14
categories: [Training]
tags: [Training]
---

Reading the following paper:
- [On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models](https://www.arxiv.org/pdf/2512.07783)


Does RL merely refine capabilities acquired during pre-training (the "refiner" view), or can it genuinely extend reasoning boundaries (the "capability" view)?

Prior analyses suffered from uncontrolled variables in opaque pre-training corpora. By utilizing a **fully controlled synthetic environment** (based on GSM-Infinite), these competing views are not mutually exclusive but rather regime-dependent. The paper provides a recipe for when RL works: it requires specific "headroom" in pre-training and data calibrated to the model's "edge of competence".

## Experimental Framework: The "GSM-Infinite" Control
To isolate causal factors, the authors move away from standard organic corpora to a controllable synthetic setup:
*   **Data Generation:** Problems are generated via Directed Acyclic Graphs (DAGs) defining dependency structures, rendered into natural language templates (e.g., "zoo," "school").
*   **Metrics:** They employ **Process-Verified Evaluation**. Instead of checking only the final answer (which is susceptible to false positives), they parse the reasoning trace into a predicted graph ($\hat{G}$) and verify it against the gold graph ($G$) step-by-step.
*   **Model:** Experiments utilize 100M parameter Qwen2.5-style decoder-only models, pre-trained on 10B tokens (Chinchilla optimal scaling).

## Key Technical Findings

### A. Extrapolative Generalization (Depth)
*Query: Can RL solve problems requiring more steps (operations) than seen in Pre-Training?*

The study reconciles the field's conflicting views by introducing the concept of the **"Edge of Competence."**
*   **Saturated State:** On In-Distribution (ID) tasks (op=2-10) where the base model is already capable, RL offers no `pass@128` gains, only sharpening `pass@1` (supporting the "refiner" view).
*   **The Sweet Spot:** RL yields genuine capability gains (up to +42% `pass@128`) on Out-Of-Distribution (OOD) tasks (op=11-14) *only* if the RL data targets this boundary—tasks difficult but not impossible for the base model.
*   **Failure Modes:** RL fails if the tasks are too far OOD (op=15-20) or if the RL data is miscalibrated (too easy or too hard).

### B. Contextual Generalization (Breadth)
*Query: Can RL transfer reasoning logic to unseen surface contexts (templates)?*

The authors test "Long-Tail" exposure by varying the ratio of specific contexts (e.g., Context B) in pre-training.
*   **The "Seed" Hypothesis:** RL cannot synthesize understanding from a void. If the model sees 0% or 0.1% of Context B in pre-training, RL fails to generalize to it, even if the underlying logic is identical to the known Context A.
*   **Minimal Exposure Sufficiency:** A "seed" exposure of $\ge 1\%$ in pre-training is sufficient. Once this representation exists, RL can robustly amplify it, enabling transfer even to complex OOD tasks within that context.

### C. The Role of Mid-Training
*Query: How should compute be allocated between Mid-Training (SFT/CPT) and RL?*

Using a fixed compute budget comparison (normalizing RL steps to token equivalents), the study finds:
*   **Compute Equivalence:** They derive $T_{RL} \approx 5.3 \cdot N \cdot r \cdot L_{total}$ to compare RL rollouts with supervised tokens.
*   **Allocation Strategy:**
    *   **Light RL / Heavy Mid-Training:** Optimal for **OOD-edge** reliability (pass@1). Mid-training installs the necessary priors that prevent the model from collapsing during exploration.
    *   **Heavy RL / Light Mid-Training:** Optimal for **OOD-hard** exploration (pass@128). Once priors are established, heavy exploration is required to solve the hardest tasks.
*   **Takeaway:** Mid-training acts as a "distributional bridge," aligning representations to the task format so RL can focus on reasoning scaling rather than format adaptation.

### D. Process Supervision vs. Reward Hacking
*   **Reward Hacking:** Outcome-based rewards ($R_{out}$) encourage short-cuts.
*   **Mitigation:** The authors implement a composite reward $R = \alpha R_{out} + (1-\alpha)R_{pv}$ (process verification).
*   **Result:** A strict reward regime—where outcome rewards are *conditional* on correct processes—yields the highest performance (+5.2% `pass@1` on OOD-hard tasks) and significantly reduces structural errors (e.g., missing dependency nodes).

## Insights
Data recipes:
1.  **Iterative RL Design:** Do not train on the hardest data immediately. Filter RL datasets for the "edge of competence"—tasks where the model fails `pass@1` but succeeds `pass@k`.
2.  **Pre-Training Composition:** Ensure "long-tail" domains are represented at least sparsely ($\approx 1\%$). You cannot RL your way out of a knowledge gap; you can only RL your way out of a *competence* gap.
3.  **Compute Allocation:** If your goal is broad reliability, front-load compute into Mid-Training. If the goal is "Eureka" moments on hard problems, reserve the budget for massive RL exploration.

{% comment %}
## Summary
To solidify the interplay between these stages, consider the analogy of **Language Immersion**:

*   **Pre-Training is Vocabulary & Grammar Study:** You must learn the basics. If you never study the word for "Train" (0% exposure), no amount of conversation practice later will help you guess it. However, you only need to see it a few times (1% exposure) to recognize it later.
*   **Mid-Training is a Phrasebook:** It bridges the gap between raw vocabulary and actual usage. It teaches you the *structure* of a conversation, ensuring you don't freeze up when spoken to.
*   **RL is Immersion/Debate:** This is where you test your limits.
    *   *Edge of Competence:* If you debate a toddler (Too Easy/ID), you learn nothing. If you debate a philosophy professor immediately (Too Hard/OOD-hard), you learn nothing. You improve fastest by debating peers slightly better than you (OOD-edge).
    *   *Capability:* RL turns your latent vocabulary into fluent, complex argumentation. It doesn't teach you new words, but it teaches you how to string them together in ways you never explicitly studied.
{% endcomment %}
