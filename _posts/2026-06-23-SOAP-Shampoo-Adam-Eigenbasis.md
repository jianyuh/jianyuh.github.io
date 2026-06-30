---
layout: post
title: "SOAP: Bridging First- and Second-Order Optimization via Shampoo's Eigenbasis"
date: 2026-06-23
categories: [LLM, Optimization, Training]
tags: [SOAP, Shampoo, Adam, AdamW, Adafactor, Second-Order Optimization, Preconditioner, Critical Batch Size]
---

Reading notes on [**SOAP: Improving and Stabilizing Shampoo using Adam**](https://arxiv.org/pdf/2409.11321) (ShampoO with Adam in the Preconditioner's eigenbasis) — an algorithm that bridges first- and second-order optimization.

Optimization efficiency is a central bottleneck in scaling Large Language Models (LLMs). While first-order methods like **Adam** dominate practical applications, second-order methods like **Shampoo** have shown remarkable theoretical and empirical promise, even outperforming Adam on benchmarks like Algoperf. However, Shampoo has historical drawbacks: significant computational overhead and a notoriously complex set of additional hyperparameters.

SOAP elegantly bridges these worlds by demonstrating a formal mathematical equivalence between **Shampoo and Adafactor**, then generalizing it. (For another matrix-aware optimizer now in production use — Muon and its Newton–Schulz orthogonalization — see the [DeepSeek-V4 architecture & training notes]({% post_url 2026-04-26-DeepSeek-V4-Arch-Train %}).)

---

## The Theoretical Insight: Shampoo is Adafactor in Disguise

The foundational insight of SOAP relies on a striking theoretical connection: **Shampoo (specifically implemented with an exponent of 1/2) is mathematically equivalent to running Adafactor in the eigenbasis of Shampoo's preconditioner.**

Let $L = \mathbb{E}[G_B G_B^\top]$ and $R = \mathbb{E}[G_B^\top G_B]$ represent the left and right preconditioners over a batch $B$, and let $Q_L$ and $Q_R$ be their respective eigenvectors. By rotating the gradient $G$ into this eigenspace as $G'_t = Q_L^\top G_t Q_R$, we can analyze the behavior of both algorithms in a shared basis.

The authors formally prove (**Claim 1**) that running an idealized Adafactor on this rotated gradient $G'_t$ matches Shampoo's eigenvalue scaling perfectly. Specifically, let the eigenvalues of $L$ be $\lambda_1, \dots, \lambda_m$. Idealized Shampoo scales the $i, j$ coordinate by $\left(\frac{\lambda_i \mu_j}{\sum_i \lambda_i}\right)^{-1/2}$. In Adafactor, the row-wise second-moment estimate is defined as $A_i = e_i^\top \mathbb{E}[G'_B \odot G'_B] \mathbf{1}_m$. The derivation elegantly proves this equivalence:

$$A_i = \mathbb{E}\left[\sum_j (u_i^\top G_B v_j)^2\right] = \mathbb{E}\left[\|u_i^\top G_B\|^2\right] = \mathbb{E}\left[u_i^\top G_B G_B^\top u_i\right] = \lambda_i$$

Because $A_i$ directly corresponds to the eigenvalues $\lambda_i$ (and similarly for the columns $C_j$ mapping to $\mu_j$), Shampoo is revealed to be nothing more than **Adafactor applied within a second-order rotated space**.

---

## The SOAP Algorithm

Building on this insight, the authors generalize the approach: rather than restricting the diagonal preconditioner to Adafactor, we can run **AdamW** directly in Shampoo's eigenspace — giving birth to SOAP.

For a given 2D weight matrix $W \in \mathbb{R}^{m \times n}$, a single step of SOAP operates as follows:

1.  **Rotate the Gradient:** $G' \leftarrow Q_L^\top G Q_R$.
2.  **Run Adam on the Rotated Gradient:**
    *   Update momentum $M \leftarrow \beta_1 M + (1-\beta_1)G'$.
    *   Update variance $V \leftarrow \beta_2 V + (1-\beta_2)(G' \odot G')$.
    *   Apply the Adam step: $N' \leftarrow M / \sqrt{\hat{V} + \epsilon}$.
3.  **Rotate Back to Original Space:** $N \leftarrow Q_L N' Q_R^\top$.
4.  **Update Weights:** $W \leftarrow W - \eta N$.
5.  **Update Preconditioners:** $L \leftarrow \beta_2 L + (1-\beta_2) G G^\top$ and $R \leftarrow \beta_2 R + (1-\beta_2) G^\top G$. Periodically (every $f$ steps), recompute the eigenvectors $Q_L, Q_R$.

---

## Key Insights and Empirical Triumphs

1.  **Drastic Efficiency Gains.** On language-model pre-training tasks for **360M and 660M** parameter models in the large-batch regime, **SOAP reduced the number of training iterations by >40% and wall-clock time by >35% compared to AdamW**. It even beat heavily tuned DistributedShampoo by **~20%** in both metrics.
2.  **Robustness to Preconditioning Frequency.** The most straightforward way to speed up Shampoo is to calculate its eigendecomposition less often, but empirical results show this rapidly degrades Shampoo's performance. **Because SOAP updates the running average of the second moment at every single step (just as Adam does) within the slowly changing coordinate basis, its performance degrades much slower than Shampoo's at high preconditioning frequencies.**
3.  **Expanding the Critical Batch Size.** As batch sizes scale up, optimizers eventually hit diminishing returns—a threshold known as the **critical batch size**. In experiments, SOAP tracks much closer to the ideal linear scaling curve than AdamW when batch size increases, meaning it effectively increases the critical batch size and unlocks greater efficiency in highly parallel, large-batch setups. (For why critical batch size matters to training throughput and compute-optimal allocation, see [Infra Math for LLM Training]({% post_url 2025-11-28-LLM-Train-GPU %}) and [The Architecture of Scaling Laws]({% post_url 2026-06-25-The-Architecture-of-Scaling-Laws %}).)
4.  **Algorithmic Simplicity & Hyperparameters.** Traditional Shampoo introduces several complex hyperparameters and graftings. Because SOAP fundamentally acts as "Adam in a rotated space," it dramatically simplifies tuning. **It requires only one additional hyperparameter compared to AdamW: the preconditioning frequency $f$.**

---

## Implementation Nuances

To make SOAP computationally viable, the authors employ clever engineering tricks:

*   **Targeted Application:** 1D layers bypass the eigendecomposition overhead entirely and just use standard AdamW. Furthermore, for massive dimensions like the first/last layers of a transformer, rotation matrices are fixed to the identity matrix.
*   **Power Iterations over Eigendecomposition:** Instead of using slow matrix inversions or standard eigendecompositions (`torch.linalg.eigh`), SOAP utilizes a single step of the **power method coupled with QR decomposition** (`torch.linalg.qr`), which is significantly faster in PyTorch and offers identical performance.

---

For the broader picture of how optimizer choice fits into practical large-scale training recipes, see the [Smol Training Playbook note]({% post_url 2025-11-29-Smol-Train %}).
