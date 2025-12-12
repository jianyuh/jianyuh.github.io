---
layout: post
title: "Pure BF16 Training via Stochastic Rounding"
date: 2025-12-11
categories: [Stochastic Rounding]
tags: [Stochastic Rounding]
---

Reading the following paper:
- [Stochastic Rounding for LLM Training: Theory and Practice](https://arxiv.org/pdf/2502.20566)


This paper challenges the prevailing "Mixed Precision" (MP) paradigm, which relies on maintaining FP32 master weights and optimizer states to preserve accuracy. The authors propose a pure BF16 training pipeline that utilizes **Stochastic Rounding (SR)** within the AdamW optimizer update step. Unlike previous attempts at low-precision training which suffered performance gaps, this method outperforms standard (BF16, FP32) Mixed Precision in terms of validation perplexity, throughput (up to $1.54\times$), and memory efficiency (30% savings) for models up to 6.7B parameters.

#### **The Core Technical Challenge**
The standard for LLM training is Mixed Precision (e.g., PyTorch AMP O1/O2), where operations occur in BF16/FP16, but master weights and accumulations remain in FP32.
*   **The Bottleneck:** Maintaining FP32 copies consumes significant memory and bandwidth.
*   **The Failure of Naive BF16:** Simply converting the entire pipeline to BF16 (Vanilla BF16) fails because the default **Nearest Rounding (NR)** is biased. When updates are small relative to the parameter magnitude (gradient $\times$ LR $\ll$ weight), NR rounds the update to zero, causing "stagnation" and significant accuracy degradation.

#### **Proposed Methodology: BF16-AdamW-SR**
The authors introduce a modified AdamW optimizer where all states (momentum, variance, weights) are stored in BF16. The critical innovation lies in the **weight update step**:

*   **Algorithm:**
    1.  Compute gradients and optimizer states ($m_t, v_t$) in BF16.
    2.  Perform the update step: $x_{t+1} \leftarrow Q_{SR}(x_t - \eta_t \cdot \text{update})$.
    3.  **$Q_{SR}$ (Stochastic Rounding):** Instead of rounding to the nearest representable number, $x$ is rounded up or down with probability proportional to the distance to the grid points. This makes the rounding an **unbiased estimator** ($E[Q_{SR}(x)] = x$), allowing small updates to accumulate over time rather than vanishing.
*   **Shared Randomness:** In distributed data-parallel settings, the random seed for SR must be synchronized across devices (forking the random state). Without this, identical inputs on different GPUs would round to different values, causing model replicas to drift apart and disrupting gradient aggregation.
*   **Implementation:** The update is implemented via "dithering"—adding uniform noise to the mantissa bits of a temporary FP32 upcast before truncation. This incurs negligible overhead.

#### **Theoretical Insights & Analysis**
The paper provides a rigorous theoretical justification for why SR works where NR fails, particularly regarding the optimizer's behavior.

**A. Implicit Regularization**
The authors define a "modified gradient flow" to analyze the error terms introduced by discretization.
*   **Theorem 1:** Training with SR implicitly minimizes a modified loss function:
    $$F_{SR}(x) \approx F(x) + \frac{\alpha}{4}\|\nabla F(x)\|^2 + \text{Quantization Penalty}$$
    This acts as a regularizer. The error term introduced by SR is $O(\alpha^3)$, whereas biased rounding (NR) introduces an $O(\alpha)$ error that accumulates at every step.

**B. The Learning Rate (LR) Paradox**
A crucial insight is the relationship between SR and Learning Rate:
*   **Low LR Stagnation:** At low learning rates, SR behaves like a random walk. If the update is much smaller than the precision gap, convergence time increases significantly.
*   **High LR Robustness:** SR requires a higher learning rate to be effective (to overcome the quantization noise). Surprisingly, the authors prove that SR **decorrelates gradients** (additive noise), making the training more robust to high LRs that would typically cause Mixed Precision training to diverge.
*   **Convergence Bound:** The convergence analysis for Adam+SR shows that the additional quantization error term can be subsumed by Adam's natural convergence bound if the learning rate is sufficiently high. In contrast, Adam+NR contains a non-vanishing error term due to bias.

#### **Empirical Results**
Experiments were conducted on GPT-2 (350M, 770M) and GPT-Neo (1.3B, 2.7B, 6.7B).

*   **Validation Perplexity:** BF16+SR achieved *lower* (better) perplexity than standard Mixed Precision (e.g., 10.05 vs. 10.11 on GPT-Neo 6.7B).
*   **Throughput:**
    *   $1.54\times$ faster than O1 MP (PyTorch `torch.amp`) for 6.7B models.
    *   Marginally faster than O2 MP (Megatron-LM style) but with better accuracy.
*   **Memory:** Up to **30% memory reduction** compared to MP (O1) by eliminating FP32 master weights and optimizer states.
*   **Robustness:** BF16+SR converged successfully with learning rates $2\times$ to $4\times$ higher than the maximum stable LR for Mixed Precision.

#### **Takeaways**
1.  **Elimination of the Master Copy:** The most significant practical contribution is demonstrating that the FP32 master copy—long considered essential for convergence—is redundant if SR is applied correctly. This frees up massive HBM for larger batch sizes or models.
2.  **Hyperparameter Shift:** Adopting this method requires a shift in tuning intuition. Push learning rates higher than usual. The "noise" of SR acts as a stabilizer against the instability usually caused by high LRs.
3.  **Hardware Alignment:** The approach aligns with hardware trends. New accelerators (e.g., AWS Trainium) support SR at the hardware level, but this paper shows it can be implemented purely in software (optimizer step) on GPUs with negligible cost.
4.  **No Auxiliary Tensors:** Unlike other recent low-precision methods (e.g., Kahan summation or composited floating point), this approach requires no auxiliary variables, keeping the implementation lightweight.

#### **Analogy**
Think of training with pure BF16 and Nearest Rounding as trying to walk up a gentle slope using a staircase where the steps are too tall; if your stride (update) is shorter than the step height, you simply cannot move up—you stay on the same step forever (stagnation).

**Stochastic Rounding** is like vibrating the staircase. Even if your stride is short, the vibration (randomness) occasionally bumps you up to the next step. Over time, your average position moves correctly up the slope. The authors' insight on **Learning Rate** is akin to realizing that if you take bigger leaps (higher LR), the vibration doesn't knock you off balance; instead, it prevents you from getting stuck in local ruts, allowing you to climb faster than someone walking carefully on a smooth ramp.
