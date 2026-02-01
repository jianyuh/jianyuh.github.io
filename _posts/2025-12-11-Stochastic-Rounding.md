---
layout: post
title: "Pure BF16 Training via Stochastic Rounding"
date: 2025-12-11
categories: [Stochastic Rounding]
tags: [Stochastic Rounding]
---

Reading the following papers:
- [Stochastic Rounding for LLM Training: Theory and Practice](https://arxiv.org/pdf/2502.20566)
- [Revisiting BFloat16 Training](https://arxiv.org/pdf/2010.06192)

---

This first paper challenges the prevailing "Mixed Precision" (MP) paradigm, which relies on maintaining FP32 master weights and optimizer states to preserve accuracy. The authors propose a pure BF16 training pipeline that utilizes **Stochastic Rounding (SR)** within the AdamW optimizer update step. Unlike previous attempts at low-precision training which suffered performance gaps, this method outperforms standard (BF16, FP32) Mixed Precision in terms of validation perplexity, throughput (up to $1.54\times$), and memory efficiency (30% savings) for models up to 6.7B parameters.

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
The first paper provides a rigorous theoretical justification for why SR works where NR fails, particularly regarding the optimizer's behavior.

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
3.  **Hardware Alignment:** The approach aligns with hardware trends. New accelerators (e.g., AWS Trainium) support SR at the hardware level, but the first paper shows it can be implemented purely in software (optimizer step) on GPUs with negligible cost.
4.  **No Auxiliary Tensors:** Unlike other recent low-precision methods (e.g., Kahan summation or composited floating point), this approach requires no auxiliary variables, keeping the implementation lightweight.

---

This second paper challenges the prevailing "mixed-precision" paradigm in deep learning, which assumes that while activations can be low-precision (16-bit), 32-bit floating-point units (FPUs) are required for weight accumulation to maintain accuracy. The authors propose **"16-bit-FPU training,"** a method using *only* BFloat16 units for all storage and computation. By identifying that "nearest rounding" during weight updates is the primary cause of convergence failure, they demonstrate that utilizing **Stochastic Rounding** or **Kahan Summation** allows pure 16-bit training to match 32-bit accuracy across state-of-the-art models (BERT, ResNet, DLRM).

#### **Motivation: The Hardware Cost of Precision**
*   **Current State:** Deep learning accelerators typically support mixed-precision training: 16-bit for signals (activations/gradients) but 32-bit for master weights and optimizer states. This forces hardware to include expensive 32-bit FPUs.
*   **The Opportunity:** BFloat16 units are significantly more efficient than 32-bit units, offering $3\times$ higher power efficiency, $1.5\times$ lower latency, and $1.5\times$ less chip area.
*   **The Problem:** Naively replacing 32-bit operations with 16-bit operations (using standard nearest rounding) leads to significant accuracy degradation (e.g., 16% training accuracy drop in BERT).

#### **Technical Diagnosis: The Vanishing Update**
The authors provide a theoretical and empirical analysis of *why* naive 16-bit training fails.

*   **The Bottleneck is Weight Updates:** Through ablation studies, the authors prove that rounding errors in the forward and backward passes (gradients/activations) have negligible impact on convergence. The critical failure point is the **model weight update** step.
*   **The Mechanism of Failure:** When using nearest rounding (the standard hardware mode), small gradient updates are often completely canceled when added to larger weight values.
    *   *The Swamping Effect:* If the update term $\alpha \nabla f(w_t)$ is smaller than half the distance between representable values (machine epsilon $\epsilon$) relative to the weight magnitude, the operation $w_{t+1} = Q(w_t - \alpha \nabla f(w_t))$ results in $w_{t+1} = w_t$.
    *   *Theoretical Lower Bound:* The authors derive a lower bound for convergence error. With nearest rounding, the weights halt in a region around the optimum proportional to $\epsilon$. Since BFloat16 has a large $\epsilon$, this "halting" prevents the model from reaching the high-precision solution required for convergence.

#### **Proposed Solutions**
To enable pure 16-bit training, the second paper proposes applying one of two numerical analysis techniques specifically to the **weight update** step, while leaving all other operations (convolutions, matrix muls) as standard nearest-rounded BFloat16.

##### **A. Stochastic Rounding (SR)**
*   **Mechanism:** Instead of rounding to the nearest value, round up or down with a probability proportional to the distance to the representable value.
*   **Benefit:** This provides an unbiased estimate of the true value. Even if updates are small, the expectation of the weight converges correctly, preventing the "halting" effect.
*   **Implementation:** Efficient on modern hardware, requiring only a shift register for random bit generation and addition, avoiding expensive multiply/divide operations.

##### **B. Kahan Summation**
*   **Mechanism:** Uses an auxiliary 16-bit variable ($c_t$) to track the "lost" numerical error from rounding during accumulation.
    *   Algorithm: The error is computed as $(result - weight) - update$ and subtracted from the *next* update.
*   **Benefit:** Allows the system to utilize standard nearest rounding (supported by all hardware) while mathematically compensating for the precision loss over time.
*   **Trade-off:** Requires $2\times$ memory for weights (to store the auxiliary variable), but can achieve slightly higher accuracy than SR.

#### **Key Experimental Results**
The authors validated their approach on ResNet (CIFAR10/ImageNet), BERT (MNLI/Wiki103), DLRM, and DeepSpeech2.

*   **Parity with 32-bit:** 16-bit-FPU training with either SR or Kahan summation matched the validation accuracy of full 32-bit training across applications (differences within $-0.1\%$ to $+0.2\%$).
*   **Ablation Proof:** Using 32-bit storage *only* for model weights (while keeping all math in 16-bit) recovered accuracy gaps, confirming the weight update hypothesis.
*   **Technique Comparison:** Stochastic rounding matched 32-bit performance in 5 out of 7 applications. Kahan summation closed the remaining gaps, proving slightly more robust but memory-intensive.
*   **Float16 vs. BFloat16:** The techniques failed when applied to standard Float16 (IEEE 754 half-precision) due to its limited dynamic range; BFloat16 is essential for this approach.

#### **Insights**

*   **Hardware Design Shift:** The second paper provides a "software proof of existence" for simpler hardware. Accelerators do not strictly need 32-bit FPUs to train state-of-the-art models. Manufacturers can dedicate silicon area entirely to 16-bit units if they support stochastic rounding or if software stacks implement Kahan summation.
*   **The "Master Copy" Myth:** The standard practice in Mixed Precision training is keeping a 32-bit "master copy" of weights. The second paper argues that a 32-bit master copy is unnecessary; a 16-bit weight plus a 16-bit Kahan error term (effectively a split 32-bit representation) or a statistically averaged 16-bit weight (via SR) is sufficient.
*   **Accuracy-Memory Trade-off:** The authors expose a tunable trade-off. Practitioners can choose Stochastic Rounding for maximum memory efficiency ($2\times$ savings over mixed precision) or Kahan Summation for maximum stability (at the cost of memory parity with mixed precision).
*   **Theoretical Robustness:** The derivation of a lower bound for nearest-rounding SGD convergence complements existing literature that mostly focuses on upper bounds. It mathematically formalizes why learning rate tuning cannot fix quantization errors.

