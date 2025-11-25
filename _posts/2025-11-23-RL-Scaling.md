---
layout: post
title: "Predictable Scaling of Reinforcement Learning for LLMs"
date: 2025-11-23
categories: [RL, Scaling]
tags: [RL, Scaling]
---

Read on the following paper:
- [The Art of Scaling Reinforcement Learning Compute for LLMs](https://arxiv.org/pdf/2510.13786).

### Context and Motivation

The paper addresses the critical issue that while RL has become central to training LLMs, the field **lacks predictive scaling methodologies** comparable to those established for pre-training. Despite rapidly increasing compute budgets (with some RL training runs consuming 100,000 H800 GPU hours), the process of scaling RL remains "more art than science". The goal of this work is to provide a scientific framework and a practical recipe for RL training that offers predictability.

### Methodology: Predictive Scaling Framework

The core contribution is establishing a predictive framework for RL performance by modeling the relationship between expected reward ($R_C$) on an iid validation set and training compute ($C$).

1.  **Sigmoidal Fit:** Unlike pre-training, which often uses power-law fits, the authors found that a **sigmoidal function** (Equation 1) provides a more robust and stable predictive fit for bounded metrics like pass rate/accuracy in RL.
    $$R_C - R_0 = (A - R_0) \times \frac{1}{1 + (C_{\text{mid}}/C)^B} \text{ (Equation 1)}$$
2.  **Scaling Parameters:** This model allows researchers to extrapolate performance from lower-compute runs to higher compute budgets. The parameters have intuitive interpretations:
    *   **Asymptotic Reward ($A$):** Represents the **asymptotic performance ceiling** achievable at large compute scales.
    *   **Scaling Exponent ($B$):** Determines the **compute efficiency** (steepness of the curve).
    *   **$C_{\text{mid}}$:** The compute point where half of the total gain is achieved.

### Empirical Findings from Ablations

The study involved a systematic investigation using **over 400,000 GPU-hours**. This investigation yielded three key principles:

1.  **Performance Ceilings are Not Universal:** Different training methods encounter different ceilings ($A$) on achievable performance, which can be modulated by choices such as loss type and batch size.
2.  **Efficiency vs. Asymptote:** Many design details (e.g., loss aggregation, normalization, curriculum) primarily **modulate compute efficiency ($B$)** without significantly shifting the performance asymptote ($A$).
3.  **Predictability:** Stable, scalable recipes follow predictable scaling trajectories, allowing for extrapolation from smaller-scale runs. The authors also note "The Bitter Lesson"—methods that appear superior at small compute budgets can be worse when extrapolated to large compute regimes.

The ablations confirmed that the **off-policy algorithm, loss function, and model precision** are among the most important decisions, potentially influencing the asymptotic performance ($A$).

### The ScaleRL Recipe

Based on empirical findings, the authors propose **ScaleRL** (Scale-able RL), a recipe that scales predictably and achieves state-of-the-art performance. ScaleRL integrates several existing methods and design choices:

*   **RL Setup:** Uses an asynchronous **PipelineRL-8** setup, which was found to be much more efficient (higher $B$) and slightly better asymptotically than the classic PPO-off-policy approach.
*   **Loss Function:** Optimizes the $J_{CISPO}$ loss, which is a **truncated importance sampling RL loss**. CISPO substantially outperformed DAPO and GSPO by improving the asymptotic pass rate ($A$). Furthermore, CISPO demonstrated robustness to hyperparameter choices, unlike GRPO/DAPO-style losses.
*   **Precision Fix:** Incorporates **FP32 precision at the LLM logits** (LM head), which dramatically improved the asymptotic performance ($A$) from 0.52 to 0.61 in initial studies by mitigating numerical mismatches between generators and trainers.
*   **Normalization/Aggregation:** Uses **prompt-level loss averaging** and **batch-level advantage normalization**.
*   **Curriculum/Filtering:** Includes **zero-variance filtering** (dropping prompts with zero advantage) and **No-Positive-Resampling** (filtering out prompts with a historical pass rate $\geq 0.9$) to improve scalability and the asymptotic reward ($A$).

### Validation and Scaling Across Compute Axes

ScaleRL was validated in a massive training run scaled up to **100,000 GPU-hours** on an 8B dense model. The extrapolated curve, fitted from initial training stages, closely matched the observed extended training points, demonstrating stability and predictive fits. ScaleRL achieved a higher asymptotic reward ($A=0.610$) and compute efficiency compared to prevalent recipes like MiniMax and DAPO.

The framework also proved predictive when scaling across multiple compute axes:

*   **Model Scale (MoE):** Scaling the recipe to a larger 17B$\times$16 MoE model (Scout) showed predictable scaling behavior. The larger MoE model achieved a **much higher asymptotic RL performance** than the 8B dense model, using only 1/6 of the 8B model's RL training compute to surpass its performance [Figure 1].
*   **Generation Length:** Increasing context from 14k to 34k tokens slowed initial progress but **consistently lifted the fitted asymptote ($A$)** (from 0.610 to 0.645), suggesting long-context RL is a ceiling-raising knob [Table 1, Figure 9].
*   **Batch Size:** Larger global batch sizes (e.g., 2048) led to slower early training but ultimately settled at a higher asymptotic performance ($A=0.645$), avoiding downstream stagnation seen in smaller batches [Figure 10].


### Asynchronous RL Setup: PipelineRL-k

ScaleRL utilizes an **asynchronous PipelineRL-8 setup** for its off-policy algorithm. The choice of off-policy algorithm is one of the most consequential design decisions, as it can shift the asymptotic performance ceiling ($A$).

*   **Mechanism:** In PipelineRL-k, the training operates in a streaming fashion. Generators continuously produce reasoning traces. As soon as trainers complete a policy update, the new parameters are immediately pushed to the generators, which use them, even if they must rely on a stale KV cache. Trainers wait if they get $k$ steps ahead of the generators. ScaleRL specifically uses $k=8$.
*   **Benefits:** PipelineRL was found to be much more **efficient** (higher scaling exponent $B$) compared to the traditional PPO-off-policy approach, allowing the model to reach the asymptotic ceiling faster. This efficiency gain is attributed to reducing idle time in the training process.
*   **Training Dynamics:** This approach maintains a tight feedback loop, keeping training closer to the on-policy regime, which reduces the mismatch between the generator and trainer distributions and ultimately affects the asymptotic performance.

### Generation Length Control: Interruptions vs. Length Penalty

To prevent the reasoning output lengths from exploding during training, which would harm both efficiency and stability, methods for controlling generation length are necessary. ScaleRL implements **interruptions**.

#### Interruptions (Used in ScaleRL)
*   **Mechanism:** Interruptions forcibly stop generations that are becoming too long by appending a specific marker phrase. This signals the LLM to terminate its reasoning process and produce a final answer.
*   **Implementation in ScaleRL:** The recipe uses the end-of-thinking phrase: **“Okay, time is up. Let me stop thinking and formulate a final answer now. </think>”**. These interruption tokens are placed randomly within a range (e.g., between 10k and 12k token length) to encourage generalization to different generation lengths.
*   **Stability:** Controlling truncation is vital, as runs exhibiting truncation rates in the range of 10–15% often destabilized training. ScaleRL runs generally kept truncations below 5% for the 8B model, contributing to stability.

#### Length Penalty (Compared to Interruptions)
*   **Mechanism:** This approach reshapes the reward function to penalize overly long completions. The penalty $R_{\text{length}}(y)$ is added **only to correct traces**.
*   **Formula (following DAPO):** The penalty uses a tolerance interval $L_{\text{cache}}$:
    $$R_{\text{length}}(y) = \text{clip}\left(\frac{L_{\text{max}} - |y|}{L_{\text{cache}}} - 1, -1, 0\right) \text{ (Equation 9)}$$.
    *   *Note:* In the comparison experiments, $L_{\text{max}}$ was set to 14k tokens and $L_{\text{cache}}$ to 2k tokens.
*   **Comparison Result:** Replacing the interruption mechanism with a length penalty in a Leave-One-Out (LOO) experiment did **not improve performance**. The interruption mechanism was retained in ScaleRL because it maintains slightly better compute efficiency ($B$) compared to the length penalty variant, while both reached similar asymptotic performance ($A$).

### Core ScaleRL Algorithmic Components

The full ScaleRL recipe integrates several components to ensure stability and high asymptotic performance:

| Component | Detail in ScaleRL | Rationale / Scaling Effect |
| :--- | :--- | :--- |
| **Loss Function** | **Truncated Importance Sampling Loss (CISPO)**. | CISPO substantially outperforms DAPO by achieving a **higher asymptotic pass rate ($A$)**. It is also markedly **more robust** to the choice of the Importance Sampling (IS) clipping parameter ($\epsilon_{\text{max}}$) than GRPO/DAPO losses. |
| **Precision Fix** | **FP32 precision at the LLM logits** (LM head). | This fix mitigates numerical mismatches between generator and trainer kernels, dramatically improving the **asymptotic performance ($A$)** (e.g., from 0.52 to 0.61 in initial studies). Its inclusion enhances stability and scalability, especially in larger MoE models [Figure 8b]. |
| **Loss Aggregation** | **Prompt-level loss averaging**. | This method achieved the highest asymptotic performance among the tested strategies (sample average, prompt average, token average) [Figure 14a]. |
| **Advantage Normalization** | **Batch-level advantage normalization**. | Advantages are normalized by the standard deviation across all generations in the batch. While prompt-level, batch-level, and no-normalization performed similarly in initial studies, batch-level was chosen as theoretically sound and marginally better. |
| **Filtering/Curriculum** | **Zero-Variance Filtering** and **No-Positive-Resampling**. | Zero-variance filtering excludes prompts where all generations have identical rewards, thus contributing zero policy gradient, which improves asymptotic performance [Figure 6a]. No-Positive-Resampling removes prompts with a historical pass rate $\geq 0.9$ from subsequent epochs, which improves scalability and the asymptotic reward ($A$) by focusing compute on harder examples [Figure 6b]. |

This combination of components results in the **$J_{\text{ScaleRL}}(\theta)$ loss function**, which integrates the chosen aggregation, normalization, filtering, and the CISPO loss.


### Conclusion

This work establishes a rigorous methodology for cost-effectively predicting the scalability of new RL algorithms for LLMs. By quantifying performance using the sigmoidal scaling law, the authors derived the ScaleRL recipe, which provides **practical predictability** in RL training, bringing it closer to the reliability long achieved in pre-training.

The methodology acts like a scientific speedometer for RL training: instead of waiting for a marathon to finish to see who is fastest, the scaling law allows researchers to predict the ultimate top speed ($A$) and acceleration ($B$) from just the beginning of the race.



