---
layout: post
title: "High Precision Used for Reasoning Recipes"
date: 2025-06-23
categories: [Determinism, ]
tags: [RL]
---

Read on [Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning](https://arxiv.org/pdf/2506.09501).


# Metrics
To quantify output instability and analyze the impact of numerical precision on LLM inference variability, the paper utilizes different metrics depending on the decoding scenario: greedy decoding and random sampling.

## 1. Greedy Decoding Scenario

In the greedy decoding setting, where the temperature is set to zero to theoretically produce deterministic outputs, the models are evaluated under 12 different runtime configurations (2 GPU types: NVIDIA L40S and A100; 2 GPU counts: 2 and 4; and 3 batch sizes: 8, 16, and 32). The following metrics are used to quantify output instability:
*   **Std@Acc (Standard deviation of accuracy)**: This metric measures the sample standard deviation of accuracy across the 12 different runtime configurations for each numerical precision (BF16, FP16, FP32). It indicates the stability of LLM inference outputs during greedy inference. Higher values signify greater variability in accuracy due to runtime configuration changes.
*   **Avg\_Std@Output\_Length (Average standard deviation of output length)**: This metric quantifies the variability in the length of generated output tokens. It's calculated by measuring the standard deviation of output tokens per example across the 12 runtime configurations, then averaging these standard deviations over the entire dataset. This provides an alternative perspective on output stability, particularly relevant for reasoning models which often produce long chains of thought. Large variances in output length, up to 9,000 tokens, can be observed with BF16 precision.
*   **Div\_Index (Divergence index)**: This metric identifies the first token position at which responses from different runs diverge for the same question. If two or more responses produce identical token sequences up to a certain position but differ thereafter, that position is the divergence index. A higher Div\_Index indicates greater consistency across responses under different runtime configurations. FP32 precision significantly reduces the percentage of divergent examples and pushes the divergence onset to much later token positions compared to BF16.
*   **Avg\_Std@top1\_prob (Average standard deviation of top-1 token prediction probability)**: This is considered a more informative metric for ablation studies as it directly reflects numerical instability from rounding errors. Before responses diverge, while the top-1 tokens are identical, their predicted probabilities may vary across settings due to floating-point computation errors. This metric computes the standard deviation of the predicted probability for the top-1 token at each position (from 0 to Div\_Index), then averages it over all positions and examples. It indicates the magnitude of numerical variation introduced by floating-point errors. BF16 consistently shows significantly higher variance in top-1 token probabilities compared to FP16 and FP32, which is critical because token selection is highly sensitive to minimal probability differences between competing tokens.
These metrics collectively demonstrate that greedy decoding, contrary to common belief, does not guarantee deterministic outputs across different hardware and system configurations, with FP32 showing near-perfect reproducibility and BF16 exhibiting substantial instability.

## 2. Random Sampling Scenario

For evaluations using random sampling (with a non-zero temperature, typically T > 0), the primary metric is:
*   **Pass@1 (Mean accuracy)**: This metric measures the mean accuracy averaged over multiple independent runs of the model. It's essentially the probability that at least one solution attempt (from K attempts) will succeed.
    *   In the context of evaluating reproducibility, the standard deviation of Pass@1 performance is reported across different system configurations (e.g., varying GPU count and numerical precision). This specifically reflects the variability introduced by limited numerical precision, rather than the inherent randomness of the model itself. For


# Ablation Study on Runtime Configurations Affecting Reproducibility of LLM Inference

The paper provides a detailed ablation study on how runtime configurations affect the reproducibility of LLMs, particularly focusing on greedy decoding settings. The analysis investigates the impact of GPU count, batch size, and GPU type on output stability across different numerical precision formats (BF16 and FP16), noting that FP32 largely mitigates these issues.

## 1. GPU Count

*   Configurations utilizing 4 GPUs generally exhibit higher probability variation compared to those with 2 GPUs, especially when using BF16 precision.
*   This increased variability is potentially due to more parallel computations leading to varied floating-point operation orderings and, consequently, different rounding errors.
*   For example, on the LiveCodeBench-Hard dataset, DeepSeek-R1-Distill-Qwen-7B with BF16 precision showed an Avg\_Std@top1\_prob of 38.6 (x10^-4) for 2 GPUs and 48.7 (x10^-4) for 4 GPUs.
*   This trend was observed across various models and tasks with L40S GPUs, where increasing GPU count from 2 to 4 typically led to higher Avg\_Std@top1\_prob.
*   However, this trend was less consistent with A100 GPUs, where in some cases, 2-GPU results were slightly higher than 4-GPU results, possibly due to A100s inherently higher instability in the experiments.

## 2. Batch Size

*   Smaller batch sizes counter-intuitively produce higher variance in token probabilities.
*   The reasoning for this is that smaller batches may necessitate more sequential processing steps, which can accumulate rounding errors.
*   Conversely, larger batch sizes tend to lead to lower Avg\_Std@top1\_prob, indicating better reproducibility.
*   This is because larger batches can benefit from parallel computation within optimized CUDA kernels, which can limit error accumulation.
*   For instance, on the MATH500 dataset with BF16, DeepSeek-R1-Distill-Qwen-7B showed an Avg\_Std@top1\_prob of 34.1 (x10^-4) for batch size 8 and 28.6 (x10^-4) for batch size 32.

## 3. GPU Architecture (Type)

*   The specific GPU hardware also plays a role in reproducibility.
*   A100 GPUs generally exhibit slightly higher probability variance than L40S GPUs under identical configurations.
*   This difference is likely attributable to variations in hardware-level floating-point implementations and memory hierarchies.
*   For example, across all tasks and models, the Avg\_Std@top1\_prob on A100 GPUs was consistently slightly higher than on L40S GPUs under the same experimental settings.

# LayerCast: An Optimized FP32 Inference Pipeline:

- Recognizing that full FP32 inference incurs significant memory and time costs (doubling memory usage compared to BF16), the paper proposes LayerCast as a more efficient solution.

- How it works: LayerCast is a hybrid precision approach that stores model parameters in memory-efficient 16-bit formats (specifically BF16 for weights and biases of linear layers) but performs all computations in full FP32 precision. It achieves this by loading model parameters initially in FP32, explicitly casting linear layer weights and biases to BF16 for storage, and then upcasting each weight back to FP32 just-in-time for matrix multiplication.

- Benefits: LayerCast achieves determinism and stability nearly identical to full FP32 inference (often with zero or near-zero standard deviation in accuracy and divergence rates below 3.4%) while significantly reducing memory usage (by 34% compared to full FP32, especially beneficial for KV cache in long-context scenarios). This balances memory efficiency with numerical stability


# FP32 Precision Mitigation Strategy for [MiniMax-M1](https://arxiv.org/pdf/2506.13585)

For MiniMax-M1, FP32 precision is used as a specific mitigation strategy to address a crucial challenge encountered during its Reinforcement Learning (RL) training, ultimately improving accuracy by enabling successful reward growth.

*   **Addressing Computational Precision Mismatch**: During the RL training of MiniMax-M1, a significant discrepancy was observed in the probabilities of rolled-out tokens between the training mode and inference mode. This "precision mismatch" between the training and inference kernels was a critical issue that prevented the model's reward from growing.
*   **Specific Application to LM Output Head**: Through layer-by-layer analysis, the researchers identified that high-magnitude activations in the Language Model (LM) output head were the primary source of this error. To fix this, they increased the precision of the LM output head to FP32.
*   **Improved Probability Correlation and Reward Increase**: This adjustment effectively realigned the theoretically identical training and inference probabilities, significantly improving their correlation from approximately 0.9x to 0.99x. This improved precision was crucial because it enabled successful reward increase during RL training, which is directly tied to the model's performance and accuracy gains.
