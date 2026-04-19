---
layout: post
title: "Training-Inference Parity in MoE Models: Where Numerics Drift"
date: 2026-04-08
categories: [MoE]
tags: [MoE, Inference, Numerics, RLHF, Floating-Point]
---

Blog: [Training-Inference Parity in MoE Models: Where Numerics Drift](https://fireworks.ai/blog/when-faster-not-identical-moe-numerics).

When deploying large language models, particularly Mixture-of-Experts (MoE) architectures like Kimi K2.5, Qwen3.5-MoE, and DeepSeek V3, the expectation is that identical weights and inputs will yield identical output distributions. However, **mathematically equivalent kernel fusions utilized during inference serving introduce subtle numerical drifts compared to training implementations**.

Maintaining sub-0.001 exactness in log-probabilities is highly critical for **RLHF / GRPO reward integrity**, as the policy model can exploit divergence gaps without actually improving its performance. These deviations compound across layers, and while they may only slightly shift lower-ranked token probabilities, they manifest as measurable KL divergence distribution mismatches.

---

## 1. The Fundamental Mathematical Root Cause

The core of every parity pitfall is a basic computational property: **floating-point (FP) addition is non-associative**.

Because each individual addition operation rounds its output to the nearest representable value, altering the summation order naturally alters the intermediate values, resulting in distinct rounding errors. While a single FP32 addition swap might seem negligible, these errors compound through 61 transformer layers. In MoE networks, this is devastating because a micro-deviation in the hidden state can flip discrete top-$k$ routing decisions, cascading the error through different expert pathways.

---

## 2. Detailed Breakdown of Parity Pitfalls

### Pitfall 1: All-Reduce Topology Discrepancies

Tensor-parallel inference requires summing the linear layer outputs via an `all-reduce` operation twice per layer (post-attention and post-MLP/MoE).

*   **Training Path (NCCL):** Uses a `reduce-scatter` across a ring topology. The accumulation buffer is divided into chunks owned by distinct GPUs. As partial sums rotate along the ring, **each chunk undergoes a different floating-point addition order**. For instance, Chunk 0 might be calculated as $r_0 + r_3 + r_2 + r_1$, while Chunk 1 is accumulated as $r_1 + r_0 + r_3 + r_2$.
*   **Inference Path (Lamport IPC / FlashInfer):** Custom serving engines replace NCCL for latency improvements, utilizing a Lamport kernel where GPUs write data to all other buffers via CUDA IPC. Consequently, every GPU calculates the **exact same fixed local sum order**: $r_0 + r_1 + r_2 + r_3$.

**Insight:** Both algorithms compute the exact identical sum in theoretical arithmetic utilizing FP32. However, because NCCL rotates chunk ownership per GPU, its local rounding sequences differ entirely from the uniform order utilized by the Lamport kernel, causing immediate numerical divergence.

### Pitfall 2: Fusing Communication with Computation (RMSNorm)

Fusing operations eliminates intermediate High Bandwidth Memory (HBM) round-trips, saving roughly 3 TB/s in bandwidth per operation. However, fusing an `all-reduce` operation with `RMSNorm` alters the underlying CUDA block layout.

*   **The Math of RMSNorm:** RMS normalization scales the hidden states based on a single reduced scalar: the sum of squares, $\sum(x^2)$. In the GPU, this scalar is calculated via a butterfly reduction—a 5-step binary tree across 32-thread warps using `__shfl_xor_sync`.
*   **The Mismatch:** A standalone `RMSNorm` keeps its own optimized block tree, pairing elements sequentially: e.g., first computing $(a+b)$ and $(c+d)$, then summing $(a+b) + (c+d)$. When fused, the thread layout is dictated by the requirements of the `all-reduce` kernel. **Different block sizes force different elements into different warps**, changing the initial butterfly pairings to things like $(a+c)$ and $(b+d)$.
*   **The Result:** The altered pairing tree produces a differently rounded $\sum(x^2)$ scalar. This slightly shifted input is then fed into the `rsqrtf` function, which sequentially scales every single dimension of the hidden state differently.

### Pitfall 3: Multi-Operation Fusions in MoE Cascades

MoE layers execute multi-operation combinations that compress three sequential kernels into a single persistent one on 58 MoE layers (layers 3–60):

1.  **MoE Finalize:** Weighted sum of the top-8 expert outputs.
2.  **All-Reduce:** Summation of partial results across GPUs.
3.  **Next-block Input RMSNorm + Residual addition.**

**Insight:** MoE routing algorithms are incredibly brittle right at the top-$k$ cutoff boundary. If inference calculates a gate score of 0.51 for Expert 1 and 0.49 for Expert 2, but the unfused training path calculated 0.49 for Expert 1 and 0.51 for Expert 2, the chosen expert flips. Once the executed expert path differs, numerical drift explodes exponentially instead of growing linearly.

---

## 3. Quantitative Measurement: The $k3$ Metric

To rigorously quantify this drift, inference engines are evaluated against reference code (with all fusions disabled) over 25 prompts of 200 generated tokens using **$k3$**, a stabilized, non-negative variant of KL divergence.

*   **Noise Floor Baseline:** $0.000070$
*   **With MLP Fusion enabled:** $0.000193$ (a 2.7x increase)
*   **Pass Threshold:** $< 0.001$

This shows that single fusions inherently drift results away from the noise floor, but the compound effect must remain below the strict threshold for the logprobs to be viable for RLHF.

---

## 4. Case Study: Qwen3.5-MoE Aggregation Precision Mismatch

During the bring-up of Qwen3.5-MoE with DeepEP parallelism, researchers discovered a dramatic split: text-token $k3$ remained at $0.005$, but **image-token $k3$ spiked to $0.296$**.

By running per-layer reduced tests, researchers observed that the initial 40 layers appeared "clean," but tiny deltas eventually compounded through the dense bidirectional attention mechanism. Swapping out just the MoE blocks for the Hugging Face reference collapsed both text and image $k3$ down to $0.000$.

The bug was purely isolated to **the casting and accumulation order in MoE aggregation**:

*   **Hugging Face Reference Path:** Multiplies routing score in `fp32`, casts *each* individual expert contribution down to `bf16`, and accumulates the sum via `index_add_` in `bf16` ($k3 = 0.000$).
*   **Fireworks Standard Path (BMM):** Multiplies routing score in `fp32`, performs a batched weighted sum accumulation in `fp32`, and casts to `bf16` *only once* at the very end ($k3 \approx 0.3$).
*   **DeepEP Combine Kernel:** Drops score precision to `bf16` before multiplying, then multiplies and sums in `bf16` ($k3 \ge 0.3$).

**Insight:** Because the HF reference inherently rounds each expert output *prior* to summation, utilizing a theoretically superior, higher-precision accumulation order (Fireworks Standard `fp32` sum) actively caused the model to break parity.

---

## Key Lessons

1.  **"Same math" ≠ "Same bits."** Mathematical parity does not equal numerical parity. Every divergence cited resulted from perfectly valid mathematically equivalent configurations that slightly altered floating-point accumulation orders.
2.  **Compound effects dominate.** A single cuBLAS tiling heuristic or a changed warp-shuffle reduction tree won't break parity. It is the cumulative effect of 61 layers of NCCL/Lamport topology differences plus 58 layers of MoE finalized fusions acting in concert that causes failure.
3.  **Provide Granular Engine Controls.** RLHF researchers require high-fidelity training reference mappings, while endpoint users prioritize low latency. Serving engines must expose per-fusion toggles rather than offering a generalized "disable optimizations" flag to support these vastly different workloads.
