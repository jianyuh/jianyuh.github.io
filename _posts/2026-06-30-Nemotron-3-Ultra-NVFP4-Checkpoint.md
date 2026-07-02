---
layout: post
title: "Creating the Nemotron 3 Ultra NVFP4 Checkpoint"
date: 2026-06-30
categories: [LLM, Quantization, Inference]
tags: [NVFP4, FP4, Quantization, Nemotron, Blackwell, Hopper, Mixed Precision, Model Optimizer, Megatron-LM, AutoQuantize]
---

Reading notes on [**Creating the NVIDIA Nemotron 3 Ultra NVFP4 Checkpoint with NVIDIA Model Optimizer**](https://developer.nvidia.com/blog/creating-the-nvidia-nemotron-3-ultra-nvfp4-checkpoint-with-nvidia-model-optimizer/) — compressing a 550B-parameter model into 4-bit floating point.

This is the quantization companion to the [Nemotron 3 Ultra deep dive]({% post_url 2026-06-06-Nemotron-3-Ultra %}) (the hybrid Mamba-Attention base model), and it leans heavily on the [4-over-6 adaptive scaling]({% post_url 2025-12-10-4Over6 %}) and [NVFP4 training]({% post_url 2025-11-20-NVFP4-Train %}) work covered previously.

---

## 1. Executive Summary & The Quantization Imperative

As large language models process increasingly larger context windows, mitigating the memory bottleneck of large model weights becomes vital. The NVIDIA team used the **NVIDIA Model Optimizer** to compress the **550-billion parameter Nemotron 3 Ultra** model into the novel **NVFP4** (4-bit floating point) format, introduced with the [NVIDIA Blackwell architecture]({% post_url 2026-04-12-blackwell-sm100 %}).

**The engineering results are highly substantial:** The model shrinks from **1,121 GB in BF16 down to 352.3 GB** (a **3.2× reduction**), cutting the hardware footprint in half. Furthermore, this checkpoint unlocks **up to 5.9× higher inference throughput** on decode-heavy workloads compared to the GLM-5.1 754B FP4 model, while matching the BF16 baseline accuracy across nearly every benchmark.

---

## 2. Hardware Flexibility and Mixed-Precision Topology

A key engineering insight is that **a single NVFP4 checkpoint can execute on both Blackwell and Hopper architectures** through dynamic weight format conversion:

*   On **Blackwell** hardware, the model leverages native **W4A4** (4-bit weights, 4-bit activations) execution.
*   On **Hopper** hardware, which lacks native FP4 tensor cores, the serving framework automatically falls back to **W4A16**. While W8A8 was considered for Hopper, its memory footprint left insufficient room for the model's Multi-Token Prediction (MTP).

A common misconception is that FP4 quantization is applied **homogeneously** across the model. In reality, **different layers are quantized to varying precision levels based on their architectural sensitivity**:

| Precision | Components |
|---|---|
| **NVFP4 (W4A4)** | MoE routed experts |
| **FP8 per-tensor** | MoE shared experts and Mamba mixer linears |
| **FP8** | KV cache |
| **FP16 with stochastic rounding** | Mamba SSM cache |
| **BF16 (Unquantized)** | Embedding, output classification, MTP layers, attention linears, latent MoE, and Mamba conv1d |

(The FP16-with-[stochastic-rounding]({% post_url 2025-12-11-Stochastic-Rounding %}) choice for the Mamba SSM cache is the same numerical trick used to keep low-precision accumulation unbiased.)

---

## 3. The Mathematics of FP4 Scaling (Derivations & Trade-offs)

The foundational challenge of FP4 quantization is its severely restricted grid. **FP4 can only represent 8 positive values: `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`.** To map floating-point weights to this grid, the system must derive a *scale* (a multiplier defining the representation's granularity). The engineers evaluated three primary mathematical approaches:

*   **Max (Absmax) Scaling:** The scale maps the largest value in the block to the maximum FP4 value. For a maximum value of 12.8, the scale is mathematically set to $12.8 / 6$. **Insight:** This method preserves extreme outliers perfectly, but compresses all smaller weights to near zero, causing significant information loss and downstream accuracy drops.
*   **Mean Squared Error (MSE) Scaling:** This sweeps for a scale that minimizes the average reconstruction error ($\sum (x - \hat{x})^2$) across the entire block. **Insight:** While it mathematically preserves the resolution of the bulk of small weights by saturating the outlier (e.g., pulling an outlier down to 2.0), the NVIDIA team found that minimizing MSE did *not* yield consistent improvements on downstream accuracy for Nemotron 3 Ultra.
*   **Four-Over-Six (4/6) Scaling (The Optimal Derivation):** This method targets a specific anomaly in the FP4 grid: the **massive gap between 4 and 6**. Any weight falling in this range is aggressively rounded, sometimes incurring a **>13% single-value error**. Four-over-six fixes this by independently calculating the reconstruction error for two different maximum mapping scales ($M = 4$ or $M = 6$) and picking the one that minimizes error for that specific block.
    *   *Derivation 1 (When $M = 6$ wins):* Given a block `[2, 4, 5.9, 6]`. If scaled to $M = 6$, `2, 4, and 6` land exactly on the grid, and `5.9` rounds trivially to 6. If the algorithm mistakenly chose $M = 4$, it would push 2 to 2.25 and 4 to 4.5, introducing unnecessary error.
    *   *Derivation 2 (When $M = 4$ wins):* For a block containing a large outlier (~30), scaling to $M = 6$ maps 30 to 4.62, which violently rounds down to 4 (a **13% rounding error**). However, scaling to $M = 4$ maps the block's values perfectly onto the grid points, reducing the MSE from **4.33 to a perfect 0.0**. (See the [4-over-6 post]({% post_url 2025-12-10-4Over6 %}) for the full worked example and the training-stability motivation.)

By applying 4/6 scaling to the FP4 routed-expert weights, the global per-tensor weight scale was raised by **1.75×**, and the median reconstruction MSE was slashed by **16.4%** compared to standard max calibration.

---

## 4. Optimizing Bits-Per-Element (BPE)

Effective **Bits-Per-Element (BPE)** defines the average number of bits required to store all weights. While BF16 uses 16 BPE, NVFP4 adds block and per-tensor scaling overhead, establishing a **theoretical minimum of 4.5 BPE**. (Note: the 32 bits required for the per-tensor scale are amortized and mathematically negligible in this calculation.)

To find the optimal balance between accuracy and size, the engineers used NVIDIA Model Optimizer's **`AutoQuantize`** constraint solver. Instead of hardcoding formats, `AutoQuantize` scores layer sensitivity and calculates a per-layer assignment to meet a strict bit budget. Sweeping operating points from **4.85 to 7.19 BPE** revealed that **5.03 BPE was the mathematical sweet spot**. At 5.03 BPE, the model achieved a **98.5% median recovery** relative to BF16 (surpassing both raw Max at 96.8% and MSE at 98.4%) and secured massive jumps in long-context reasoning benchmarks like **AA-LCR**.

---

## 5. Systems Engineering & Distributed Quantization

Handling a 550B model requires heavy infrastructure. The engineering team deployed **NVIDIA Megatron-LM** rather than Hugging Face Transformers for the post-training quantization (PTQ) pipeline. By sharding the model across **16 × B300 GPUs** using expert and data parallelism ($EP = DP = 16$), the total end-to-end quantization time was driven down from **120 minutes (HF) to just 45 minutes**.

At the structural level, quantization configs are processed as ordered YAML rules matching module-name patterns (e.g., `*weight_quantizer` vs `*input_quantizer`). For NVFP4, these dictate **`E2M1` elements mapped into 16-wide blocks constrained by `E4M3` block scales**. Because weights and activations are distinct objects in the logic, one can precisely quantize weights with 4/6 scaling while allowing activations to fall back to default dynamic scaling profiles.

---

For the broader landscape of post-training quantization strategies, see [Open Source Model Quantization Strategies]({% post_url 2026-01-23-Quantization %}); for where weight/KV-cache compression sits in the inference-efficiency stack, see the [LLM efficiency notes]({% post_url 2026-06-26-Efficiency-in-LLMs-Fast-Inference-Memory-Bandwidth %}).
