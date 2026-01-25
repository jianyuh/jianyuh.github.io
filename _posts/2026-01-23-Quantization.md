---
layout: post
title: "Open Source Model Quantization Strategies"
date: 2026-01-23
categories: [Quantization]
tags: [Quantization]
---

This breakdown details quantization strategies across leading open-source model families and inference engines.

### 1. Model-Specific Quantization Matrix

This table details the primary quantization "recipes" used for specific model families in vLLM and SGLang.

| Model Family | Quantization Recipe | Bit Precision (Weight / Act) | Target Modules | vLLM Implementation | SGLang Implementation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepSeek-V3 / R1** | **FP8 Block-Wise** | **W8A8** (FP8 E4M3) | **Experts (MoE)** + Linear Layers | **DeepGEMM** (Integrated). vLLM now supports DeepGEMM for MoE and MQA logits. | **DeepGEMM** (Native). Uses this custom kernel to handle 128x128 block scaling efficiently. |
| **Kimi K2 Thinking** | **Native INT4** (QAT) | **W4A16** (INT4 / FP16) | All Linear Layers | **Compressed-Tensors**. Loaded via compressed-tensors format using optimized INT4 kernels. | **Native Support**. Day-0 support in v0.5.5; maps to FlashInfer INT4 kernels. |
| **GPT-OSS (120B/20B)** | **MXFP4** | **W4A8** (FP4 / FP8) | **MoE Layers** (Weights in e2m1 format) | **Marlin MXFP4**. Uses a specialized Marlin kernel variant for MXFP4 MoE operations. | **MXFP4**. Supported via recent updates to handle the specific Microscaling format. |
| **Llama 3.1 / 3.3** | **FP8** (FBGEMM) | **W8A8** (FP8) | Linear Layers (QKV, MLP) | **Machete** (Hopper). A mixed-input GEMM kernel optimized for H100s. | **FlashInfer / CUTLASS**. Uses JIT-compiled kernels for FP8 GEMM. |
| **Qwen 2.5 / Mistral** | **GPTQ / AWQ** | **W4A16** (INT4 / FP16) | Linear Layers | **Marlin** (Ampere). Standard highly-optimized INT4 kernel. | **FlashInfer**. Maps GPTQ/AWQ metadata to FlashInfer W4A16 kernels. |
| **General (Any)** | **AutoRound** | **W4A8** (INT4 / INT8) | Linear Layers | **AutoRound Plugin**. Supported via llm-compressor integration. | **AutoRound Native**. Directly loads AutoRound checkpoints for low-bit W4A8 inference. |

### 2. KV Cache Quantization Strategies

Quantizing the KV Cache is independent of model weights and is critical for increasing **context length** and **batch size**.

| Strategy | Precision | Supported Engines | Hardware Requirement | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **FP8 E4M3** | 8-bit | **vLLM & SGLang** | Hopper (H100), Ada (L40S) | Preferred format. Higher precision (3 mantissa bits) preserves accuracy better than E5M2. |
| **FP8 E5M2** | 8-bit | **vLLM & SGLang** | Hopper, Ampere (via cast) | Higher dynamic range but lower precision. Often used if E4M3 scaling factors are missing. |
| **NVFP4** | 4-bit | **Experimental** | Blackwell (B200) | Emerging standard. Reduces cache size by another 50% vs FP8. |

### 3. Kernel Deep Dive

To understand *why* performance differs, it is necessary to examine the specific kernel handling the mathematics:

*   **DeepGEMM (SGLang/vLLM):** Written specifically for **DeepSeek-V3**. Unlike standard kernels that scale per-tensor, DeepGEMM handles **fine-grained block-wise scaling** (a different scale factor for every 128x128 block). It uses FP32 accumulation to prevent overflow inherent in low-precision FP8 additions.
*   **Machete (vLLM):** A "mixed-input" kernel for **NVIDIA Hopper**. It enables **W4A16** (4-bit weights, 16-bit activations) by dequantizing weights inside the kernel pipeline. It utilizes the Tensor Memory Accelerator (TMA) to hide conversion costs behind memory transfers.
*   **Marlin (vLLM):** The standard for **INT4 on Ampere (A100)**. It restructures the weight matrix in memory to perfectly align with the GPU's access patterns, using asynchronous data movement (cp.async) to achieve near-theoretical peak bandwidth.
*   **FlashInfer (SGLang):** A JIT (Just-In-Time) compiler. Instead of pre-shipping binaries, it generates optimal kernel code at runtime based on the specific batch size and head dimension. It also supports fused kernels (e.g., RoPE fused into attention) to minimize launch overhead.

