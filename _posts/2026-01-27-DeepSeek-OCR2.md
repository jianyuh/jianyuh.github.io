---
layout: post
title: "DeepSeek-OCR 2 with DeepEncoder V2"
date: 2026-01-27
categories: [DeepSeek]
tags: [DeepSeek]
---

Reading the following paper:
- [DeepSeek-OCR 2: Visual Causal Flow](https://github.com/deepseek-ai/DeepSeek-OCR-2/blob/main/DeepSeek_OCR2_paper.pdf)

## 1. Summary & Motivation
DeepSeek-OCR 2 investigates a fundamental shift in Vision-Language Model (VLM) architecture: replacing the rigid, raster-scan processing of visual tokens with **Visual Causal Flow**. Conventional VLMs, which flatten 2D images into 1D sequences (top-left to bottom-right) using fixed positional encodings, contradict human visual perception. Human vision is "foveal" and "causally-driven," scanning images based on semantic logic rather than spatial coordinates. To mimic this, DeepSeek-OCR 2 introduces **DeepEncoder V2**, an encoder capable of dynamically reordering visual tokens based on image semantics prior to interpretation by the LLM decoder.

## 2. Technical Architecture: DeepEncoder V2

The core innovation lies in the encoder design, which abandons the traditional CLIP component in favor of a compact LLM-style architecture.

### A. LLM as Vision Encoder
*   **Base Model:** The encoder is instantiated using **Qwen2-0.5B**. Its 500M parameter count is comparable to a standard CLIP ViT (300M) but leverages the advanced optimizations of the LLM community (e.g., MoE, efficient attention).
*   **Dual-Stream Attention Mechanism:** The model employs a hybrid attention mask to achieve "visual causal flow":
    1.  **Visual Tokens (Bidirectional):** The original visual tokens utilize bidirectional attention (similar to ViT), maintaining a global receptive field where tokens can attend to each other.
    2.  **Causal Flow Queries (Unidirectional):** A set of learnable queries is appended as a suffix. These queries use a causal triangular mask. Each query can attend to all visual tokens and all *preceding* queries, but not future ones.
*   **Function:** This design allows the queries to progressively distill and reorder information from the visual tokens. Only these reordered **causal flow tokens** are fed to the final LLM decoder.

### B. Vision Tokenizer
Before the LLM-style encoder, the image is processed by a Vision Tokenizer.
*   **Structure:** It combines an 80M-parameter SAM-base model with two convolutional layers.
*   **Compression:** It achieves a **16× token compression** rate, reducing the dimension to 896. This is critical for managing computational costs while retaining the benefits of the subsequent global attention module.

### C. Multi-Crop Strategy & Token Budget
DeepSeek-OCR 2 utilizes a multi-crop strategy to handle varying resolutions without maintaining multiple query sets.
*   **Global View:** 1024×1024 resolution yielding **256** query embeddings.
*   **Local Views:** 768×768 resolution yielding **144** query embeddings per crop.
*   **Dynamic Range:** The number of crops ($k$) ranges from 0 to 6. The total token count fed to the LLM is calculated as $k \times 144 + 256$, resulting in a range of **256 to 1120 tokens**.
*   **Efficiency:** The upper bound (1120) aligns with the maximum visual token budget of Gemini-3 Pro, positioning this as a highly efficient architecture.

### D. Decoder
The decoder remains the **DeepSeek-MoE** (3B parameters, ~500M active), inherited from the previous DeepSeek-OCR iteration.

## 3. Training Pipeline
The model is trained in three distinct stages:

1.  **Encoder Pretraining:** The DeepEncoder V2 (Vision Tokenizer + LLM-style encoder) is trained using a language modeling objective. It is coupled with a lightweight decoder for next-token prediction to learn feature extraction and token reordering.
2.  **Query Enhancement:** The encoder is integrated with the DeepSeek-3B decoder. The Vision Tokenizer is frozen, but the LLM encoder and decoder are jointly optimized to enhance query representations.
3.  **Continue-Training LLM:** The entire DeepEncoder V2 is frozen. Only the DeepSeek-LLM parameters are updated. This stage accelerates training (more than doubling speed) and helps the LLM adapt to the reordered visual tokens.

## 4. Evaluation and Performance

### Benchmarks
*   **OmniDocBench v1.5:** DeepSeek-OCR 2 achieved **91.09%**, a **3.73%** improvement over the DeepSeek-OCR baseline, despite using a lower maximum token count (1120 vs 1156).
*   **Reading Order:** The Edit Distance for reading order (R-order) improved significantly (lowered from 0.085 to **0.057**), validating that the model successfully learns to reorder tokens based on logic rather than just spatial position.

### Production Readiness
In practical scenarios (where ground truth is unavailable), the model showed superior stability. The repetition rate dropped from 6.25% to **4.17%** for online user logs and from 3.69% to **2.88%** for PDF pretraining data.

## 5. Insights

### Insight 1: Two-Cascaded 1D Causal Reasoning
The most profound theoretical contribution is the concept of achieving 2D reasoning through two cascaded 1D structures.
*   **Stage 1 (Encoder):** Performs "reading logic reasoning," causally reordering 2D visual information into a 1D sequence via query tokens.
*   **Stage 2 (Decoder):** Executes task-specific reasoning over this optimized sequence.
This decouples the "where to look" (visual flow) from the "what does it mean" (semantic generation), effectively bridging the gap between 2D images and 1D LLMs without enforcing arbitrary raster-scan biases.

### Insight 2: The "Prefix" Necessity
Encoder-decoder structure using cross-attention (mBART-style) failed to converge. Success required appending visual tokens as a **prefix** to the causal queries in a decoder-only style architecture. This suggests that visual tokens must remain "active" throughout the layers to facilitate effective information exchange with the causal queries.

### Insight 3: Towards Native Multimodality
The architecture suggests a path toward unified omni-modal encoding. Because the encoder is essentially an LLM accepting embeddings, one could potentially compress text, audio, and visual content within the same parameter space by simply configuring modality-specific learnable queries. This moves beyond "multimodality as alignment" (projectors) toward "multimodality as native processing."
