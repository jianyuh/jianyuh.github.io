---
layout: post
title: "DeepSeek-V4 Infra: Overlap, TileLang, FP4 QAT, and Hybrid KV Cache"
date: 2026-04-25
categories: [LLM]
tags: [DeepSeek, MoE, FP4, TileLang, Expert Parallelism, KV Cache, Infra]
---

Paper: [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf).

DeepSeek-V4 pushes million-token context and 1.6T-parameter scale through a stack of pioneering infrastructure optimizations—from kernel-level scheduling and a custom DSL, to FP4 quantization-aware training, to a hybrid KV cache redesign for non-uniform attention. Building on the [DeepSeek-V3 architecture]({% post_url 2024-12-26-deepseek-v3 %}) and the [DeepSeek-V3 ISCA hardware lessons]({% post_url 2025-05-15-DeepSeek-V3-ISCA %}), V4's Section 3 (General Infrastructures) is the most technically dense part of the report related to infra. Here are reading notes.

---

## 1. Fine-Grained Communication–Computation Overlap in Expert Parallelism

Mixture-of-Experts relies heavily on Expert Parallelism (EP), which typically strains interconnect bandwidth due to massive all-to-all communications.

*   **The Core Insight:** Profiling reveals that within a single MoE layer, the total communication time is strictly less than the computation time. If communication and computation are fully fused into a single pipelined kernel, the interconnect bandwidth ceases to be the bottleneck.
*   **Wave-Based Scheduling:** DeepSeek splits and schedules experts into "waves". Once a wave of experts finishes its dispatch communication, its computation begins immediately. In steady state, the computation of the current wave, the token transfer for the next wave, and the result-sending of completed experts occur concurrently. This yields up to a **1.96× speedup** in latency-sensitive RL rollouts.

### Bandwidth Requirement Derivation

To achieve full overlap, the compute-to-bandwidth ratio $C/B$ must be less than or equal to the computation-to-communication volume ratio $V_{comp}/V_{comm}$:

*   For DeepSeek-V4-Pro, one token-expert pair requires $6hd$ FLOPs (SwiGLU gate, up, and down projections).
*   Communication requires $3h$ bytes (FP8 Dispatch + BF16 Combine).
*   Therefore the requirement simplifies to:

$$\frac{C}{B} \leqslant 2d = 6144 \text{ FLOPs/Byte}$$

**Insight:** Exactly 1 GBps of interconnect bandwidth is sufficient to hide 6.1 TFLOP/s of compute. The authors advise hardware vendors to target this *specific balance point* rather than unconditionally scaling bandwidth.

---

## 2. Kernel Development with TileLang

To avoid maintaining hundreds of fine-grained Torch ATen operators, DeepSeek uses **TileLang**, a Domain-Specific Language that balances development speed with runtime efficiency.

*   **Host Codegen for CPU Overhead:** CPU-side orchestration (Python-based validation and marshaling) bottlenecks highly optimized kernels. DeepSeek co-generates a lightweight host launcher at the IR level using the TVM-FFI framework. This bypasses Python execution paths, dropping CPU validation overhead from hundreds of microseconds to **under $1\mu s$**.
*   **SMT-Solver-Assisted Integer Analysis:** TileLang integrates the Z3 SMT solver, translating integer expressions into Quantifier-Free Non-Linear Integer Arithmetic (QF_NIA). This tackles non-linear reasoning over variable tensor shapes, unlocking advanced vectorization and memory-hazard detection optimizations with minimal compilation time overhead.
*   **Bitwise Reproducibility:** Fast-math optimizations are disabled by default. The compiler relies on strict IEEE-754 semantics and explicit layout annotations to ensure accumulation orders exactly match handwritten CUDA baselines.

---

## 3. Batch-Invariant and Deterministic Kernels

Guaranteeing identical outputs regardless of batch positioning or parallelization strategies is vital for debugging and post-training consistency.

### Batch Invariance in Attention

Traditional "split-KV" methods (distributing a sequence's attention across multiple SMs) break batch invariance because the accumulation order changes. DeepSeek built a **dual-kernel strategy**:

1.  The first kernel computes attention for fully occupied waves on a single SM.
2.  The second kernel uses distributed shared memory to handle the partially-filled final wave across multiple SMs, strictly preserving the accumulation order of the first kernel while avoiding wave-quantization stalls.

### Determinism in Backpropagation

Non-associativity of floating-point atomic additions (like `atomicAdd`) introduces randomness:

*   **Attention BWD:** Separate accumulation buffers per SM, followed by a deterministic global reduction.
*   **MoE BWD:** Token-order pre-processing and buffer isolation across ranks resolve non-deterministic write-position negotiation.

This connects directly to the broader push for [reproducible numerics in RL infra]({% post_url 2025-06-23-FP32-Reasoning %}), where determinism is now a first-class infrastructure requirement rather than a debugging convenience.

---

## 4. FP4 Quantization-Aware Training

DeepSeek introduces **MXFP4** quantization for MoE expert weights and the Query-Key (QK) indexer path, drastically cutting memory consumption and accelerating extreme long-context attention. (See also the broader [NVFP4 training]({% post_url 2025-11-20-NVFP4-Train %}) note for context.)

*   **Lossless Dequantization:** FP32 master weights are quantized to FP4 (E2M1) but **dequantized to FP8 (E4M3) for compute**. Because FP8 has two extra exponent bits, it fully absorbs the fine-grained scale factors of FP4 sub-blocks, making dequantization *mathematically lossless*.
*   **STE-Based Backward:** During backward passes, gradients are computed against the FP8 weights and propagated via a Straight-Through Estimator (STE) directly to FP32 master weights, removing the need for a separate FP4 backward pipeline.

---

## 5. Innovations in the Training Framework

### Muon Optimizer Integration

Muon requires full gradient matrices, conflicting with the traditional ZeRO optimizer which partitions element-wise. DeepSeek uses a **hybrid ZeRO bucket assignment**:

*   Dense parameters are constrained by a maximum ZeRO size and assigned via a knapsack algorithm (padding incurs $<10\%$ memory overhead).
*   MoE parameters are optimized independently, with gradients synchronized using **BF16 stochastic rounding** to halve communication volume.

### Context Parallelism for Compressed Attention

Context Parallelism (CP) splits sequences across GPUs, but Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) need to compress blocks of $m$ consecutive tokens that may straddle GPU boundaries. DeepSeek's two-stage solution:

1.  Rank $i$ sends its trailing $m$ uncompressed tokens to rank $i+1$, which compresses them locally.
2.  An all-gather operation uses a fused select-and-pad operator to align compressed blocks globally.

### Extended Autograd Checkpointing

Instead of standard layer-level recomputation, DeepSeek uses **TorchFX** to trace the computation graph and isolate minimal *recomputation graphs* at the tensor level. This automatically deduplicates shared storage pointers and avoids GPU memory copies entirely.

---

## 6. Inference Framework and KV Cache Management

Hybrid attention—mixing sparse, heavy compressed, and sliding-window attention—breaks the fundamental rules of traditional PagedAttention.

### Heterogeneous Cache Layout

*   **State Cache:** Sliding Window Attention (SWA) and trailing uncompressed tokens act as state-space models. They are allocated a fixed-size, dynamically assigned cache pool.
*   **Classical Cache:** High-performance attention kernels require aligned blocks. DeepSeek co-designs the kernel to process blocks aligned to $\text{lcm}(m, m')$, where $m$ and $m'$ are the compression rates for CSA and HCA—natively satisfying PagedAttention alignment rules.

### On-Disk Storage for Shared Prefixes

To prevent recomputing shared system prompts or documents, CSA/HCA compressed caches are dumped to SSDs. Because the SWA cache is uncompressed and **8× larger**, DeepSeek offers three configurable strategies:

1.  **Full Caching** — zero recompute, high I/O cost.
2.  **Periodic Checkpointing** — middle ground.
3.  **Zero Caching** — the last $n_{win} \times L$ tokens are simply recomputed on the fly using the cached CSA/HCA data.

---

## 7. Takeaways

1.  **The bottleneck is now scheduling, not bandwidth.** With wave-based EP overlap, 1 GBps of interconnect hides 6.1 TFLOP/s of compute. Hardware vendors should target the $C/B$ balance point rather than scaling bandwidth blindly.
2.  **DSLs have caught up with handwritten CUDA.** TileLang + TVM-FFI + Z3 + IEEE-754 strictness deliver bitwise-reproducible kernels with sub-microsecond launch overhead, removing one of the long-standing reasons to write CUDA by hand.
3.  **FP4 is viable for training, not just inference.** The "FP4 storage, FP8 compute" trick exploits FP8's wider exponent to absorb FP4 scale factors losslessly—pairing well with STE to avoid a separate FP4 backward path.
4.  **Hybrid attention requires a hybrid KV cache.** When the attention pattern itself is heterogeneous (sparse + heavy-compressed + sliding-window), PagedAttention's uniform block assumption breaks. The fix is to align kernel blocks to $\text{lcm}$ of compression rates and to split state-style vs. classical caches into separate pools.
