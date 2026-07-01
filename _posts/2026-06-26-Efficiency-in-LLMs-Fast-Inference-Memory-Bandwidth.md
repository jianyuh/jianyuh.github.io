---
layout: post
title: "Efficiency in LLMs: Mastering Fast Inference and Memory Bandwidth"
date: 2026-06-26
categories: [LLM, Inference, Systems]
tags: [Inference, Memory Bandwidth, Roofline, FlashAttention, PagedAttention, RadixAttention, Speculative Decoding, Quantization, GPTQ, AWQ, MLA, KV Cache, Sparse Attention]
---

Reading notes distilling the core technical realities, derivations, and algorithmic insights from **Alex Smola's** 2026 Columbia Machine Learning Summer School presentation, [*Efficiency in LLMs: Hardware, Serving, and Compression — a Practitioner's Tour of Fast Inference*](https://alex.smola.org/posts/45-mlss-efficiency/main.pdf).

The central thesis of modern LLM inference is brutally simple: **Decode is a memory-traffic problem, not a math problem.** We are in a regime where the chip starves waiting for data. Every trick in the modern LLM serving stack is ultimately an attempt to **move fewer bytes**.

---

## 1. The Physics of Fast Inference: The Arithmetic Intensity Ridge

To understand LLM inference, you must understand the **Arithmetic Intensity Ridge**. Arithmetic intensity is the ratio of FLOPs performed to bytes moved.

*   **The Ridge Point:** On a 2026 **Blackwell B200** (4,500 TFLOP/s FP8, 8 TB/s memory bandwidth), the hardware can perform $\sim 562$ operations in the time it takes to fetch a single byte from HBM. (For the roofline mental model and Blackwell's tensor-core details, see [How to Think About GPUs for LLM Scaling]({% post_url 2026-05-10-Scaling-Book-GPUs %}) and [Blackwell SM100]({% post_url 2026-04-12-blackwell-sm100 %}).)
*   **Prefill vs. Decode:**
    *   **Prefill (Compute-Bound):** Streams the weights once to process $L_{ctx}$ tokens. Intensity is $\sim L_{ctx}$. This sits on the flat "compute ceiling" of the roofline model.
    *   **Decode (Bandwidth-Bound):** Every generated token requires streaming the *entire* weight matrix and the *entire* KV cache. Intensity is $< 1$. This sits far down the sloped "bandwidth wall" of the roofline, operating at $< 1\%$ of peak FLOPs. For **Qwen3-8B**, generating one token requires moving **22 GB of data for only 16 GFLOPs** of math.

**The Shoreline Problem.** Hardware scaling is *worsening* this bottleneck. (This same asymmetric-scaling pressure drives the FlashAttention-4 redesign discussed in [FlashAttention-4 and Asymmetric Hardware Scaling]({% post_url 2026-03-06-FA4 %}).) A chip's compute capacity scales with its silicon area ($O(n^2)$), but its I/O bandwidth to HBM is limited by the PHYs and SerDes on its perimeter edge ("shoreline", $O(n)$). Consequently, compute is scaling at $\sim 4\times$/generation, bandwidth at $\sim 2\times$, and memory capacity at $< 1.4\times$.

### FlashAttention: SRAM Tiling to the Rescue

Attention inherently requires materializing an $N \times N$ matrix $S = QK^T$, resulting in massive HBM reads/writes ($\Theta(Nd + N^2)$). FlashAttention solves this by tiling $Q$, $K$, and $V$ blocks through the ultra-fast, tiny on-chip SRAM (which is $\sim 10\times$ faster than HBM) and avoiding the materialization of $S$ entirely.

**The Online Softmax Derivation.** To compute the softmax without seeing the whole row at once, we track a running maximum $m$ and sum $\ell$. For an incoming tile of scores $x$ and values $V$:

1.  $m' = \max(m, x)$
2.  $\ell' = e^{m-m'}\ell + e^{x-m'}$
3.  $O' = e^{m-m'}O + e^{x-m'}$
4.  $V_{out} = O / \ell$

This drops HBM traffic to $\Theta(N^2 d^2 / M)$ (where $M$ is SRAM size), utilizing up to **85% of peak GPU FLOP/s**.

---

## 2. Serving: Keeping the GPU Fed

If a GPU is idle or recomputing data, you are losing money. Modern serving engines (like **vLLM** — see [vLLM V1 internals]({% post_url 2025-11-30-vLLM %}) — and **SGLang**) rely on orchestration to amortize costs. Splitting these phases across hardware is its own topic; see [Disaggregate Prefill and Decoding]({% post_url 2025-03-30-prefill-decoding-disagg %}) and [Nvidia Inference: Disaggregated Decode]({% post_url 2026-03-30-nvidia-inference %}).

### PagedAttention & RadixAttention

Before vLLM, memory fragmentation in the KV cache limited batch sizes, as systems reserved contiguous blocks for maximum sequence lengths (wasting up to **80%** of space). **PagedAttention** treats the KV cache like virtual memory, mapping logical token sequences to non-contiguous 16-token physical blocks via a block table, dropping fragmentation to near-zero.

**RadixAttention** takes this further by turning the KV cache into a **radix tree** to store reused prefixes (system prompts, few-shot examples).

*   Each node edge represents a token sequence.
*   New requests walk the tree to find the deepest cached match, guaranteeing an instant cache hit (skipped prefill).
*   An LRU policy evicts the coldest leaves.

### Speculative Decoding: Maximal Coupling

Because decode is bandwidth-bound and leaves FLOPs idle, we can use a cheap "draft" model to propose $k$ tokens, and the "target" model to verify all $k$ in a single forward pass. (See the dedicated [speculative decoding primer]({% post_url 2024-12-15-speculative-decoding %}) and the diffusion-drafter follow-up, [DFlash & DSpark]({% post_url 2026-06-29-DFlash-DSpark-Diffusion-Speculative-Decoding %}).) Because this single pass streams the weights once, it costs the same bandwidth as verifying one token.

**The Math (Maximal Coupling).** We want the final output distributed strictly as the target $p$, but we sample from the draft $q$.

1.  Accept the draft token $x$ with probability $\min\left(1, \frac{p(x)}{q(x)}\right)$.
2.  If rejected, resample from a corrected distribution: $x \sim \frac{(p-q)^+}{\sum (p-q)^+}$.
3.  The overall acceptance rate $a$ is $1 - TV(p, q)$ (Total Variation distance).
4.  Expected tokens generated per step is $\frac{1-a^{g+1}}{1-a}$ (where $g$ is draft tokens).

This mathematically guarantees the output distribution is unaltered, avoiding any quality degradation.

---

## 3. Weight Compression: Squeezing the Math

Fewer bits per weight means fewer bytes dragged across the HBM bus. While native formats like **FP4** (which provides just 16 values, utilizing a 1-bit mantissa) are incredibly fast, they require block-level scaling (e.g., E4M3 scales) to handle outliers without massive clipping. (On training natively in these low-precision formats, see [NVFP4]({% post_url 2025-11-20-NVFP4-Train %}) and [MXFP8 Training]({% post_url 2025-12-07-MXFP8-Train %}).)

For models trained in higher precision, we must compress via **Post-Training Quantization (PTQ)** (for a survey of open-model strategies, see [Open Source Model Quantization Strategies]({% post_url 2026-01-23-Quantization %})).

### GPTQ: The 2nd-Order Hessian Approach

Simple **Round-To-Nearest (RTN)** quantization fails because an outlier forces a wide grid scale, crushing the other 99% of weights into just a few effective levels. **GPTQ** instead frames quantization as a least-squares problem over a calibration set $X$:

$$\min \|WX - \hat{W}X\|_2^2 = \text{tr}\left[(W - \hat{W})^T XX^T (W - \hat{W})\right]$$

Here, $Q = XX^T$ acts as the **Hessian** (curvature). We split the error $\delta = w - \hat{w}$ into quantized coordinates ($\delta_q$) and free coordinates ($\delta_f$). Using the **Schur Complement**, we solve for the optimal adjustment to the unquantized weights:

$$\delta_f = -Q_{ff}^{-1} Q_{fq} \delta_q$$

This means the rounding error of one column is pushed onto the remaining unquantized weights, compensating for the error on the fly.

*Insight:* **AWQ** takes the opposite approach. Instead of fixing mistakes after they happen, it identifies the 1% of salient weights (based on activation variance) and multiplies them by a scale $s$, while dividing the corresponding activations by $s$ ($Wx = (W \cdot s)(x / s)$). This shrinks the salient weight into the grid, preserving it without blowing up the scale.

---

## 4. KV Cache Compression: The 1M Token Challenge

At 1M tokens, the KV cache dwarfs the model weights. For **Qwen3-8B**, the weights take 16 GB, but the KV cache for a 1M token context demands **147 GB**. Scaling requires attacking this from multiple axes:

### 1. Shrink the State: Multi-Head Latent Attention (MLA)

Standard MHA caches $K$ and $V$. **MLA** (used in [DeepSeek models]({% post_url 2024-12-26-deepseek-v3 %})) never materializes them. Instead, it compresses them into a single 576-dimensional latent vector $c^{KV} = W_{DKV} h$. During generation, it recovers the keys and values linearly:

$$k^C = W_{UK} c^{KV}$$
$$v^C = W_{UV} c^{KV}$$

Crucially, it splits the rotary positional embeddings (RoPE) into a separate 64-dim uncompressed vector $k^R$, because positional data resists low-rank compression. This drops the cache size per token from $\sim 2300$ KB (in standard MHA) to just **70 KB**, a **98% reduction**.

### 2. Shrink the Bits: KIVI and Rate-Distortion

You can quantize the cache, but activations have structural quirks. **KIVI** notices that $K$ has extreme outliers along specific *channels* (due to RoPE), so it quantizes $K$ **per-channel**. $V$ lacks this structure, so it is quantized **per-token**. This asymmetry allows **2-bit caching** with near-lossless quality.

From an information-theory perspective, scalar quantization of Gaussian data wastes exactly **0.254 bits/sample** compared to the optimal Rate-Distortion bound $D(R) = \sigma^2 2^{-2R}$. Methods like **TurboQuant** randomly rotate the KV vectors to make their coordinates perfectly Gaussian, allowing scalar quantizers to exactly hit the $R(D)$ limit without calibration.

### 3. Read Less: Sparse Attention

Even if stored compactly, decoding 1M tokens requires scanning 1M states. **Sparse Attention** algorithms (like **NSA, MoBA, or DSA**) learn to route queries to a small top-$k$ subset of blocks ($k \ll L$), dropping read times by an order of magnitude. (DSA in particular powers long-context serving in [DeepSeek-V3.2]({% post_url 2025-12-01-DeepSeek-V3.2 %}).)

By stacking these orthogonal techniques (e.g., MLA $\to$ 4-bit Quantization $\to$ NSA sparse reads), the seemingly impossible 147 GB / 1M token cache is compacted into just a few GB of storage and $\sim 0.5$ GB/token of memory traffic, making it entirely feasible on a single GPU.
