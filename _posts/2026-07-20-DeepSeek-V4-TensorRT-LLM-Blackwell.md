---
layout: post
title: "DeepSeek-V4 on Blackwell: Model-Specific and Agentic Optimizations in TensorRT-LLM"
date: 2026-07-20
categories: [Systems, Inference]
tags: [DeepSeek-V4, TensorRT-LLM, Blackwell, GB300, MoE, KV Cache, Agentic, Speculative Decoding]
---

Reading notes on:
- [DeepSeek-V4 on NVIDIA Blackwell: Model-Specific and Agentic-Workload Optimizations in TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.md)

This is the serving-side companion to DeepSeek-V4's architecture and training write-ups — see [DeepSeek-V4 Architecture & Training]({% post_url 2026-04-26-DeepSeek-V4-Arch-Train %}) and [DeepSeek-V4 Infra]({% post_url 2026-04-25-DeepSeek-V4-Infra %}). NVIDIA's co-design targets both fixed-shape throughput and dynamic, multi-turn agentic workloads on GB300. Through kernel fusions, algorithmic tricks like Temporal Correlation Top-K (GVR), and multi-level KV-aware routing, they reached a **64.5% throughput increase** for fixed-shape work (984 → 1,618 tokens/s/GPU), and **57.5 CPG at SLO20 / 19.2 CPG at SLO60** on Artificial Analysis AgentPerf.

---

![DeepSeek-V4 hybrid attention: SWA, CSA, and HCA interleaved by compression ratio](/assets/images/dsv4_hybrid_attention.svg)

## 1. DeepSeek-V4 Architectural Paradigms

Two variants: **Flash** (284B total / 13B active) and **Pro** (1.6T total / 49B active). TensorRT-LLM natively implements the heterogeneous sublayers.

**Hybrid attention (SWA, CSA, HCA)** — interleaved by a layer's compression ratio:
- **SWA (ratio 0):** dense attention over the most recent 128 raw tokens.
- **CSA (ratio 4):** recent 128 raw tokens + 4× compressed history; an Indexer selects entries via Top-K (K=512 Flash, 1024 Pro).
- **HCA (ratio 128):** 128 raw tokens + *all* 128× compressed history, densely, no Indexer.

TensorRT-LLM uses a unified **dual-pool MLA** kernel: it processes the 128-token window from one pool and compressed entries (Top-K for CSA, full stream for HCA) from a second, combining both in a single online-softmax reduction. A per-query `topk_lens` vector drives the variable workload while fixed-width index buffers preserve CUDA Graph compatibility. This is the same sparse-attention lineage explored in [IndexCache: Cross-Layer Index Reuse]({% post_url 2026-06-27-IndexCache-Cross-Layer-Index-Reuse %}).

**Online Sequence Compressor** — token-level, softmax-gated pooling produces history for CSA/HCA. Each token's projected KV entry gets a learned compression-weight vector plus a position bias; the group reduces to one entry via a softmax-weighted sum. CSA uses window $m=4$ (overlapping, $2m$ receptive field, dual streams for main KV and Indexer keys); HCA uses disjoint $m'=128$ windows.

**mHC (Manifold-constrained Hyper-Connections)** — every attention/MoE sublayer is wrapped in an mHC module that expands the residual stream 4×, mixes it via a doubly-stochastic matrix (20 Sinkhorn-Knopp iterations), and writes back through a post-mapping; an HC head collapses the stream before the LM head.

**MoE** — 6 routed + 1 shared expert per token out of 256 (Flash) / 384 (Pro). Layers 1–3 use deterministic hash routing; deeper layers use a learned gate with Sqrt-Softplus affinities and a score-correction bias.

**Cache lifecycle heterogeneity** — sliding-window cache (SWA KV + short-lived compressor state, recyclable when the 128- or 8-token windows expire) vs. persistent compressed caches (4× and 128× finalized entries that grow slowly). This extends the hybrid KV-cache theme from [DeepSeek-V4 Infra]({% post_url 2026-04-25-DeepSeek-V4-Infra %}).

---

## 2. Part I — Micro-optimizations & Kernel Fusions (Fixed-Shape)

To hit 1,618 tokens/s/GPU on the 8K/1K InferenceX workload:

- **FP8 dataflow:** for Pro, inverse RoPE and 1×128 E4M3 quantization are fused into the FMHA correction epilogue, writing FP32 scales and E4M3 outputs directly for the `o_a_proj` BMM — the epilogue-fusion philosophy of [CODA]({% post_url 2026-07-14-CODA-GEMM-Epilogue-Programming %}) applied in production.
- **Top-K via GVR (Gated Value Retrieval):** exploits temporal correlation — reuse the previous step's indices as candidates, refine against current scores. Up to **2.17× speedup** over radix-selection.
- **Multi-stream overlap:** the CSA prologue launches the main-attention Compressor and query-independent Indexer early on dedicated streams, overlapping shared projections and Q norms.
- **mHC fusion:** adjoining post-mappings and pre-mappings fuse into shape-specialized FMA/MMA kernels; the next sublayer's RMSNorm folds into the fused epilogue, removing a memory round-trip.
- **DeepGEMM MegaMoE:** symmetric memory + in-kernel sync fuse dispatch, twin expert GEMMs, SwiGLU, and combine (MXFP8 activations × MXFP4 weights); input prep collapses from 6 ops to 1.
- **SWA scratch reuse** and **PDL (Programmatic Dependent Launch):** a consumer grid launches before its producer retires, eliminating host-device boundaries.

---

## 3. Part II — Agentic Workloads (Multi-Turn, High-Reuse)

AgentPerf measures Concurrency Per GPU (CPG) under TTFT and decode-speed SLOs. Heavy prefix reuse moves the bottleneck from raw throughput to routing, cache lifecycle, and control-plane serialization.

**Two-level context-locality routing** — prefix reuse only helps if turns hit the exact worker/rank holding the KV.
- **Instance-level (CTX servers):** map stable session IDs to servers; for unknown sessions, incremental tokenization + a block-match score *penalized by load*, so shared system prompts don't dilute conversation-specific routing.
- **Rank-level (ADP ranks):** ranks probe local radix trees with a cache salt; score `(Total Input − Matched Prefix) + Load Penalty`, so long cached prefixes trump generic load-balancing unless a rank hits capacity.

**Cache lifecycle tuning** — the bug: sliding-window state was retained for the entire prompt history across turns and CTX→GEN transfers. The fix: a `per_request` policy that drops sliding-window blocks older than the active window on completion/transfer.

**Eradicating host overhead** — with minimal GPU work on cached prefills, Python control-plane tasks dominate TTFT. Fixes: packed block-level C++ hashing, skipping collectives when no tasks pend, no re-tokenization (CTX forwards IDs), removed deep copies, and replacing Python JSON with `pydantic-core` / `msgspec` MessagePack to bypass GIL-bound serialization.

---

## 4. Insights

- **From FLOPs to memory-lifecycle governance:** differentiating cache lifetimes (transient 128-token windows vs. persistent 128× compressed history) turns the KV manager from a linear allocator into a multi-pool, lifecycle-aware memory hypervisor.
- **System co-design over microbenchmarks:** the gains came from preventing GPU starvation (PDL, C++ orchestrator, GVR), not one magic kernel.
- **What's next:** a fully C++ KV manager (dropping the GIL under high prefix-reuse), full NVFP4 execution, and native integration of DeepSeek-V4's DSpark speculative decoding — the block-diffusion / semi-autoregressive drafting analyzed in [DFlash & DSpark]({% post_url 2026-06-29-DFlash-DSpark-Diffusion-Speculative-Decoding %}).
