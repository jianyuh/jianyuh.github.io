---
layout: post
title: "The State of Scaling LLM Inference (NV vs. AMD)"
date: 2026-02-24
categories: [AI]
tags: [LLM Inference, GPU, NVIDIA, AMD, SemiAnalysis]
---

Reading the following article:
- [InferenceX v2: NVIDIA Blackwell Vs AMD vs Hopper - Formerly InferenceMAX](https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs)

The latest release of the open-source InferenceX v2 benchmark (formerly InferenceMAX) offers a masterclass in the engineering and economic realities of large-scale AI inference. Covering nearly 1,000 frontier GPUs across NVIDIA (Blackwell, Hopper) and AMD (MI300/320/350 series), the report moves past static benchmarks to evaluate modern, complex production deployments.

Below are technical reading notes on distributed inference, hardware showdowns, and the unit economics of serving large language models.

### 1. The "Holy Trinity" of Production Inference

Top-tier AI labs are no longer running simple, single-node inference; they are composing three cutting-edge optimizations to maximize throughput and minimize latency.

*   **Disaggregated Prefill (PD Disaggregation):** The lifecycle of an LLM request involves a compute-heavy *prefill* phase (processing all prompt tokens in parallel) and a memory-bandwidth-bound *decode* phase (generating tokens one by one). When sharing the same GPU engine, bursty prefill requests stall in-flight decode batches, destroying efficiency. Disaggregated serving separates these phases onto dedicated GPU pools, allowing each to be scaled and optimized independently.
*   **Wide Expert Parallelism (WideEP):** Modern Mixture-of-Experts (MoE) models, like DeepSeek R1 with its 671B total parameters and 256 routed experts, demand smart weight distribution. Tensor Parallelism (TP) works well for small batch sizes but requires expensive all-reduce communication per layer. WideEP instead distributes experts across multiple nodes (e.g., 4 experts per GPU on a 64-GPU cluster instead of 32 on an 8-GPU node), massively increasing arithmetic intensity (tokens per expert) and leveraging the aggregate HBM bandwidth of the entire cluster.
*   **Multi-Token Prediction (MTP):** Instead of using a separate draft model for speculative decoding, MTP adds auxiliary prediction heads to the main model, allowing it to propose and verify multiple tokens in a single forward pass. The benchmark shows MTP provides massive cost savings—slashing the cost on GB300 Dynamo TRT from ~2.35 to ~0.11 dollar per million tokens at 150 tok/sec/user—with no measurable accuracy impact (verified via GSM8k).

### 2. NVIDIA’s Rack-Scale Dominance: The NVLink Advantage

NVIDIA’s Blackwell architecture, specifically the rack-scale GB200/GB300 NVL72, is a beast. The benchmark reveals Blackwell delivering up to **100x better performance (FP8/FP4)** compared to a strong H100 baseline. As the authors note, Jensen Huang actually "underpromised and overdelivered" on Blackwell’s capabilities.

The secret sauce is the **scale-up network domain**. WideEP requires bandwidth-intensive all-to-all communication to route tokens to their corresponding experts. A standard server connects 8 GPUs via NVLink, forcing any larger cluster to rely on slower InfiniBand/Ethernet for cross-node traffic. The NVL72 rack, however, connects all 72 GPUs in a single NVLink domain with 900 GB/s uni-directional bandwidth per GPU. This allows WideEP to scale without hitting a network bottleneck, letting NVIDIA dominate total token throughput.

### 3. AMD’s "Composability" Crisis

On paper, AMD’s hardware is highly competitive. The MI355X performs admirably in FP8 disaggregated inference, even edging out the B200 at certain mid-range interactivity levels.

However, AMD’s Achilles’ heel remains its software stack. The biggest issue highlighted in the report is **composability**: while AMD can achieve strong results with individual optimizations in isolation, combining Disaggregated Prefill, WideEP, and FP4 results in severely subpar performance. The MI355X gets decisively beaten by the B200 when these state-of-the-art techniques are layered together.

Furthermore, AMD suffers from ecosystem fragmentation. Instead of fully committing engineering resources to upstream open-source engines like vLLM and SGLang, AMD has historically relied on isolated forks and recently launched "ATOM"—an engine that lacks crucial production features like disaggregated serving and CPU KV offloading, resulting in zero production adoption.

### 4. Economic Insight: Demystifying "Fast Mode"

The report offers a sharp economic breakdown of features like Anthropic’s recently released Claude "Fast Mode" (which charges roughly 6–12x the price for 2.5x the speed).

There is no "secret hardware" powering this. It is simply a manifestation of the **latency-throughput tradeoff curve**. To achieve higher interactivity (faster tokens per second per user), inference engines must run at smaller batch sizes. Because the hourly fixed cost of the GPUs remains the same, smaller batches mean fewer total tokens generated per hour, so the cost per token must rise to cover the hardware. "Fast Mode" is the hyperscaler sliding you up the Pareto curve—from a "metro bus" (high batch, high throughput, low cost) to a "race car" (low batch, low latency, high cost).

### 5. Future Outlook

The gap between NVIDIA and AMD is currently defined by software maturity and rack-scale interconnects. The InferenceX team plans to transition to multi-turn, real-world agentic coding datasets (like WildChat) and enable CPU KV cache offloading. This will further stress-test memory architectures, where AMD’s larger HBM capacity (288 GB on MI355X vs. 192 GB on B200) might finally shine—provided AMD can fix its software composability issues.
