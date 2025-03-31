---
layout: post
title: "Disaggregate Prefill and Decoding"
date: 2025-03-30
categories: [LLM inference]
tags: [disagg]
---

## Reference List

- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079).

- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670)
- [NVIDIA Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/?ncid=so-link-889273)

## Summary

Disaggregating the prefill and decoding phases in LLM inference enables speedups by optimizing each phase for its distinct computational demands, resource requirements, and bottlenecks.

### 1. Phase-Specific Computational Characteristics
- Prefill Phase (Compute-Bound):

Processes the entire input prompt in parallel, requiring heavy matrix operations (e.g., attention across all tokens).

Benefits from high-throughput compute resources (e.g., powerful GPUs/TPUs) to maximize parallel processing.

- Decoding Phase (Memory-Bound):

Generates tokens sequentially, with each step dependent on prior outputs.

Limited by memory bandwidth (loading model weights repeatedly) rather than raw compute.

### 2.Resource Specialization
- Prefill: Allocate high-FLOPS devices (e.g., A100/H100 GPUs) to exploit parallelism.

- Decoding: Use memory-optimized hardware (e.g., inference chips with large caches) or techniques like KV caching to reduce redundant memory access.

### 3. Batching Efficiency
- Prefill: Batch multiple prompts to maximize GPU utilization (large, static workloads).

- Decoding: Use smaller, dynamic batches tailored to latency-sensitive token generation, avoiding interference from prefill’s bulkier computations.

### 4. Memory and Latency Optimization
- Prefill: Precompute and cache attention keys/values (KV cache) during prompt processing, reused in decoding to avoid redundant computation.

- Decoding: Focus on minimizing latency by streamlining memory access (e.g., keeping weights in faster cache memory).

### 5. Asynchronous and Scalable Workflow
- Decouple prefill and decoding into separate systems, enabling:

- Overlap: Start decoding immediately after prefill completes, hiding latency.

- Scalability: Independently scale prefill (throughput-oriented) and decoding (latency-oriented) resources based on demand.

### 6. Reduced Contention
- Avoid resource competition (e.g., GPU cores vs. memory bandwidth) by isolating phases, ensuring neither starves the other.

## Key Technical Insights

- KV Caching: Storing precomputed attention states during prefill drastically reduces decoding overhead.

- Hardware Fit: Match compute-heavy prefill to GPUs and memory-bound decoding to optimized inference accelerators.

- Pipeline Parallelism: Overlap prefill for one request with decoding for another, improving overall throughput.

## Example Workflow
- Prefill Server: Processes a batch of prompts in parallel, generates KV caches.

- Decoding Server: Uses cached KV states to generate tokens efficiently, even with low batch sizes.

## Result
- Higher Throughput: Efficient batching and parallelism in prefill.

- Lower Latency: Optimized memory access and specialized hardware for decoding.

- Cost Efficiency: Right-sizing resources for each phase reduces idle time and operational costs.

