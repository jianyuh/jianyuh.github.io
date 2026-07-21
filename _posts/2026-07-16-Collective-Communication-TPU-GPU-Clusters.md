---
layout: post
title: "Inside TPU and GPU Clusters: The Anatomy of Collective Communication"
date: 2026-07-16
categories: [Systems, Distributed]
tags: [Collective Communication, TPU, GPU, All-Reduce, All-to-All, NVLink, SHARP, Torus, Fat Tree]
---

Reading notes on:
- [Inside TPU and GPU Clusters: The Anatomy of Collective Communication](https://www.aleksagordic.com/blog/collective-operations)

Training and serving large models in 2026 fundamentally relies on distributed systems, and sharding strategies map directly onto a small set of collective primitives:

- **Data Parallelism** → gradient sync via **All-Reduce**.
- **Tensor/Model Parallelism & FSDP** → **All-Gather** and **Reduce-Scatter**.
- **Expert Parallelism (MoE)** → **All-to-All**.

This complements the mental model in [How to Think About GPUs for LLM Scaling]({% post_url 2026-05-10-Scaling-Book-GPUs %}) and the low-latency transport work in [NCCL GIN & MSCCL++]({% post_url 2026-06-24-NCCL-GIN-and-MSCCLpp-GPU-Communication %}).

---

![TPU nearest-neighbor torus vs. GPU hierarchical fat tree, and their collective strategies](/assets/images/collective_topologies.svg)

## 1. TPU Topology & Latency Math

TPU clusters use **nearest-neighbor connectivity**, forming grid topologies with periodic wraparound (torus networks).

- **Dimensionality:** TPU v2, v3, v5e, v6e use a 2D torus (4 neighbors); v4p, v5p, TPU7x, and 8t use a 3D torus (6 neighbors).
- **Scale:** the largest ICI-connected block is a **TPU Pod** (superpod). A v5p pod scales to 16×20×28 chips (8,960 TPUs).
- **Wraparound penalty:** requesting a slice smaller than a full dimension (e.g. an 8×16 slice of a 16×16 v5e pod) loses the wraparound links. Without the torus, collectives run over a 1D chain instead of a ring, roughly **doubling** ring-collective time along that axis.

**Latency-vs-throughput crossover.** Using v5e unidirectional ICI bandwidth (45 GB/s) and ~1 µs per-hop latency:

$$45\ \text{GB/s} \times 1\ \mu s = 45\ \text{KB}$$

If a message chunk is ~45 KB or smaller, per-hop latency dominates. For larger payloads (~10 MB), TPUs reach near-peak collective bandwidth and standard throughput equations hold.

---

## 2. GPU Topology & Bisection-Bandwidth Math

NVIDIA clusters use **hierarchical switching networks** — typically a **Fat Tree**. Reference: the DGX H100 SuperPod (1024 GPUs).

- **Scale-up domain (node):** 8 H100 GPUs (or 72 in a GB200 NVL72) with **all-to-all** connectivity via NVSwitch/NVLink.
- **Scale-out domain:** nodes connect via InfiniBand; 32 nodes form a Scalable Unit (SU) via leaf switches, then spine switches.

A full fat tree matches upstream to downstream injection bandwidth (non-oversubscribed). Cross-partition bandwidth is limited by the smaller side of the partition:

- **Intra-node (8 GPUs, 4 vs 4):** $4 \times 450\ \text{GB/s} = 1.8\ \text{TB/s}$ unidirectional; **3.6 TB/s** bidirectional.
- **SU-level (32 nodes, 16 vs 16, 400 GB/s/node):** $16 \times 400 = 6.4\ \text{TB/s}$ uni; **12.8 TB/s** bi.
- **Cluster-level (128 nodes, 64 vs 64):** $64 \times 400 = 25.6\ \text{TB/s}$ uni; **51.2 TB/s** bi.
- **Uneven (88 vs 40):** limited by the smaller side, $40 \times 400 = 16\ \text{TB/s}$ uni.

---

## 3. Collective Mechanics

For large messages, **ring** algorithms are the default; for small, latency-dominated payloads, **tree** algorithms ($\log_2$ steps) win.

- **All-Gather:** gathers shards so each chip holds the full matrix. On a 2D torus, a 2D All-Gather uses all 4 links for a **2× speedup** over 1D.
- **Reduce-Scatter:** the dual of All-Gather — same schedule, but sums shards as they move. Because of this duality, if the forward pass uses All-Gather, the backward pass typically uses Reduce-Scatter (and vice versa).
- **All-Reduce:** composed as **Reduce-Scatter followed by All-Gather**.
- **All-to-All:** a distributed sharded transpose — the backbone of MoE dispatch/combine, where tokens grouped by source chip are routed and regrouped by destination expert.

---

## 4. GPU-Specific Optimizations

GPUs run ring algorithms by choosing a **logical ring** layered over the NVSwitch all-to-all fabric.

**SHARP (in-network reduction).** Instead of a memory-bound Reduce-Scatter + All-Gather, GPUs send partials to the switch, which reduces them (NVLink 4 NVSwitch: 400 GFLOP/s FP32) and multicasts results back.
- *Theory:* ~1.75× on an 8-GPU node (approaching 2× for large $N$).
- *Practice:* only ~30% (1.3×) — reduce/multicast pipelines don't overlap perfectly, peak bandwidth needs multi-GB messages, and multicast wastes 1/8 of node bandwidth returning data to the source GPU.

**Tree algorithms (recursive doubling/halving).** Pair GPUs in rounds, cutting steps to $\log_2(N)$ instead of $N-1$. Same ideal byte cost, but less pipeline-friendly for huge tensors — they dominate only in low-latency regimes.

**Hierarchical inter-node collectives.** Crossing InfiniBand, algorithms pipeline across hierarchy levels. For a tensor of $D$ bytes, throughput-bound time $T \approx D / BW_\text{node} = D / 400\text{e}9$; factoring intra-node stages, $T \approx \max(D/BW_\text{gpu}, D/BW_\text{node})$.

**Real-world routing nuances.**
1. **Rail optimization:** peak node bandwidth needs rail-aware rank placement so traffic stays balanced across network rails.
2. **Megatron TP:** reducing weights sharded over a tensor-parallel axis does *not* All-Reduce across those ranks — it reduces element-wise across the **data-parallel replicas** of each TP shard.
3. **Sparse MoE on NVL72:** dense All-to-All equations assume balanced routing. In a 72-GPU node selecting 8 experts, a token reaches only ~11% of GPUs ($8/72$) — the pattern is ragged and sparse, not a dense transpose. This is exactly the distributed-MoE regime that shapes ASIC-scale designs like [LongCat-2.0]({% post_url 2026-07-03-LongCat-2.0 %}) and the sequence-parallel collectives in [Scaling Video Training with Sequence Parallelism]({% post_url 2026-06-14-Scaling-Video-Training-SP %}).
