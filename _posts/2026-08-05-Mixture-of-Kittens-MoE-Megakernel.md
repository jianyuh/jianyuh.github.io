---
layout: post
title: "Mixture-of-Kittens: A MoE Megakernel for GB300 NVL72"
date: 2026-08-05
categories: [Systems, LLM]
tags: [MoE, Expert Parallelism, GB300, NVLink, Megakernel, Blackwell, MXFP8, DeepEP, Kernel Optimization]
---

Reading notes on:
- [Mixture-of-Kittens: our open-source MoE megakernel for NVL72s](https://cursor.com/blog/mixture-of-kittens) (Cursor)

As expert counts and expert-parallel (EP) degrees scale, the MoE layer has flipped from a structural advantage into the **primary execution bottleneck** — often over 50% of end-to-end training time, dominated by high-frequency all-to-all token routing that serializes into interconnect contention and execution bubbles. Mixture-of-Kittens (MoK) is a MoE megakernel co-designed for the single NVLink domain of **GB300 NVL72**, in the same "fuse the whole layer" spirit as the epilogue-programming work in [CODA]({% post_url 2026-07-14-CODA-GEMM-Epilogue-Programming %}) and the communication-library lessons from [NCCL GIN & MSCCL++]({% post_url 2026-06-24-NCCL-GIN-and-MSCCLpp-GPU-Communication %}).

## 1. The NVL72 Bottleneck

The GB300 NVL72 abandons discrete fat-tree topologies for a single-hop, **non-blocking NVLink fabric**: the 72-GPU rack is one unified NVLink domain — effectively a distributed shared-memory clique with **1.8 TB/s per-GPU bidirectional bandwidth**. That enables fine-grained compute/comm overlap and specialized kernels, but it exposes two asymmetries:

- **Grace–GPU imbalance.** Blackwell GPUs finish streams faster than the integrated Grace CPUs can launch kernels and manage runtime metrics. This "GPU starvation," compounded by high-latency CPU–GPU sync, means anything on the host critical path stalls the accelerators. The mandate: move routing, scheduling, and buffer management **entirely on-device**.
- **Communication contention.** Standard general-purpose primitives leave NVLink lanes underutilized during imbalanced routing.

The rest of MoK is a set of co-designs that keep the GPUs at 100% duty cycle by taking the CPU off the critical path.

## 2. Push vs. Pull: Communication Direction Co-Design

The first lever is *direction*. Push-based dispatch is standard for scattering, but pull-based dispatch fits MoE's zero-copy layouts and expert imbalance far better:

| Feature | Push-based | Pull-based |
| :--- | :--- | :--- |
| Schedule metadata | 3-column $\{src_{idx}, dst_{rank}, dst_{idx}\}$ | 2-column $\{src_{rank}, src_{idx}\}$ |
| Complexity | high (multi-pass sort/coord) | low (direct buffer alignment) |
| Local memory ops | local copies often required | **zero-copy** (direct landing) |
| Lane symmetry | unidirectional-dominant | **high bidirectional (TX/RX)** |

Push requires a 3-column table plus multiple sorting passes to keep source ranks from colliding on a destination address. Pull aligns a 2-column table directly with destination buffers, landing tokens zero-copy in contiguous memory with no cross-rank coordination.

**On-device schedule generation.** MoK builds the schedule table $\Phi$ entirely on-device at **<3% runtime overhead**, in two passes over the routing tensor $E$: first count per-expert, per-rank tokens ($T$, $M$) and prefix-sum into offsets $S$; then place each token at $\Phi[S[l] + p]$ using a stable cross-rank position $p = \sum_{r'<R}\min(M[l,r'],o) + |\{r'<r : M[l,r']>o\}|$.

**The payoff is in the signaling.** Pull requests exploit both TX and RX lanes concurrently, delivering **up to 29% higher bandwidth utilization** despite higher metadata bytes. And pull signaling is *local*: instead of waiting on up to 71 peer signals (~103 μs) as push does, the rank issuing the load waits only for its data to arrive (~18 μs) — no global barrier. Net: **82% less signaling overhead**. MoK uses an asymmetric scheme to reuse $\Phi$ across passes — forward = pull-dispatch / push-combine, backward = pull-reverse-combine / push-reverse-dispatch.

![MoK pull-based dispatch: 2-column zero-copy schedule, bidirectional NVLink lanes, local signaling replaces the 72-peer barrier](/assets/images/mok_pull_dispatch.svg)

## 3. The Goldilocks Minibatch Size $T$

The token count per transfer $T$ governs compute/comm overlap: too small and tensor cores starve, too large and the sparse-wave "tail effect" wastes work. Deriving from first principles on Blackwell — where one wave occupies all SMs at a $128\times256$ output tile, and hiding GEMM epilogue latency needs at least two waves ($2C$):

- Up/Gate projections (parallel), output tiles $\frac{128\,T\,256}{2I} \ge 2C$;
- Down projection (sequential), bottlenecked by hidden dim $H$: $\frac{128\,T\,256}{H} \ge 2C$.

Unified: $T \ge \dfrac{2C \cdot 128 \cdot 256}{\min(2I, H)}$.

**Kimi 2.5 case study** ($H=7168$, $I=2048$, Blackwell $C=148$): $\min(2\cdot2048, 7168) = 4096$, so $T \ge \frac{2\cdot148\cdot128\cdot256}{4096} = 2368$. Empirically latency stabilizes at $T \approx 2560$ (5.981 ms at 512 → 3.425 ms at 2560, flat past there), matching the bound.

## 4. Ring Token Buffers: No CPU-GPU Sync

MoE routing produces unknown per-rank token counts, which normally forces token dropping or costly CPU-GPU sync for buffer allocation. MoK's **macrobatch** architecture uses a fixed-size **ring buffer** (typically 200–500 MB) so the dispatch kernel cycles at minibatch granularity without CPU reallocation. Two refinements keep it flowing:

- **Interleaving** — combine of macrobatch $N$ overlaps dispatch of macrobatch $N{+}1$, so a slot freed by combine is immediately reused by dispatch, preventing SM stalls at the ring boundary.
- **Reversed ring** — the backward pass needs forward activations, but a finite ring overwrites early tokens. Walking tokens in **reverse** during the forward pass preserves exactly the tokens the backward pass needs first, minimizing activation replay.

## 5. Megakernel Fusion and System-Level Tricks

Fusing all MoE components into one megakernel overlaps instructions at the SM-task level and eliminates kernel-launch boundaries, with deterministic software partitioning of hardware. Key optimizations:

- **Mixed precision.** BF16 and MXFP8 both supported; the shared expert stays BF16 for stability while routed experts use MXFP8. Activation quantization is fused directly into dispatch loads, expert-grouped GEMMs, and SwiGLU — minimizing memory traffic. This is the same low-precision discipline as [NVFP4 training]({% post_url 2025-11-20-NVFP4-Train %}) and [NVFP4 in the RL loop]({% post_url 2026-07-11-NVFP4-RL %}).
- **SonicMoE gradients.** Router-weight gradients are computed from the inner product of SwiGLU activations and down-projection dgrads, so MoK never saves the full down-projection output — a large activation-memory saving.
- **Hardware-native work-stealing.** Blackwell's **Cluster Launch Control (CLC)** lets the megakernel yield SMs to higher-priority inter-rack traffic (FSDP all-gather, InfiniBand RDMA). Without CLC the MoE kernel would serialize behind inter-rack traffic and bottleneck the whole cluster.

## 6. Results

Validated on a **512-GPU** production cluster (GB300 NVL72 nodes) against DeepEP and HybridEP. Kernel-level, MoK hits ~1380 TFLOP/s on Kimi K2.7 Code shapes vs. HybridEP's ~900:

| Model shape | Mode | MoK TFLOP/s | Speedup vs. fastest baseline |
| :--- | :--- | :--- | :--- |
| Kimi K2.7 | MXFP8 FWD | ~1380 | 2.37× |
| GLM-5.2 | MXFP8 FWD | ~1310 | 2.15× |
| Qwen 3.5 | MXFP8 FWD | ~1220 | 1.95× |
| DeepSeek-V4-Pro | MXFP8 FWD | ~1410 | 2.22× |

Aggregated: **MXFP8 2.37× / 1.78× (fwd/bwd)**, **BF16 1.92× / 1.58×**. End-to-end on 512 GPUs, replacing the DeepEP-based stack lifts throughput from **760.9 → 1070.2 tokens/s/GPU — a 41% gain**.

**Takeaway.** MoK is what MoE kernels look like once you stop treating the rack as a network and start treating it as one coherent NVLink machine: pull-based zero-copy dispatch to saturate bidirectional lanes, an analytically-sized minibatch, ring buffers and on-device scheduling to take Grace off the critical path, and CLC work-stealing to coexist with inter-rack traffic. It also fits the broader expert-parallel arc — the perfect-load-balancing theorem of [Kimi K3's MoonEP]({% post_url 2026-07-18-Kimi-K3 %}), the [MoE parallel folding]({% post_url 2026-01-18-EP %}) plans, and [Megatron-Core MoE]({% post_url 2026-03-11-MoE-Megatron %}) — and marks the shift into an "agent era" of kernel development where distributed-systems architecture and low-level hardware primitives fuse in a single artifact.
