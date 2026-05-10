---
layout: post
title: "How to Think About GPUs for LLM Scaling"
date: 2026-05-10
categories: [LLM, Hardware]
tags: [GPU, TPU, NVLink, SHARP, Roofline, Parallelism, MoE, DeepSeek, LLaMA]
---

Read [How to Scale Your Model — Section 12: How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus). The book is a systems-level walkthrough of LLM scaling on TPUs; Section 12 is the bonus chapter that re-derives the same rooflines on NVIDIA GPUs (H100 / B200 / GB200 NVL72) and contrasts them with TPUs at every level — chip, node, scale-out network, and collectives.

---

## 1. Chip Level: SMs, Tensor Cores, and Why GPUs Look Like TPUs Now

A modern ML GPU is "a bunch of compute cores that specialize in matmul (SMs) connected to a stick of fast memory (HBM)." Components inside an H100/B200 SM:

- **CUDA Cores**: SIMT vector ALUs, 32 fp32 per subpartition. Do ReLUs, pointwise ops, reductions. Analogous to TPU VPU lanes, but each thread has its own instruction pointer (warp divergence is real).
- **Tensor Core (TC)**: dedicated matmul unit. On H100, ~1024 bf16 FLOPs/cycle/TC (≈ 8×8×8 matmul). The TC carries the vast majority of FLOPs (990 bf16 TFLOPs/s on H100 vs. 66 TFLOPs/s from CUDA cores).
- **Warp Scheduler**: dispatches up to 64 resident warps per SM, hides memory latency by switching between them.

Each SM has 4 identical subpartitions, each with its own TC + Warp Scheduler + 16k 32-bit registers. The B200 introduces **TMEM** (256kB/SM) because the TC has grown so large its accumulators no longer fit in SMEM.

### GPU ↔ TPU Cheat Sheet

| GPU                          | TPU              | What it is                          |
| :--------------------------- | :--------------- | :---------------------------------- |
| Streaming Multiprocessor (SM)| Tensor Core      | Core "cell" containing other units  |
| Warp Scheduler               | VPU              | SIMD vector arithmetic unit         |
| CUDA Core                    | VPU ALU          | SIMD ALU                            |
| SMEM (L1)                    | VMEM             | Fast on-chip cache                  |
| Tensor Core                  | MXU              | Matmul unit                         |
| HBM                          | HBM              | Main high-BW memory                 |

The instructive comparison is the **count**: an H100 has 132 SMs × 4 subpartitions = 528 independent SIMD units (~16k ALUs total). A TPU v5p has 2 Tensor Cores × 4 VPU slots = 8. GPUs are far more **modular**; TPUs are far more **monolithic**. Modularity buys flexibility ("just launch dozens of kernels") at the cost of harder-to-reach peak performance — the L2 cache is shared, memory coalescing is fragile, and the compiler controls less.

### Memory Hierarchy

| GPU   | SMs | SMEM/SM | L2     | HBM    | HBM BW   | bf16 TFLOPs/s |
| :---- | :-: | :-----: | :----: | :----: | :------: | :-----------: |
| V100  | 80  | 96 kB   | 6 MB   | 32 GB  | 0.9 TB/s | —             |
| A100  | 108 | 192 kB  | 40 MB  | 80 GB  | 2.0 TB/s | 312           |
| H100  | 132 | 256 kB  | 50 MB  | 80 GB  | 3.4 TB/s | 990           |
| H200  | 132 | 256 kB  | 50 MB  | 141 GB | 4.8 TB/s | 990           |
| B200  | 148 | 256 kB  | 126 MB | 192 GB | 8.0 TB/s | 2,250         |

TPU VMEM is roughly 2× larger and ~7× more bandwidth than GPU L2. That's why TPUs are often easier to keep close to roofline for inference — weights and activations can be parked in VMEM where GPU code has to fight the L2 cache.

---

## 2. Networking: Node → Leaf → Spine

A DGX H100 SuperPod has three levels:

| Level     | # GPUs | Degree | Switch BW (full-duplex)   | Collective BW         |
| :-------- | :----: | :----: | :-----------------------: | :-------------------: |
| Node      | 8      | 8      | 6.4 TB/s (NVLink/NVSwitch)| **450 GB/s** per GPU  |
| Leaf (SU) | 256    | 32     | 25.6 TB/s                 | **400 GB/s** per node |
| Spine     | 1024   | 4      | 51.2 TB/s                 | 400 GB/s per node     |

Within a node, GPUs talk over NVLink with full all-to-all connectivity through NVSwitch. Beyond a node, it's a fat-tree InfiniBand network (8×400 Gbps IB links per node = 400 GB/s node egress).

Important caveat: NVIDIA *advertises* 450 GB/s NVLink, but in practice, AllReduce throughput tops out around **370 GB/s** even on multi-GB messages — and on more typical sizes (e.g. a LLaMA-3 70B MLP shard of ~58 MB), only ~150 GB/s. **Adjust your rooflines by ~20%.**

### SHARP (In-Network Reductions)

Hopper-era NVIDIA switches support SHARP — switches themselves perform reductions and multicast results, which *theoretically* halves AllReduce cost. In practice the speedup is **~30%**, not 75%, so it just compensates for the bandwidth gap rather than meaningfully scaling things further.

---

## 3. Collectives: The Formulas Worth Memorizing

For an array of $B$ bytes, with $N$ GPUs per node, $W$ = per-GPU egress bandwidth:

- **Intra-node AllGather / ReduceScatter:** $T \approx \frac{B(N-1)}{NW} \approx \frac{B}{W}$
- **Intra-node AllToAll:** $T \approx \frac{B}{NW}$ (2× faster than on TPUs at the node level)
- **Cross-node AllGather / RS:** $T \approx \frac{B}{W_\text{node egress}}$ — driven by node egress, not GPU egress
- **Cross-node AllToAll:** $T \approx \frac{B}{M \cdot W_\text{node egress}}$ where $M = N/8$ is the number of nodes — drops from $B/(8 \cdot 450\text{e9})$ within a node to $B/(2 \cdot 400\text{e9})$ across just 2 nodes — **>4× degradation**.
- **AllReduce:** 2× the AG/RS cost (unless SHARP).

A useful sub-rule: when an array is sharded along an inner axis $Y$, the outer reduction's cost drops by roughly the number of *nodes* spanned by $Y$ — which is why DeepSeek-V3's 2-way DP across nodes lets it dodge the leaf-level bottleneck.

---

## 4. LLM Rooflines on GPUs

When does each parallelism strategy stop being compute-bound? Let $C$ = peak FLOPs/s, $W$ = collective bandwidth, $\alpha = C / W$.

For an H100, intra-node $\alpha = 990\text{e}12 / 450\text{e}9 \approx 2200$. Beyond a node, $\alpha = 990\text{e}12 / 400\text{e}9 \approx 2475$.

### Data Parallelism / FSDP

Compute-bound requires per-GPU **token** batch size:

$$\frac{B}{X} > \frac{C}{W_\text{collective}} \approx 2500 \text{ tokens / GPU}$$

For comparison, TPUs need ~850. This is why LLaMA-3 (16k H100s) needs a ~16M-token global batch, and DeepSeek V3 (2048 H800s with only 300 GB/s NVLink) needs ~3300 tokens/GPU, i.e. ~6.7M total (they used 4M).

**MoE penalty:** the bound inflates by $E/k$. For a model with $E=128$ experts and $k=4$ active, the per-GPU batch jumps to ~80k tokens — borderline absurd. This is the structural reason MoE training relies so heavily on pipeline + expert parallelism instead of pure FSDP.

### Tensor Parallelism

Compute-bound when $Y < F \cdot W / C$. For LLaMA-3 ($F = 28{,}672$), this allows ~11-way TP intra-node — rounded down to **8-way (a single NVLink domain)**. Spanning 2 nodes barely buys you anything (~16-way max). Beyond that you're comms-bound.

### Expert Parallelism

For an MoE with $F < 8 C/W_\text{node}$: limited to 1–2 nodes of EP (DeepSeek-V3 territory, with small $F$). For $F > 8 C/W_\text{node}$: full multi-node EP up to $E$ nodes is feasible.

### Pipeline Parallelism

Comms cost is **basically free** — the per-layer cost scales as $\frac{2BD}{W \cdot N_\text{layers}}$. So why isn't everyone PP-maxxed?

1. **Code complexity** (zero-bubble schedules don't fit GSPMD well).
2. **PP fights ZeRO-3.** Each microbatch can't amortize a full weight AllGather.
3. **Bubbles + step imbalance** still leave waste even with careful scheduling.

But PP buys a hidden win: each PP stage spans more nodes, which scales up the *DP* AllReduce bandwidth. That's why LLaMA-3 uses 16-way PP — it cuts the FSDP critical batch size by 16×.

---

## 5. Two Worked Examples

**LLaMA-3 (16k H100, 16M tokens):**
- 8-way TP within a node
- 16-way PP
- 128-way ZeRO-1 DP

Dense model, small per-GPU batch (~1k tokens). The 16-way PP is what makes it work — without it, the FSDP roofline would demand a much larger global batch.

**DeepSeek V3 (2048 H800, 62.9M tokens):**
- 64-way EP across 8 nodes
- 16-way PP
- 2-way ZeRO-1 DP

Sparse MoE ($k=8$, $E=256$). With 1024-way model parallelism (EP × PP), the residual DP AllReduce happens at the *spine* level over only 2 nodes — which gets the (N-1)/N = 2× bandwidth bonus.

---

## 6. GB200 NVL72: What Changes

Blackwell with NVLink 5 doubles intra-node bandwidth (450 → 900 GB/s) but on a B200 SuperPod the **node egress stays at 400 GB/s** while FLOPs/chip ~2.25×. Net effect: **cross-node rooflines get harder** — the DP critical batch size grows from 2475 to ~5625 tokens/GPU.

GB200 NVL72 fixes this with a 72-GPU NVLink domain and **3.6 TB/s node egress** (4× H100). That widens the cross-node compute-bound region by ~4× and makes EP across nodes substantially cheaper. Within a node, however, doubled FLOPs ≈ doubled BW, so rooflines look largely unchanged.

A bonus from Grace Hopper / Grace Blackwell: the CPU↔GPU NVLink-C2C is at full GPU-to-GPU bandwidth, so host memory becomes a viable offload target without a bandwidth cliff.

---

## TL;DR

- **GPU rooflines are stricter than TPU rooflines** for cross-chip work. Per-GPU FSDP batch ≈ 2500 tokens (vs. ~850 on TPU); only ~8-way TP fits in an NVLink domain.
- **Empirical NVLink BW is ~80% of spec** — re-derive your batch sizes accordingly.
- **SHARP doesn't save you.** ~30% in practice, not the 2× theoretical.
- **MoE forces you off pure DP.** $E/k$ penalty makes pipeline + expert parallelism mandatory.
- **PP is ~free in comms** but expensive in code. The real reason to use it: it shrinks the DP critical batch size by $N_\text{stages}$.
- **GB200 NVL72** is the structural fix for cross-node bottlenecks; B200 SuperPod alone makes them slightly worse.

The chapter does for GPUs what the rest of the book did for TPUs: gives you a small set of inequalities you can plug into during model design, before any code is written. Worth a careful read alongside [Section 5 (Training)](https://jax-ml.github.io/scaling-book/training) and [Section 7 (Inference)](https://jax-ml.github.io/scaling-book/inference) for the TPU side of the same picture.
