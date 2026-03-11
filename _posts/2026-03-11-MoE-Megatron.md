---
layout: post
title: "Scalable MoE Training with Megatron Core"
date: 2026-03-11
categories: [MoE]
tags: [MoE]
---

Reading the following paper:
- [Scalable Training of Mixture-of-Experts Models with Megatron Core](https://arxiv.org/pdf/2603.07685)


### 1. Introduction & The Core Asymmetry of MoE
MoE models like DeepSeek-V3 and Qwen3 have redefined the compute-optimal frontier, achieving equivalent or better quality than dense models while dramatically reducing training FLOPs. However, as NVIDIA's new report on Megatron-Core MoE highlights, this algorithmic efficiency introduces brutal systems challenges rooted in a single, fundamental asymmetry: **Sparsity**.

In a dense transformer, every parameter activates for every token (total parameters scale proportionally with per-token compute). MoE breaks this lockstep. DeepSeek-V3, for instance, has 685B total parameters but activates only 37B per token—an 18x gap. This **Parameter-Compute Mismatch** creates massive memory pressure to store all $E$ experts, high communication volume to route tokens, but leaves insufficient local compute to hide these latencies.

Megatron-Core addresses this by systematically breaking down **Three Walls of MoE**: Memory, Communication, and Compute Efficiency. 

### 2. The Dense-Sparse Mismatch & MoE Parallel Folding
Traditional parallelism (3D/4D parallelism) assumes uniform computation across layers. Attention layers rely heavily on Tensor Parallelism (TP) to shard large QKV matrices and Context Parallelism (CP) for long sequences. However, MoE layers require Expert Parallelism (EP) to distribute the massive pool of experts across GPUs. 

Historically, frameworks forced MoE and Attention to share the same parallelism configuration, carving EP out of the Data Parallelism (DP) group ($EP \le DP$). This "Dense-Sparse Mismatch" resulted in either artificially low TP (bottlenecking attention) or artificially high TP (fragmenting small expert MLPs into inefficient slivers).

**Insight & Solution:** Megatron-Core introduces **MoE Parallel Folding**, completely decoupling the parallelism mappings. 
*   Attention groups: `TP × CP × DP × PP`
*   MoE groups: `ETP × EP × EDP × PP`

The only constraint is that Pipeline Parallelism (PP) must remain consistent. By folding EP across the `TP × CP` domains, a 256-GPU cluster can run Attention at `TP=4, CP=2` while MoE runs independently at `ETP=1, EP=64`. This keeps both layers in their optimal utilization regimes and traps high-bandwidth All-to-All communication inside the intra-node NVLink domain.

### 3. Breaking the Memory Wall: The "Zero-Overhead" Trick
Activations dominate the memory footprint in large-scale MoE training. For DeepSeek-V3, activations account for ~131 GB out of a 199.5 GB per-GPU requirement. While Megatron-Core uses fine-grained recomputation and stream-overlapped CPU offloading, the most elegant solution is **Memory-Efficient Permutation**, which provides massive memory savings with *zero computational overhead*.

**The Math:**
Given an input token representation $x$, the router selects a set of experts $\mathcal{T}(x) \subset \{1, \dots, E\}$ with probabilities $p_i$. A standard expert is a 2-layer MLP without bias: $E_i(x) = W^{(i)}_2 \phi(W^{(i)}_1 x)$.

Normally, the weighted combination occurs *after* the expert outputs are computed:
$$y = \sum_{i \in \mathcal{T}(x)} p_i \cdot W^{(i)}_2 \phi(W^{(i)}_1 x)$$
In this formulation, computing the backward pass for the router weights ($\partial \mathcal{L} / \partial p_i$) requires the system to save the massive post-expert tensor $E_i(x)$. 

Because scalar multiplication commutes with linear mappings, Megatron-Core absorbs $p_i$ *into* the activation function before the second linear layer:
$$y = \sum_{i \in \mathcal{T}(x)} W^{(i)}_2 \left( p_i \cdot \phi(W^{(i)}_1 x) \right)$$
**Insight:** By applying $p_i$ earlier, the gradient $\partial \mathcal{L} / \partial p_i$ now only depends on $\phi(W^{(i)}_1 x)$. Because $W^{(i)}_1 x$ (denoted as $z_i$) is already stored for the SwiGLU backward pass, the framework doesn't need to save any additional tensors for the router's backward pass. For DeepSeek-V3, this simple algebraic manipulation saves ~26.3 GB of activation memory per GPU for free.

### 4. Breaking the Communication Wall: 1F1B & The W/D Split
With high EP, token dispatch and combine (All-to-All) can consume up to 60% of step time. Optimized custom dispatchers like HybridEP (for NVLink) and DeepEP (for InfiniBand) fuse permutations and improve pure bandwidth utilization. However, true scalability requires hiding communication latency behind computation.

Megatron-Core utilizes a 1F1B (One Forward, One Backward) overlapping schedule, interleaving computation streams with communication streams. But here's the catch: MoE expert MLPs are so small that their forward compute isn't long enough to hide the backward All-to-All dispatch. 

**Insight - The W/D Split:** To maximize overlap, Megatron-Core breaks the backward MLP dependency. Calculating the Weight gradient (`W/mlp`) doesn't depend on the backward token dispatch—only the Data gradient (`D/mlp`) does. By splitting these operations, the system can execute `W/mlp` concurrently with the forward MLP (`F/mlp`) of an adjacent microbatch, artificially extending the compute window and successfully hiding the communication latency. 

### 5. Breaking the Compute Efficiency Wall: Dropless MoE & CUDA Graphs
In fine-grained MoE (e.g., 256 small experts), GPU kernels shrink. The GPU executes operations so fast that the CPU cannot launch the next kernel quickly enough, leaving the GPU idle (Host-Boundedness). 

CUDA Graphs typically solve this by capturing a sequence of kernel launches into a single graph. However, modern architectures (like DeepSeek-V3) use **dropless routing**—meaning expert token allocations are strictly dynamic based on runtime probability distributions. CUDA Graphs require *static* tensor shapes and crash upon dynamic shapes. 

**The Trio Solution for Full Graph Coverage:**
To capture the entire layer, Megatron-Core engineers developed three brilliant mitigations:
1.  **Device-Initiated Kernels:** Grouped GEMMs and HybridEP now read their dynamic shape configurations directly from device memory, bypassing CPU synchronization completely.
2.  **ECHO (Elastic Cloning for Hot Experts):** A massive load imbalance means worst-case buffer allocations are huge. ECHO uses a dynamic bin-packing planner to clone "hot" expert weights to under-utilized GPUs (spare slots) during the forward pass, and reduces gradients back to the "home" expert in the backward pass. This flattens the variance in token distribution.
3.  **Paged Stashing:** Even with ECHO, pre-allocating memory for the *worst-case* token limit causes severe $O(\text{layers} \times \text{worst\_case})$ memory fragmentation. Megatron-Core adapts OS-level paging: a single worst-case `tmp` buffer is shared across all layers for computation. After a layer computes, its actual tokens are "stashed" into a tightly packed, paged memory buffer, dropping memory requirements to $O(\text{worst\_case} + \text{actual\_total})$.

### 6. Reduced Precision (FP8/FP4) & Long Context Dynamics
The framework integrates deep mixed-precision support down to **NVFP4 (4-bit)**. 
**Insight:** MoE inherently amplifies the risks of reduced-precision. If the router uses low-precision, expert collapse can occur. The core philosophy is *Selective Precision*: the router, embeddings, and main gradients stay in FP32/BF16, while the bulk $O(N)$ GEMMs execute in E4M3 FP8 or E2M1 FP4. To enable Graph compatibility, padding is fused into token permutation, enforcing 128-token multiple alignment for Grouped Quantization.

**The Long Context Shift:**
When sequence length ($s$) crosses ~64K, a phase shift occurs: Scaled Dot-Product Attention (SDPA), which is $O(s^2)$, overtakes the MoE MLP ($O(s)$) as the dominant compute bottleneck (jumping to ~70% of total FLOPs). 
To combat this, Megatron-Core uses **Dynamic Context Parallelism (Dynamic-CP)** over packed sequences (THD format). Rather than forcing a statically high CP degree that wastes cross-device bandwidth for shorter sequences within a batch, Dynamic-CP resizes the CP degree on a *per-microbatch* basis dynamically at runtime.

### Final Thoughts
Training 685B+ parameter MoE models is no longer just an algorithmic challenge; it is fundamentally a distributed systems orchestration problem. The optimization pipeline operates recursively: reducing precision (FP8) frees up memory, which is then re-invested into extra buffers to overlap EP communication, which subsequently exposes CPU latency, triggering the need for CUDA Graphs, which requires fixing dynamic memory allocation. Megatron-Core's holistic, cross-stack design highlights exactly what it takes to scale out the next generation of LLMs.
