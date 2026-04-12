---
layout: post
title: "Scalable Training of Mixture-of-Experts Models with Megatron Core"
date: 2026-03-11
categories: [MoE]
tags: [MoE, Megatron, NVIDIA, distributed-training, FP8]
---

Reading the following paper:
- [Scalable Training of Mixture-of-Experts Models with Megatron Core](https://arxiv.org/pdf/2603.07685)

This paper presents **Megatron-Core MoE**, the MoE training stack within NVIDIA's Megatron-Core framework. It addresses the fundamental systems challenges of training trillion-parameter-class MoE models at high throughput on NVIDIA GPU clusters. The headline results: **1,233 TFLOPS/GPU** on GB300 and **1,048 TFLOPS/GPU** on GB200 for DeepSeek-V3-685B.

---

## 1. MoE Fundamentals

### Architecture

Given an input token representation **x**, the router computes:

$$\mathbf{p}(\mathbf{x}) = \text{Softmax}(\mathbf{W}_r \mathbf{x})$$

The MoE layer output is:

$$\text{MoE}(\mathbf{x}) = \sum_{i \in \text{TopK}(\mathbf{p}(\mathbf{x}))} p_i(\mathbf{x}) \cdot E_i(\mathbf{x})$$

where $E_i$ is the $i$-th expert network. Three key advantages:
- **Scalable capacity**: model size grows independently of per-token compute
- **Computational efficiency**: only $K$ of $E$ experts activate per token
- **Specialization**: different experts learn different input patterns

### The Parameter-Compute Mismatch

This is the central insight of the paper. In a dense transformer with $N_\text{total}$ parameters, FLOPs per token is approximately $6N_\text{total}$, so parameters and computation scale in lockstep.

For MoE, per-token computation is approximately $6N_\text{active}$ where $N_\text{active} \propto K$ while $N_\text{total} \propto E$, and $K \ll E$.

**Concrete example**: DeepSeek-V3 has 685B total parameters but only 37B active per token, an **18x gap**.

This creates the **Three Walls**:

| Wall | Root Cause | Manifestation |
|------|-----------|---------------|
| **Memory** | All $E$ experts' params/grads/optimizer states in memory, but only $K$ activate | 199.5 GB per GPU for DeepSeek-V3 |
| **Communication** | EP requires all-to-all collectives to route tokens across GPUs | 20-60% of training time |
| **Compute** | Small per-expert GEMMs underutilize Tensor Cores; many kernel launches | Host-boundedness |

---

## 2. MoE Layer Architecture: Four-Stage Forward Pass

**Stage 1: Route.** A learned linear projection maps each token's hidden state to $E$ logits: $\mathbf{l} = \mathbf{W}_r^\top \mathbf{x} \in \mathbb{R}^E$. A score function, either softmax or sigmoid as used in DeepSeek-V3, converts these into probabilities. Top-$k$ selection chooses the active experts.

**Stage 2: Dispatch.** Tokens are permuted so those destined for the same expert are contiguous. Three backends are described: AllGather, All-to-All, and Flex via DeepEP/HybridEP.

**Stage 3: Expert Computation.** All local experts run in a single Grouped GEMM call. Each expert is a two-layer MLP with optional gating:
$$E_i(\mathbf{x}) = \mathbf{W}_2^{(i)} \phi(\mathbf{W}_1^{(i)} \mathbf{x})$$

**Stage 4: Combine.** Inverse communication returns tokens, unpermutation restores order, and shared expert output is added.

---

## 3. Parallel Folding and Multi-Dimensional Parallelism

### The Dense-Sparse Mismatch

A single Transformer block contains two fundamentally different computation patterns:

| Aspect | Attention (Dense) | MoE (Sparse) |
|--------|-------------------|--------------|
| **TP** | Large QKV matrices benefit from high TP | Small per-expert dims make high TP counterproductive |
| **CP** | Long sequences benefit from high CP | No sequence dependency; CP is irrelevant |
| **EP** | Not applicable | Essential for distributing experts |

Prior frameworks forced `World Size = TP x CP x PP x DP`, where `EP <= DP`. This creates three problems:
1. **Multiplicative GPU requirements**: EP=8 forces DP>=8, and with CP=8 the minimum becomes 64 GPUs.
2. **Forced suboptimal parallelism**: high TP fragments small experts, while low TP underparallelizes attention.
3. **Cross-node communication**: EP constrained within DP forces all-to-all across slow interconnects.

### Parallel Folding Solution

**Core idea**: decouple attention and MoE parallelism mappings.

- **Attention layers** form groups over `TP x CP x DP x PP`
- **MoE layers** form groups over `ETP x EP x EDP x PP`
- **Only constraint**: PP must remain consistent

Key benefits:
1. **Breaks EP <= DP**: EP can fold across TP x CP groups. Example: attention TP=4, CP=2, DP=8, PP=4 on 256 GPUs. Traditionally EP<=8; with folding EP=64 becomes possible.
2. **Reduces minimum GPUs**: CP=8 and EP=8 traditionally requires 64 GPUs; with folding, only 8.
3. **Independent optimization**: attention uses high TP, while MoE uses ETP=1 for full expert width.
4. **NVLink locality**: both CP and EP all-to-all stay within the NVLink domain.

### Gradient Handling

Expert gradients are scaled by `edp_size / dp_size` to account for the different effective batch sizes seen by experts versus dense layers.

---

## 4. Breaking the Memory Wall

### Memory Anatomy (DeepSeek-V3, PP4 x VPP4 x EP64, 256 GPUs)

| Component | Memory/GPU | Optimization |
|-----------|-----------|--------------|
| Weights & Gradients | 36.4 GB | PP, EP, TP sharding |
| Main Weights & Optimizer States | 32.1 GB | Distributed optimizer, BF16 moments |
| Activations | **131.0 GB** | Low precision, recomputation, offloading |
| **Total** | **199.5 GB** | |

**Key insight**: activations dominate, exceeding weights and optimizer states combined.

### Memory-Efficient Permutation (Zero Overhead)

Standard formulation applies routing weights **after** expert computation:
$$y = \sum_{i \in \mathcal{T}(\mathbf{x})} p_i \cdot \mathbf{W}_2^{(i)} \phi(\mathbf{W}_1^{(i)} \mathbf{x})$$

Memory-efficient version absorbs $p_i$ **before** the second linear layer:
$$y = \sum_{i \in \mathcal{T}(\mathbf{x})} \mathbf{W}_2^{(i)} \left(p_i \cdot \phi(\mathbf{W}_1^{(i)} \mathbf{x})\right)$$

Since $\mathbf{W}_2^{(i)}$ is a pure linear map with no bias, scalar multiplication commutes:
$$p_i \cdot \mathbf{W}_2^{(i)} \mathbf{h} = \mathbf{W}_2^{(i)} (p_i \cdot \mathbf{h})$$

**Why this saves memory**: in the standard version, computing $\partial\mathcal{L}/\partial p_i$ requires retaining each expert output $E_i(\mathbf{x})$. In the efficient version, $p_i$ multiplies $\phi(\mathbf{z}_i)$ directly, so $\partial\mathcal{L}/\partial p_i$ only depends on $\phi(\mathbf{z}_i)$, which can be recomputed from $\mathbf{z}_i = \mathbf{W}_1^{(i)} \mathbf{x}$ already saved for SwiGLU backward. This saves roughly **26.3 GB per GPU** for DeepSeek-V3 with essentially zero extra compute.

### FP8/FP4 Activations

Linear layer inputs stored in FP8 instead of BF16 reduce memory by 50% per tensor. For DeepSeek-V3, that is roughly **16 GB saved**. FP4 pushes this further to a 75% reduction.

### Fine-Grained Recomputation

Two composable techniques:
1. **Granular recomputation**: selectively recompute only specific operations such as activation functions, LayerNorm, and MLA up-projection, typically with under 5% compute overhead.
2. **Output-discarding recomputation**: release checkpointed module outputs immediately after downstream consumption and restore them via recomputation during backward.

| Recomputation Target | Memory Saved/GPU |
|---------------------|-----------------|
| MLA Up-Projection | 30.4 GB |
| SwiGLU Activation | 3.8 GB |
| LayerNorm | 8.2 GB |
| **Total** | **42.4 GB** |

**Critical insight**: full-layer recomputation of MoE is especially expensive because it re-triggers EP all-to-all communication. Fine-grained recomputation avoids that penalty.

### Fine-Grained Activation Offloading

**Forward**: input activations are offloaded to CPU through a dedicated D2H stream, overlapping with the next module's computation.

**Backward**: Layer-Staggered Reload reloads the same module type from the **next** layer while gradients are computed for the current layer. Only one activation per module type resides on GPU at a time.

**Peak memory advantage over full recomputation**:
- Full recomputation: $L \times \text{layer\_input} + 1 \times \text{layer\_intermediate}$
- Offloading: $1 \times \text{layer\_input} + 1 \times \text{layer\_intermediate}$

Results: **10-18% memory reduction** with only **1.6-2% throughput overhead**. For Qwen3-235B, offloading enabled a lower TP degree and about **15% throughput improvement**.

### Precision-Aware Optimizer

Adam stores first and second moments. The optimization here is to store moments in BF16 or FP8, then cast to FP32 inside TransformerEngine's FusedAdam kernel for the actual update.

Memory per parameter per DP rank decreases from $6 + 12/d$ bytes to $6 + 8/d$ bytes, saving on the order of **10-12 GB** from the 32.1 GB optimizer-state budget.

### FSDP for MoE

**Dual DeviceMesh design**: dense layers shard across the full DP group, while expert layers shard across the EDP group.

Two key optimizations:
1. **Non-uniform sharding**: flatten and concatenate module parameters, then shard non-uniformly so shard boundaries align with communication buffers for zero-copy collectives.
2. **Persistent double buffers with NCCL User Buffer Registration**: pre-allocate two persistent buffers and cycle between them. This reduces SM footprint from 8-32 SMs to 1-4 SMs.

---

## 5. Breaking the Communication Wall

### Communication Anatomy

For DeepSeek-V3: 58 MoE layers times 2 operations per layer gives **116 dispatch/combine operations per forward pass**. Backward doubles this. At 50 GB/s inter-node bandwidth, a single 200 MB dispatch already costs milliseconds, which compounds rapidly over an iteration.

### HybridEP

Developed by NVIDIA for NVLink-rich topologies such as NVL72.

**Dispatch**: reads data from global memory into shared memory based on routing info, then writes to destinations via FIFO queues. For inter-node traffic, GPUs with the same local index across nodes exchange first, then forward within the node.

**Combine**: fuses reduction into the communication kernel itself. Cross-node data is reduced first, then the remaining intra-node reduction completes locally.

Performance on GB200 with EP64:
- HybridEP dispatch: **675 us** vs all-to-all **930 us**
- HybridEP combine: **744 us** vs all-to-all **827 us**

### EP Communication Overlapping

**1F1B forward-backward overlap** merges the forward pass of one microbatch with the backward pass of another.

Two key optimizations:
1. **Stream Separation**: compute stream and communication stream run in parallel.
2. **W/D Split**: backward MLP is split into weight gradient (`W/mlp`) and data gradient (`D/mlp`). Only `D/mlp` depends on backward dispatch, which opens more room to hide communication.

Result: EP communication overhead drops from **30-40% to under 5%** of iteration time for DeepSeek-V3 on H100.

---

## 6. Breaking the Compute Efficiency Wall

### Grouped GEMM

DeepSeek-V3's 256 small experts produce GEMMs with M dimensions around 128 tokens per expert, far below the regime needed for peak Tensor Core efficiency.

Four implementations are discussed:
1. **Multi-stream cuBLASLt GEMMs**: individual GEMMs launched into multiple CUDA streams
2. **CUTLASS Grouped GEMM**: a fused single-kernel path
3. **cuBLASLt Grouped GEMM** (device-initiated): reads shapes from device memory, making it CUDA-Graph-compatible
4. **cuteDSL Grouped GEMM**: fuses SwiGLU activation and FP8 quantization into the GEMM epilogue

### Permutation Fusion

Three-stage pipeline:
- Preprocessing: generate the Row ID map once
- Permute: move tokens according to offset maps
- Unpermute: inverse permutation plus FP32 accumulation

### Router and Aux-Loss Fusion

Three fused kernels are described:
- score computation with top-$k$ and softmax/sigmoid
- score computation for auxiliary loss
- auxiliary loss computation

### CUDA Graphs

**Full vs Partial**:
- **Full CUDA Graphs** capture the entire forward-backward pass, but only for drop-and-pad MoE
- **Partial CUDA Graphs** capture only static components, while dynamic token dispatch/expert GEMM/combine remain outside

Memory optimizations include graph count reduction, pool sharing, and static buffer reuse across pipeline stages.

Result: about **10% end-to-end speedup** with roughly **7 GB** extra memory on DeepSeek-V3 GB200.

### Full CUDA Graphs for Dropless MoE: Three Complementary Techniques

**Challenge 1**: kernel launch without knowing problem size.
Solution: **device-initiated Grouped GEMM** so cuBLASLt reads shapes from device memory.

**Challenge 2**: memory allocation without knowing actual size, which otherwise forces worst-case buffers.

#### ECHO (Elastic Cloning for Hot Experts)

Popular experts can receive far more tokens than others. ECHO dynamically clones hot experts onto spare slots on underutilized ranks.

**Forward**: a planner identifies hot experts, builds the hot-expert map and updated routing map, copies weights to spare slots, and routes tokens to both home and cloned experts.

**Backward**: gradients from cloned experts are reduced back to the home expert.

#### Paged Stashing

Decouples worst-case computation buffers from activation storage:
- **Single tmp buffer** sized for worst case and shared across all layers
- **Paged stashing buffer** stores only the actual tokens used per layer

This reduces memory from $O(\text{layers} \times \text{worst\_case})$ to $O(\text{worst\_case} + \text{actual\_total})$.

---

## 7. Reduced-Precision Training (FP8/FP4)

### Strategy: Selective Precision

Three principles:
1. **Protect routing**: keep the router in FP32
2. **Preserve key components**: embeddings, output layers, main gradients, master weights, and optimizer states stay high precision
3. **Quantize bulk computation**: expert GEMMs and activations go low precision

### FP8 Recipes

**Per-Tensor FP8**: one scale per tensor. Uses E4M3 inputs and weights with E5M2 gradients.

**Blockwise FP8**: uses E4M3 for all tensors, with activations and gradients quantized in 1 x 128 tiles and weights in 128 x 128 blocks.

**MXFP8**: uses 1 x 32 element granularity and native Blackwell hardware support. A key caveat is that parameter AllGather still communicates in BF16 because forward and backward require different quantization directions.

### NVFP4

Uses E2M1 with **two-level microscaling**:
- per-tensor FP32 scale
- per-block E4M3 scale with blocks of 16 elements

Three algorithmic additions are highlighted for stable training:
1. **Random Hadamard Transforms**
2. **2D scaling**
3. **Stochastic rounding**

### FP8/FP4 Primary Weights

The framework eliminates redundant BF16 copies by casting directly from FP32 to FP8/FP4. Distributed optimizer quantization proceeds by taking local abs-max, then global abs-max through AllReduce, then quantizing with that shared scale.

### MoE-Specific FP8/FP4 Challenges

**Padding alignment**: FP8 GEMMs require token-dimension alignment to 16 or 32 depending on the recipe.

**Grouped quantization**: multiple expert input tensors are quantized in a single fused kernel.

**NVFP4 quantization fusion**: fuses Hadamard transforms with quantization to avoid extra BF16 traffic.

---

## 8. Long-Context MoE Training

### The Computational Shift

At 64K tokens, **SDPA consumes 69% of FLOPs** compared with roughly 10-15% at short context. Since SDPA scales as $O(s^2)$ while MoE scales as $O(s)$, attention becomes the dominant cost.

**Key recommendation**: do not recompute core attention at long sequence lengths. At 64K, SDPA recomputation adds about **18% compute overhead** but saves only **9 GB** memory, while recomputing non-SDPA components saves much more memory with lower performance impact.

### CP vs TP Trade-offs

- **P2P CP**: preferred across nodes because ring-style KV exchange overlaps naturally with SDPA
- **All-to-all CP**: converts sequence-sharded layouts to head-sharded layouts before SDPA
- **TP**: preferred within nodes for sharding linear weights

Practical guideline: **all-to-all CP + TP inside nodes; P2P CP across nodes**.

### Dynamic Context Parallelism

For variable-length sequences, the system selects CP degree per microbatch based on actual sequence lengths. The per-token loss is:

$$\mathcal{L} = \frac{\sum_{t \in \mathcal{V}} \ell_t}{|\mathcal{V}|}$$

where $\mathcal{V}$ is the set of valid non-padding tokens.

---

## 9. Production Features

### Load Balancing Strategies

Three approaches are discussed:
- **Auxiliary loss**: gradient-based, differentiable, soft balance
- **Sinkhorn**: assignment-based, non-differentiable, hard balance
- **Aux-loss-free / Expert Bias**: feedback-based, non-differentiable, adaptive

### Latent MoE

Latent MoE inserts a shared down-projection before expert dispatch and an up-projection after combine, reducing both all-to-all volume and per-expert weight size by the compression ratio $\alpha = d / \ell$.

### Flexible Asymmetric VPP

Allows different numbers and types of layers per virtual pipeline stage, which is important for balancing models like DeepSeek-V3 that mix dense, MoE, and MTP layers.

### Upcycling

Converts dense checkpoints into MoE via virtual-group initialization by sharding dense MLP weights, duplicating shards, and initializing router weights so the initial MoE exactly matches the dense model's output.

### Multi-Token Prediction (MTP)

MTP supervises multiple future tokens at each position while preserving causal dependencies through hidden-state transitions. During inference, the model falls back to ordinary single-token prediction.

### Muon Optimizer

Muon applies matrix-aware orthogonalized updates rather than element-wise AdamW-style updates. Production support includes split-QKV handling, distributed optimizer integration, CPU offloading, and **MuonClip** to prevent attention explosions in large-scale training.

---

## 10. Performance Results

| Model | System | #GPUs | Dtype | TFLOPS/GPU | Tokens/s/GPU |
|-------|--------|-------|-------|-----------|-------------|
| DeepSeek-V3 | GB300 | 256 | MXFP8 | **1,233** | 4,730 |
| DeepSeek-V3 | GB200 | 256 | MXFP8 | **1,048** | 4,020 |
| DeepSeek-V3 | GB200 | 256 | BF16 | 857 | 3,298 |
| DeepSeek-V3 | H100 | 1,024 | FP8-BLK | 368 | 1,412 |
| Qwen3-235B | GB300 | 256 | MXFP8 | **974** | 6,583 |
| Qwen3-235B | GB200 | 256 | MXFP8 | 919 | 6,212 |
| Qwen3-235B | GB300 | 128 | MXFP8 (131K seq) | 1,150 | 1,556 |

GB200 and GB300 deliver roughly **3x higher token throughput** than H100. Long-context DeepSeek-V3 at 256K tokens still reaches **88% of short-context MFU**.

---

## 11. Case Study: DeepSeek-V3 GB200 vs H100

| Config | GB200 (256 GPUs) | H100 (1,024 GPUs) |
|--------|------------------|-------------------|
| **TP/PP/EP** | 1/4/64 | 2/8/64 |
| **Precision** | MXFP8 | FP8-Blockwise |
| **Dispatcher** | HybridEP | DeepEP |
| **Recompute** | mlp only | mlp, mla_up_proj, moe_act, layernorm |
| **CUDA Graphs** | Enabled | - |
| **EP Overlap** | - | Enabled |
| **Performance** | 1,048 TFLOPS/GPU | 368 TFLOPS/GPU |

**Key insight**: the same model requires different optimization strategies on different hardware. GB200's larger memory and NVL72 topology shift the bottleneck toward CPU overhead, making CUDA Graphs critical. H100's tighter memory and NVL8 topology shift the bottleneck toward communication, making EP overlap critical.

---

## 12. Systematic Optimization Workflow

A practical three-phase workflow emerges from the case studies on Mixtral, DeepSeek-V3, and Qwen3.

### Phase 1: Establish Memory-Feasible Parallelism

Memory feasibility is the first hard constraint.

| Strategy | Peak Activation | Weight Memory | Optimizer States | Comm (Per-Layer) |
|----------|----------------|---------------|-----------------|-----------------|
| TP | $1/d$ (with SP) | $1/d$ | $1/d$ | High |
| EP | ~1 (load-dependent) | $1/d$ (MoE only) | $1/d$ | Medium |
| PP | 1 (>1 with VPP) | $1/d$ | $1/d$ | Medium |
| CP | $1/d$ | 1 | $1/d$* | Medium |
| DP | 1 | 1 | $1/d$* | Low |

`*` Requires distributed optimizer.

Practical tip: `--fake-init-process-group` lets you emulate distributed training on a single GPU for parallelism experimentation.

### Phase 2: Select Optimal Parallelism Strategy

Five guidelines:
1. **Minimize model parallelism, maximize data parallelism**
2. **Keep EP and TP within the NVLink domain**
3. **Use PP for multi-node scaling**
4. **Prefer EP over TP for expert layers**
5. **Enable CP for long sequences**

### Phase 3: Profile and Optimize Bottlenecks

**Memory bottleneck**:
- FP8 training
- selective recomputation
- precision-aware optimizer
- activation offloading
- optimizer offloading

**Communication bottleneck**:
- overlap gradient reduce and parameter gather
- TP communication overlap
- tune EP dispatcher
- overlap EP communication
- tune pipeline layout

**CPU overhead bottleneck**:
- disable Python GC in hot paths
- reduce kernel launches
- enable CUDA Graphs

**Computation bottleneck**:
- Grouped GEMM
- kernel fusions
- FP8 precision

**Key insight**: the optimization loop is iterative. Solving one wall often exposes the next.

---

## 13. RL Post-Training Insights

- **Router Replay**: log expert assignments during inference and replay them during training
- **Packing-aware dynamic batch size**: keep effective token counts stable
- **Attention cost metric**: sort microbatches by $\sum (\text{seq\_len})^2$ to reduce synchronization bubbles
- **Dynamic CP**: choose CP degree per microbatch rather than provisioning for the worst case

---

## 14. Conclusion

MoE sparsity introduces two fundamental problems:
1. **Parameter-compute mismatch**, which creates the Three Walls of memory, communication, and compute efficiency
2. **Dense-sparse mismatch**, which requires decoupled parallelism between attention and expert layers

The overall takeaway is that Megatron-Core MoE is a full-stack systems response to those two mismatches. Memory savings from FP8, recomputation, and offloading are not isolated wins; they enable communication overlap, better parallelism choices, and eventually CUDA Graph coverage. The paper is valuable precisely because it does not present a single trick. It shows how large-scale MoE training becomes feasible only when routing, parallelism, kernels, optimizer states, and long-context execution are all tuned together.
