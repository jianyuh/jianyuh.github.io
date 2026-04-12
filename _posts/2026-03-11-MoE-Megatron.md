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

**Full vs Partial**: Full CUDA Graphs capture the entire forward-backward pass, but only for drop-and-pad MoE. **Partial CUDA Graphs** capture only the static components:
- **Graphable**: attention, router, EP preprocessing, shared experts, dense MLP
- **Not graphable**: token dispatch, expert GEMM with dynamic M dimension, token combine

Memory optimizations:
- **Graph count reduction**: without PP, microbatches share graphs (`L x 2`); with PP, each microbatch needs its own (`L x M x 2`). An `is_first_microbatch` GPU flag controls microbatch-specific behavior.
- **Pool sharing**: all graphs share one pool when captured in execution order.
- **Buffer reuse**: static I/O buffers are reused across graphs per PP execution order.

Result: about **10% end-to-end speedup** with roughly **7 GB** extra memory on DeepSeek-V3 GB200.

### Full CUDA Graphs for Dropless MoE: Three Complementary Techniques

**Challenge 1**: kernel launch without knowing problem size.
Solution: **device-initiated Grouped GEMM**, where cuBLASLt reads shapes directly from device memory. The cuteDSL path can also fuse SwiGLU and quantization into the epilogue.

**Challenge 2**: memory allocation without knowing actual size. Without mitigation, worst-case buffers waste $O(\text{EP\_size})$ more memory.

#### ECHO (Elastic Cloning for Hot Experts)

Popular experts can receive far more tokens than others. ECHO dynamically clones hot experts onto spare slots on underutilized ranks.

**Forward**: the ECHO planner identifies hot experts, generates the hot expert map and updated routing map, copies weights to spare slots through HybridEP, and routes tokens to both the home and cloned experts.

**Backward**: expert-gradient dispatch collects gradients from cloned experts back to the home experts.

#### Paged Stashing

Decouples worst-case computation buffers from activation storage:
- **Single tmp buffer** sized for worst case and shared across all layers
- **Paged stashing buffer** stores only the actual tokens used per layer

This reduces memory from $O(\text{layers} \times \text{worst\_case})$ to $O(\text{worst\_case} + \text{actual\_total})$.

Implementation detail: `PagedStashBuffer` uses 64 tokens per page with a circular-buffer free list. Stash and reload kernels are device-initiated and overlap with computation via dedicated Pack and Unpack CUDA streams.

---

## 7. Reduced-Precision Training (FP8/FP4)

### Strategy: Selective Precision

Three principles:
1. **Protect routing**: keep the router in FP32
2. **Preserve key components**: embeddings, output layers, main gradients, master weights, and optimizer states stay high precision
3. **Quantize bulk computation**: expert GEMMs and activations go low precision

### FP8 Recipes

**Per-Tensor FP8** (Hopper and Blackwell): one scale per tensor. Two variants are discussed:
- **Delayed scaling**: uses historical `amax`, but is not the recommended path
- **Current/live scaling**: computes scale just in time

The hybrid format uses E4M3 for inputs and weights and E5M2 for gradients.

**Blockwise FP8** (recommended on Hopper): uses E4M3 for all tensors, with activations and gradients quantized in `1 x 128` tiles and weights in `128 x 128` blocks. The paper notes this recipe has already been proven at scale on models such as DeepSeek-V3, Minimax-M2, and Ant Ling-2.0.

**MXFP8** (recommended on Blackwell): uses `1 x 32` element granularity with native Blackwell Tensor Core support and hardware-accelerated scaling. The important caveat is that parameter AllGather still communicates in BF16 because MXFP8 uses different quantization directions for forward and backward.

### NVFP4

Uses E2M1 with **two-level microscaling**:
- **Per-tensor FP32 scale**: remaps the overall distribution into a range compatible with block scaling
- **Per-block E4M3 scale**: blocks of 16 elements then map into FP4 range

Three critical algorithmic additions are required for stable training:
1. **Random Hadamard Transforms (RHT)**: applied to weight-gradient computation to reduce outlier impact
2. **2D scaling**: `16 x 16` weight-block scaling keeps forward and backward consistent
3. **Stochastic rounding**: applied on gradients to reduce rounding bias during FP4 conversion

### FP8/FP4 Primary Weights

The framework eliminates a redundant BF16 copy by casting directly from FP32 to FP8 or FP4. The distributed-optimizer quantization flow is:
1. Get local abs-max from the master weights
2. AllReduce to get global abs-max
3. Use global abs-max plus master weights for the partial cast

For the blockwise recipe, specialized kernels aware of the 2D weight layout compute abs-max over `128 x 128` blocks.

### MoE-Specific FP8/FP4 Challenges

**Padding alignment**: FP8 GEMMs require alignment to 16 for per-tensor and blockwise FP8, or 32 for MXFP8 and NVFP4. Because token dimension varies dynamically, the paper describes two solutions: routing-map padding and fusing padding directly into permutation.

**Grouped quantization**: multiple expert input tensors are quantized in a single fused kernel, and the implementation is CUDA-Graphable.

**NVFP4 quantization fusion**: RHT is fused with quantization to avoid extra BF16 traffic. Hadamard is still computed twice, once for `amax` and once inside the fused quantization path, but this remains faster than materializing a full BF16 buffer. Stochastic rounding uses `cuRANDDx` for in-kernel random-number generation.

---

## 8. Long-Context MoE Training

### The Computational Shift

At 64K tokens, **SDPA consumes 69% of FLOPs**, versus roughly 10-15% at short sequence lengths. SDPA scales as $O(s^2)$ while MoE scales as $O(s)$, so attention becomes the dominant cost.

**Key recommendation**: do not recompute core attention at long sequence lengths. At 64K, SDPA recomputation adds about **18% compute overhead** but saves only **9 GB** of memory. Recomputing non-SDPA components instead saves **89.8 GB** with lower performance impact.

### CP vs TP Trade-offs

- **P2P CP**: preferred across nodes because ring-style KV exchange overlaps naturally with SDPA
- **All-to-all CP**: converts sequence-sharded layouts to head-sharded layouts before SDPA
- **TP**: preferred within nodes for sharding linear weights

Practical guideline: **all-to-all CP + TP inside nodes; P2P CP across nodes**.

### Dynamic Context Parallelism

For variable-length sequences, the system selects CP degree per microbatch based on actual sequence lengths. Multiple CP groups are pre-constructed per rank during initialization. The per-token loss is:

$$\mathcal{L} = \frac{\sum_{t \in \mathcal{V}} \ell_t}{|\mathcal{V}|}$$

where $\mathcal{V}$ is the set of valid non-padding tokens.

---

## 9. Production Features

### Load Balancing Strategies

Three approaches are discussed:
- **Auxiliary loss**: gradient-based, differentiable, soft balance
- **Sinkhorn**: assignment-based, non-differentiable, hard balance; iterates row and column normalization to convergence
- **Aux-loss-free / Expert Bias**: feedback-based, non-differentiable, adaptive; updates expert bias based on token-count feedback

### Latent MoE

Latent MoE inserts a shared down-projection before expert dispatch and an up-projection after combine:

$$\text{output}(\mathbf{x}) = \mathbf{W}_\uparrow \cdot \left(\sum_{i \in \mathcal{T}_{K,E}} p_i E_i(\mathbf{W}_\downarrow \cdot \mathbf{x}; \ell)\right) + \sum_j E_j^\text{shared}(\mathbf{x}; d)$$

The compression ratio $\alpha = d / \ell$ reduces both all-to-all volume and per-expert weight size by a factor of $\alpha$.

### Flexible Asymmetric VPP

Allows different numbers and types of layers per virtual pipeline stage. For DeepSeek-V3 with 61 decoder layers plus 1 MTP layer, `PP=16`, and `VPP=2`, the first stage holds the embedding plus 3 dense decoder layers to match the cost of 2 MoE layers, while the MTP layer sits in its own standalone stage and the loss is separated.

### Upcycling

Converts a dense checkpoint into MoE via virtual-group initialization: shard MLP weights in the intermediate dimension (`4h -> 2h`), duplicate the shards, then initialize half the router weights and duplicate them. This guarantees Top-2 routing initially selects one expert from each shard pair, so the MoE output exactly matches the dense model at the start of training.

### Multi-Token Prediction (MTP)

MTP optimizes the model to predict multiple consecutive future tokens at each position, densifying the supervision signal. Unlike parallel independent predictions, it preserves **causal dependencies** between predictions through hidden-state transitions, which improves convergence and generation quality. During inference, the model falls back to ordinary single-token prediction for compatibility.

Flexible pipeline parallelism allows MTP layers to be placed strategically within the VPP layout. In the DeepSeek-V3 example with `PP=16` and `VPP=2`, the MTP layer is placed in a standalone pipeline stage on PP rank 14 to balance the workload.

### Muon Optimizer

Unlike AdamW, which performs element-wise updates, Muon applies a **matrix-aware** optimization by orthogonalizing entire weight matrices. The production integration provides:
1. **Split QKV support**: efficient orthogonalization even when attention projection matrices are stored as separate Q, K, and V tensors
2. **Distributed optimizer integration**: optimizer states are sharded across data-parallel ranks while preserving correct orthogonalization semantics
3. **CPU offloading**: Muon's orthogonalization buffers can be offloaded when GPU memory is tight

**MuonClip** addresses a separate stability problem in trillion-parameter training, where query-key dot products can grow without bound and trigger attention explosions. The paper notes hardware-accelerated implementations in cuDNN, `cudnn-frontend`, and Transformer Engine.

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

**Key insight**: the same model requires fundamentally different strategies on different hardware. GB200's 192 GB memory and NVL72 topology largely eliminate the communication wall, shifting the bottleneck toward CPU overhead and making CUDA Graphs essential. H100's 80 GB memory and NVL8 topology instead make cross-node communication the main bottleneck, so EP overlap becomes essential, and FP8 is what frees enough memory to hold the extra overlap buffers.

---

## 12. Systematic Optimization Workflow

A three-phase workflow emerges from tuning Mixtral, DeepSeek-V3, and Qwen3 across GB200 and H100. The process is inherently **iterative**: solving one bottleneck often exposes the next.

### Phase 1: Establish Memory-Feasible Parallelism

Memory feasibility is the first hard constraint. The paper explicitly frames the impact of each parallelism strategy on per-GPU memory:

| Strategy | Peak Activation | Weight Memory | Optimizer States | Comm (Per-Layer) |
|----------|----------------|---------------|-----------------|-----------------|
| TP | $1/d$ (with SP) | $1/d$ | $1/d$ | High |
| EP | ~1 (load-dependent) | $1/d$ (MoE only) | $1/d$ | Medium |
| PP | 1 (>1 with VPP) | $1/d$ | $1/d$ | Medium |
| CP | $1/d$ | 1 | $1/d$* | Medium |
| DP | 1 | 1 | $1/d$* | Low |

`*` Requires distributed optimizer.

**Practical tip**: use `--fake-init-process-group` to emulate distributed training on a single GPU for rapid iteration on parallelism configurations without allocating a full cluster.

### Phase 2: Select Optimal Parallelism Strategy

Five guidelines:
1. **Minimize model parallelism, maximize data parallelism**: keep TP, EP, PP, and CP as small as possible while still avoiding OOM. Use the distributed optimizer (`--use-distributed-optimizer`) to shard optimizer states across DP ranks.
2. **Keep EP and TP within the NVLink domain**: make sure `EP x TP` fits inside the local NVLink island, typically 8 GPUs per node or 72 GPUs for NVL72. When scaling beyond that, prefer PP over stretching TP or EP across nodes.
3. **Use pipeline parallelism for multi-node scaling**: PP distributes layers across nodes. Enable VPP to reduce pipeline bubbles when `PP >= 2`, and balance work across VPP ranks.
4. **Prefer EP over TP for expert layers**: EP yields larger local GEMMs, lower communication overhead, simpler computation graphs, and eliminates local token permutation when `EP = num_experts`. The paper gives a concrete example: Mixtral-8x7B with `EP8 x TP1` outperforms `EP4 x TP2`.
5. **Enable context parallelism for long sequences**: use CP when sequence length is at least about 8K tokens. For sequences shorter than about 4K, the CP overhead can exceed the benefit.

### Phase 3: Profile and Optimize Bottlenecks

Diagnose which wall dominates, then apply targeted fixes.

**Memory bottleneck**: symptom is forced full recomputation or overly aggressive parallelism just to avoid OOM.

| Optimization | Overhead | Config Flag |
|-------------|----------|-------------|
| FP8 Training | Low | `--fp8-format --fp8-recipe` |
| Selective Recomputation | Low | `--recompute-granularity --recompute-modules` |
| Precision-Aware Optimizer | Low | `--use-precision-aware-optimizer` |
| Activation Offloading | Medium | `--fine-grained-activation-offloading --offload-modules` |
| Optimizer Offloading | Medium | `--offload-optimizer-states` |

**Communication bottleneck**: symptom is a profile dominated by collectives.

| Communication Type | Config Flag |
|-------------------|-------------|
| DP gradient reduce/param gather | `--overlap-grad-reduce --overlap-param-gather` |
| TP communication | `--tp-comm-overlap` |
| EP dispatcher | `--moe-token-dispatcher-type` |
| EP all-to-all hiding | `--overlap-moe-expert-parallel-comm` |
| PP send/recv | `--pipeline-model-parallel-layout` |

**CPU overhead bottleneck**: symptom is Nsight Systems showing gaps between GPU kernels.

| Optimization | Config Flag |
|-------------|-------------|
| Disable Python GC | `--manual-gc --manual-gc-interval 10` |
| Reduce kernel launches | Decrease TP or increase MBS |
| Enable CUDA Graphs | `--cuda-graph-impl transformer_engine` |

**Computation bottleneck**: symptom is low SM utilization even after communication and CPU gaps are under control.

| Optimization | Config Flag |
|-------------|-------------|
| Grouped GEMM | `--moe-grouped-gemm` |
| Kernel fusions | `--moe-router-fusion --moe-permute-fusion` |
| FP8 precision | `--fp8-format --fp8-recipe` |

**Key insight**: the same model on different hardware needs different optimization priorities. On NVL8, where EP crosses nodes, the Communication Wall dominates and can consume 30-50% of step time in all-to-all. On NVL72, where EP stays inside NVLink, enabling FP8 often shifts the bottleneck to CPU overhead instead.

### Iterative Nature

The ordering matters: memory in Phase 1, then parallelism in Phase 2, then profiling in Phase 3. But the process is cyclical. Memory optimizations may enable smaller parallelism degrees, which pushes you back to Phase 1. Phase 3 optimizations such as EP communication overlap and CUDA Graphs also consume memory, which can force you to revisit earlier choices.

---

## 13. RL Post-Training Insights

- **Router Replay**: logs expert assignments during inference and replays them during training for more stable optimization
- **Packing-aware dynamic batch size**: keeps total effective tokens per batch more consistent
- **Attention cost metric**: sorts microbatches by $\sum (\text{seq\_len})^2$ in serpentine order to reduce synchronization bubbles
- **Dynamic CP**: selects CP degree per microbatch rather than provisioning for the worst-case sequence mix

---

## 14. Conclusion

MoE sparsity introduces two fundamental challenges:
1. **Parameter-compute mismatch**, which creates the Three Walls of memory, communication, and compute efficiency
2. **Dense-sparse mismatch**, which requires decoupled parallelism between attention and expert layers

Key contributions summarized:

| Contribution | Impact |
|-------------|--------|
| Parallel Folding | Breaks `EP <= DP`; enables flexible parallelism mapping |
| Memory Optimization | 199.5 GB to under 80 GB per GPU for DeepSeek-V3 |
| Communication Optimization | All-to-all shifts from foreground bottleneck to mostly background work |
| Compute Efficiency | Grouped GEMM plus CUDA Graphs plus sync-free execution |
| FP8/FP4 Training | Cross-cutting improvements across all three walls |
| Long-Context Training | Keeps sub-sequence lengths manageable through CP and TP scaling |
| RL Support | Packed sequences, Dynamic-CP, and router replay |

Final performance: DeepSeek-V3 reaches **1,233 / 1,048 TFLOPS/GPU** on GB300 and GB200 with 256 GPUs, versus **368 TFLOPS/GPU** on H100 with 1,024 GPUs. GB200 and GB300 deliver about **3x higher token throughput** than H100.

The broader takeaway is that Megatron-Core MoE is a full-stack systems response to these two mismatches. Memory savings from FP8, recomputation, and offloading are not isolated wins; they enable communication overlap, better parallelism choices, and eventually CUDA Graph coverage. Large-scale MoE training becomes feasible only when routing, parallelism, kernels, optimizer states, and long-context execution are all tuned together.
