---
layout: post
title: "Infra Math for LLM Training"
date: 2025-11-28
categories: [Training, Infra]
tags: [Training, Infra]
---

Read on the following playbook:
- [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook).


### I. Fundamental Memory Quantification

Training LLMs requires storing four primary components in GPU memory: model weights/parameters ($\Psi$), gradients, optimizer states, and activations.

#### A. Parameter, Gradient, and Optimizer State Memory
The memory required for parameters, gradients, and optimizer states remains constant regardless of batch size or sequence length. Memory consumption varies based on precision and the use of FP32 master weights for stability:

1.  **Full Precision (FP32) Baseline:** Parameters ($4\Psi$) + Gradients ($4\Psi$) + Adam Optimizer States (momentum/variance, $8\Psi$) = **$16\Psi$** bytes per parameter.
2.  **Mixed Precision (BF16 with FP32 Gradients):** This contemporary standard uses BF16 for computation but retains FP32 copies for stability. The total memory consumption is approximately **$20\Psi$** bytes per parameter, calculated as:
    *   BF16 Parameters ($2\Psi$) + BF16 Gradients ($2\Psi$) + FP32 Master Weights ($4\Psi$) + FP32 Gradients ($4\Psi$) + FP32 Optimizer States ($8\Psi$).

#### B. Activation Memory
Activation memory is the dynamic component that grows significantly with inputs. It is particularly challenging because it scales linearly with both sequence length ($seq$) and batch size in samples ($bs$). The total activation memory ($M_{act}$) in mixed precision for a model with $L$ layers, hidden dimension $h$, and $n_{heads}$ is approximated by:
$$M_{act} = L \cdot seq \cdot bs \cdot h \cdot (34 + \frac{5 \cdot n_{heads}}{h})$$

### II. Memory Management Techniques

Techniques like Gradient Accumulation and Activation Recomputation are essential for controlling activation memory growth.

1.  **Gradient Accumulation:** This method allows achieving a large **global batch size ($gbs$)** by splitting it into smaller **micro batch sizes ($mbs$)**. The relationship is defined by:
    $$gbs = mbs \times grad_{acc}$$
    where $grad_{acc}$ is the number of accumulation steps. This technique enables effectively infinite batch sizes without increasing activation memory.
2.  **Activation Recomputation (Gradient Checkpointing):** This approach discards activations during the forward pass and recomputes them during the backward pass to save memory. The trade-off is computation for memory. **Selective Recomputation** is an optimized approach, often focusing on recomputing attention computations while checkpointing computationally expensive feedforward layers.

### III. Distributed Parallelism Strategies (5D Parallelism)

The scaling toolkit consists of five parallelism dimensions: Data Parallelism (DP), Tensor Parallelism (TP), Sequence/Context Parallelism (SP/CP), Pipeline Parallelism (PP), and Expert Parallelism (EP).

#### A. Data Parallelism (DP) and ZeRO
DP replicates the model across $N_d$ GPUs, distributing data (micro-batches). The global batch size formula is expanded to integrate DP:
$$gbs = mbs \times grad_{acc} \times N_d$$

**ZeRO (Zero Redundancy Optimizer)** eliminates memory redundancy in DP by partitioning model components across the $N_d$ Data Parallel ranks. Using $\Psi$ (parameters) and $k$ (memory multiplier for optimizer states, typically $k=12$ if FP32 master weights are kept), the memory per GPU is reduced as follows:

| Stage | Components Partitioned | Memory Consumption (per GPU) |
| :--- | :--- | :--- |
| **Vanilla DP** | None (full replication) | $2\Psi + 2\Psi + k\Psi$ |
| **ZeRO-1** | Optimizer States (OS) | $2\Psi + 2\Psi + \frac{k\Psi}{N_d}$ |
| **ZeRO-2** | OS + Gradients (G) | $2\Psi + \frac{2\Psi}{N_d} + \frac{k\Psi}{N_d}$ |
| **ZeRO-3 (FSDP)** | OS + G + Parameters ($\Psi$) | $\frac{2\Psi + 2\Psi + k\Psi}{N_d}$ |

ZeRO-3 requires continuous parameter gathering (all-gather) during the forward and backward passes.

#### B. Tensor Parallelism (TP)
TP shards weights, gradients, optimizer states, and activations along the **hidden dimension** ($h$), minimizing communication.

1.  **Column Linear:** Splits the weight matrix columns. Requires input **broadcast** and output **all-gather**.
2.  **Row Linear:** Splits the weight matrix rows. Requires input **scatter** and output **all-reduce**.

TP introduces synchronization points (like AllReduce) directly into the computation path (e.g., after the attention block or Feed Forward Layer) that are difficult to fully overlap with computation.

#### C. Sequence and Context Parallelism (SP/CP)
These methods shard tensors along the input sequence dimension ($seq$):

1.  **Sequence Parallelism (SP):** Complements TP by splitting operations not handled by TP (like LayerNorm and Dropout) along $seq$. This reduces the maximal activation size per GPU to $\mathbf{b \cdot s/N_{sp} \cdot h/N_{tp}}$.
2.  **Context Parallelism (CP):** Targets extreme sequence lengths (128k+ tokens) by applying sequence splitting to modules typically handled by TP. Attention modules require communication to exchange key/value (KV) pairs, efficiently managed using **Ring Attention**.

#### D. Expert Parallelism (EP)
EP distributes individual feedforward experts in **Mixture-of-Experts (MoE) models** across different workers. Token routing relies on an **all-to-all** communication operation.

### IV. Pipeline Parallelism (PP)

PP partitions the model layers across different GPUs. This reduces parameter memory per GPU but introduces an efficiency cost called the **pipeline bubble** (idle time).

*   **Ideal Total Training Time** (assuming $t_f$ and $t_b$ are forward and backward times per micro-batch per stage): $t_{ideal} \approx m \cdot (t_f + t_b)$.
*   **Pipeline Bubble Time** (for $p$ stages): $t_{bubble} = (p - 1) \cdot (t_f + t_b)$.
*   **Bubble Ratio ($r_{bubble}$):** The ratio of wasted time to ideal execution time is proportional to the pipeline degree ($p$) and inversely proportional to the number of micro-batches ($m$):
    $$r_{bubble} = \frac{(p - 1)}{m}$$

The **One-Forward-One-Backward (1F1B)** schedule minimizes activation memory by allowing activations to be released sooner, requiring memory storage for only $p$ micro-batches, versus $m$ micro-batches in the naive AFAB schedule. Advanced schedules like **Zero Bubble** and **DualPipe** split the backward pass into computation for input (B) and weights (W) to fill pipeline gaps and achieve near-zero idle time.

### V. Low-Level GPU Optimization and Mixed Precision

GPU throughput relies heavily on custom kernel optimization, exploiting the memory hierarchy (fast SRAM vs. slow Global Memory/HBM).

*   **Fused Kernels:** Combining successive operations (especially point-wise operations like LayerNorm) into a single kernel prevents repeated movement of intermediate results between compute units and global memory.
*   **Flash Attention:** A core kernel optimization for attention that uses **tiling** and **fusing** to compute the attention matrix ($S = QK^T$) entirely on fast **SRAM**, avoiding materializing the large $N \times N$ matrix on slower HBM, which provides both significant speedup and memory saving.

#### Mixed Precision and FP8
Lower precision formats save memory and increase hardware FLOPS, though numerical stability is a concern.

| Format | Total Bits | Sign | Exponent | Mantissa | Key Feature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **float32** | 32 | 1 | 8 | 23 | Default, high range and precision |
| **bfloat16** | 16 | 1 | 8 | 7 | Retains float32 range, sacrificing resolution |
| **float8 (e4m3)** | 8 | 1 | 4 | 3 | High throughput, very small range/resolution |

**FP16/BF16 Training** requires FP32 master weights, loss scaling, and accumulation in FP32 to ensure numerical stability.

**FP8 Training** leverages the fact that NVIDIA H100 GPUs offer twice the theoretical FLOPS for FP8 compared to BF16. While experimental, optimized FP8 techniques (like DeepSeek-V3's tile quantization) aim to reduce the parameter memory burden significantly. For example, some FP8 recipes aim to reduce total memory consumption to around $\mathbf{10\Psi}$ bytes per parameter, compared to the $\mathbf{20\Psi}$ BF16 baseline.


{% comment %}
### I. Notation and Model Sizing

The core analysis relies on a standard set of parameters:
*   $L$: The number of layers.
*   $d_{model}$: The embedding dimension of the transformer model.
*   $d_{head}$: The embedding dimension of each head (typically 128).
*   $n_{heads}$: The number of heads.
*   $n_{kv\_heads}$: The number of heads for K and V (smaller than $n_{heads}$ for Grouped Query Attention (GQA)).
*   $d_{ffn}$: The hidden dimension between Fully Connected (FC) layers in the Feed-Forward Network (FFN).
*   $E$: The number of experts in an MoE configuration.

#### 1. Dense LLM Parameters and FLOPS

In dense models, the majority of parameters reside in linear layers. For modern SwiGLU architectures, the dimension constraint is adjusted to account for 3 FC layers (compared to 2 in older architectures), aiming to keep parameters per layer constant:
$$d_{ffn} \approx \frac{8}{3} \times d_{model}$$

Before GQA, $n_{kv\_heads} = n_{heads}$, leading to the typical parameter count per dense layer:
$$n_{params\_per\_dense\_layer\_typical} = 12 \times d_{model}^2$$

The total parameter count must also include the token embedding and output layers, each contributing $\text{vocab\_size} \times d_{model}$ parameters.

The floating-point operations (FLOPS) for one training iteration involve contributions from linear layers and scaled dot product attention. Assuming three Fused Multiply-Add (FMA) operations per parameter (1 in forward, 2 in backward) and 2 FLOPS per FMA:

$$\text{flops\_from\_linear\_layers\_per\_sequence} = 6 \times n_{params\_per\_dense\_layer}$$

The total FLOPS per dense layer per sequence (with $\text{seqlen}$ and $L=1$) is defined as:
$$\text{flops\_per\_dense\_layer\_per\_sequence} = 6 \times (\text{n\_params\_per\_dense\_layer} + \text{seqlen}^2 \times d_{model})$$

Using the typical parameter approximation ($12 \times d_{model}^2$), this simplifies to:
$$\mathbf{\text{flops\_per\_dense\_layer\_per\_sequence\_typical} = 6 \times (12 \times d_{model} + seqlen^2) \times d_{model}}$$

#### 2. MoE Model Sizing and FLOPS

For Mixture-of-Experts (MoE) models, it is crucial to distinguish between total parameters and *activated* parameters. To match the number of activated parameters to a dense baseline, we often adjust $d_{ffn}$. If MoE is applied every layer, using 1 shared expert, and capacity = 1:

$$\mathbf{d_{ffn, MoE} = \frac{d_{ffn, dense}}{\text{Experts Activated}}}$$

The relationship between total and activated parameters can be approximated for large $E$:
$$\mathbf{n_{params\_per\_moe\_layer\_typical} \approx \frac{2 E}{5} \times n_{activated\_params\_per\_moe\_layer\_typical} \quad \text{when } E \gg 1}$$

The total parameter count for a large MoE model (e.g., Llama 3 405B dense baseline, 32 routed experts, 1 shared expert) can reach approximately 2 Trillion parameters.

Since FLOPS scale with activated parameters, the FLOPS for an MoE layer are defined as:
$$\text{flops\_per\_moe\_layer\_per\_sequence} = 6 \times \text{seqlen} \times n_{activated\_params\_per\_moe\_layer} + 6 \text{ seqlen}^2 \times d_{model}$$

Crucially, when capacity = 1 and the model is sized to match the dense activated parameter count, the typical FLOPS per MoE layer per sequence remain identical to the dense layer FLOPS:
$$\mathbf{\text{flops\_per\_moe\_layer\_per\_sequence\_typical} = 6 \times (2 \times (5 + n_{kv\_heads} / n_{heads}) \times d_{model} + seqlen^2) \times d_{model} = \text{flops\_per\_dense\_layer\_typical}}$$

### II. Communication and Parallelization

This section addresses the communication overheads associated with different parallelization strategies, using $\mathbf{msl}$ (micro sequence length) defined as $\text{micro\_batch\_size} \times \text{seqlen} / CP$.

#### 1. Tensor Parallelization (TP/SP)

For Sequence Parallelization (SP), a variant of TP, the key bottlenecks are the receiver message size of all-gather and the sender size of reduce-scatter.

$$\mathbf{\text{tp\_all\_gather\_size} = msl \times d_{model} \times \text{sizeof}(fp8) = \frac{\text{micro\_batch\_size} \times seqlen}{CP} \times d_{model}}$$
$$\text{tp\_reduce\_scatter\_size} = msl \times d_{model} \times \text{sizeof}(bf16) = 2 \times \text{tp-all-gather-size}$$

The computation-to-communication ratio ($C/C$) during output projection (often the communication bottleneck) for standard 1D TP is:

$$\mathbf{\text{C/C ratio (1D TP)} = \frac{2 \times msl \times d_{model}^2 / TP}{msl \times d_{model} / (TP - 1) \times 2} = \frac{d_{model} \times (TP - 1)}{TP}}$$

For instance, a model with $d_{model} = 32,768$ and $TP=16$ has a C/C ratio of 2,185, suggesting high bandwidth demands (e.g., needing 1.8 TB/s to saturate 4 PF/s peak performance).

#### 2. 2D Tensor Parallelization

To alleviate the extreme bandwidth requirements of 1D TP, 2D TP (decomposed into $TP_x \times TP_y$) is motivated. The computation-to-communication ratio improves:

$$\mathbf{\text{C/C ratio (2D TP)} = \frac{2 d_{model}}{TP \times (\frac{1}{TP_y - 1} + \frac{2}{TP_x - 1})}}$$

#### 3. Data/Context Parallelization (DP/CP)

The computation-to-communication ratio for balancing one micro-batch computation with communication in DP/CP (specifically FSDP/CP) is defined by the ratio of forward FLOPS to the message size of the all-gather operation:

$$\mathbf{\text{C/C ratio (DP/CP)} = \frac{2 \times (2 \times d_{model} \times (1 + n_{kv\_heads} / n_{heads}) + 3 \times d_{ffn} + seqlen^2) \times d_{model}}{(((2 \times d_{model} \times (1 + n_{kv\_heads} / n_{heads}) + 3 \times d_{ffn}) \times d_{model}) / TP)}}$$

### III. Memory Space Needs

#### 1. Activations in Pipeline Parallelization (PP)

Due to typical layer interleaving in PP to reduce the pipeline bubble, recomputation of activations from the earliest layer on a device is usually infeasible. The memory space required for activations is proportional to the number of parameters assigned to that device:

$$\mathbf{\text{PP\_activations\_layer} = n_{params\_per\_dense\_layer} \times \frac{n_{layers}}{PP}}$$

The memory consumption in bytes is dependent on the activation precision:
$$\mathbf{\text{PP\_activations\_layer\_bytes} = n_{params\_per\_dense\_layer} \times \frac{n_{layers}}{PP} \times \text{sizeof(activation\_size)}}$$

Peak memory usage must account for the accumulation of activations across all simultaneous micro-batches being processed in the pipeline.

#### 2. Sharded Optimizer States

When using the AdamW optimizer, two momentum values are required per parameter. Assuming fp32 precision for momentum and parameters, this totals 12 Bytes per parameter (BF16 parameters, BF16 gradients, FP32 momentums. Optimistical as FP32 master weights are ignored, and no FP32 accumulated gradients)

The total memory required for model states (weights + optimizer states):
$$\mathbf{\text{model\_states\_Bytes\_adamW} = 12 \times n_{params}}$$

For sharded implementations (like FSDP), the memory required per accelerator is reduced:
$$\mathbf{\text{model\_states\_Bytes\_per\_accelerator\_adamW} = \frac{12 \times n_{params}}{\text{n\_accelerators\_per\_replica\_group}}}$$
{% endcomment %}
