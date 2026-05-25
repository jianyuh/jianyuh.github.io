---
layout: post
title: "Demystifying Event Tensors and Dynamic Megakernels for LLM Inference"
date: 2026-05-23
categories: [Compilers, Inference]
tags: [Megakernel, Compiler, LLM-Inference, MoE, CUDA, Scheduling, MLSys]
---

Reading notes on:
- [Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernels](https://arxiv.org/pdf/2604.13327), presented at [MLSys 2026]({% post_url 2026-05-24-MLSys-2026 %}).

---

## The Bottleneck in GPU Scheduling

Optimizing latency and hardware utilization is the critical frontier for machine learning systems, particularly in large language model (LLM) inference. Current GPU execution models suffer from two primary overheads:

1. **Kernel launches:** Traditional systems like PyTorch launch kernels sequentially from the host CPU. In auto-regressive decoding, hundreds of fine-grained operations occur per step. A kernel launch takes 5–10 $\mu s$, while the kernel itself might execute in just 2 $\mu s$, making launch overhead the dominant bottleneck.
2. **Kernel boundaries:** Consecutive kernels enforce implicit global synchronization. Even if a downstream kernel only depends on a subset of the previous kernel's output, it must wait for the entire kernel to finish, severely restricting fine-grained inter-kernel parallelism.

While CUDA Graphs alleviate launch overheads by capturing static execution sequences, they fail to break kernel boundaries. **Megakernels** — which fuse multiple operators into a single persistent kernel — eliminate these boundaries and expose inter-kernel parallelism by distributing fine-grained tasks across streaming multiprocessors (SMs). However, traditional megakernels struggle with the inherent **shape dynamism** (e.g., continuous batching) and **data-dependent dynamism** (e.g., Mixture-of-Experts routing) of modern LLMs, often requiring prohibitive recompilation or failing entirely.

---

## The Core Abstraction: Event Tensors

To solve this, the authors introduce the **Event Tensor**, a unified compiler abstraction that encodes fine-grained dependencies between tiled tasks at the SM level. Instead of treating synchronization events as standalone runtime objects, this approach elevates them into **first-class multi-dimensional tensors within the compiler IR**.

To understand how this represents task dependencies mathematically, consider a split-K summation algorithm over an input tensor $A$ with shape $(n \times 32, 128)$. The full operation $C[i] = \sum_{k\in[0,128)} A[i, k]$ is divided into two stages:

1. **Partial sums:** $B[i, j] = \sum_{k\in[j \cdot 32,\; j \cdot 32+32)} A[i, k]$
2. **Final summation:** $C[i] = \sum_{k\in[0,4)} B[i, k]$

Instead of waiting for the entire matrix $B$ to be computed, we partition the work. Each row $C[i]$ only depends on $B[i, :]$. We define tasks and an Event Tensor $E$:

- **Task** $\hat{B}_{i,j}$: Computes $B[i \cdot 32 : i \cdot 32+32,\; j]$
- **Event** $E_i$: Represents $E[i]$
- **Task** $\hat{C}_i$: Computes $C[i \cdot 32 : i \cdot 32+32]$

The dependency relation is compactly modeled as $\hat{B}_{i,j} \rightarrow E_i$ (Task B produces Event E) and $E_i \rightarrow \hat{C}_i$ (Task C consumes Event E). The compiler uses Einsum-like notations (e.g., `in_edges={E: "i->i"}`) to explicitly map these coordinates.

---

## Taming Dynamism

The true power of the Event Tensor lies in handling the unpredictable nature of LLM serving:

- **Shape dynamism:** Because events are mapped to tensors, they inherit deep learning compilers' support for **symbolic shapes**. An Event Tensor can be initialized with a dynamic dimension, such as batch size $B$. This creates a generic dependency template that is instantiated at runtime with concrete values, completely bypassing the need for runtime recompilation or repeated CUDA Graph capture.
- **Data-dependent dynamism:** Irregular task graphs, such as Mixture-of-Experts (MoE) layers, are resolved seamlessly. In MoE, runtime routing decisions stored in a `topk` tensor dictate dependencies. Event Tensors manage this via:
  1. **Data-dependent event update:** The `topk` tensor dynamically determines which grouping tiles notify which expert events.
  2. **Data-dependent task triggering:** An `exp_indptr` tensor (storing the prefix sum of tiles per expert) triggers a variable number of consumer tasks, activating GroupGEMM tiles in the range `(exp_indptr[i], exp_indptr[i+1])`.

---

## Compiler Transformations: Static vs. Dynamic Scheduling

The **Event Tensor Compiler (ETC)** automatically lowers these abstracted graphs into highly optimized persistent megakernels using two distinct scheduling transformations.

### 1. Static scheduling (for predictable workloads)

For workloads with predictable execution times (like dense attention layers or All-Gather operations), ETC uses static scheduling. It pre-computes per-SM execution queues on the host and fuses device functions into a continuous main loop. Dependencies are lowered into raw hardware atomics: `notify()` performs an atomic decrement on an integer tensor, and `wait()` initiates a spin-wait until the counter reaches zero.

*Insight:* This introduces almost zero runtime overhead and excels in latency-critical, low-batch inference. For instance, in a Qwen3-32B dense model (Batch Size=1), static scheduling achieves up to a 1.15× speedup over vLLM.

### 2. Dynamic scheduling (for irregular workloads)

When task completion times are unpredictable (e.g., MoE routing or network-contended Reduce-Scatter), ETC deploys an on-GPU dynamic scheduler. When an event's dependencies complete, the unblocked tasks are atomically pushed to a centralized global memory queue. Idle SMs then atomic-pop tasks from this queue.

*Insight:* This provides exceptional load balancing. In a complete MoE layer (Qwen3-30B-A3B), ETC's dynamic scheduler mitigates straggler SMs, outperforming specialized kernels like FlashInfer by up to 1.23× at 1024 tokens. Furthermore, an "early push" runtime optimization proactively pushes consumer tasks to the queue concurrently with the execution of producer tasks, hiding the scheduling overhead.

---

## System-Level Impact and Insights

By encapsulating control flow explicitly within the compiled code, ETC requires minimal runtime architecture — avoiding the heavy, generic task graph executors standard in previous models.

The most striking architectural advantage is **Ahead-of-Time (AOT) compilation for dynamic workloads**. Highly optimized baseline systems like vLLM and SGLang rely on Just-In-Time (JIT) compilation and require capturing dozens of static CUDA Graphs to accommodate different batch sizes, resulting in massive engine warmup overheads. In evaluating Qwen3-32B, SGLang took 583 seconds and 51 graph captures to warm up; vLLM required 123 seconds and 67 graph captures. By contrast, ETC compiles a single, shape-generic megakernel offline, completing engine initialization in just **35 seconds with zero runtime compilation**.

---

## Conclusion

Event Tensors bridge the gap between fine-grained parallel programming and deep learning tensor compilers. By elevating synchronization primitives to symbolic, multi-dimensional tensor IRs, the abstraction allows automated, compiler-driven fusion of subgraphs (like GEMM + Communication or MoE routing) without sacrificing dynamic shape flexibility. This effectively eliminates kernel launch bounds, smooths wave quantization, and redefines how we can achieve ahead-of-time predictability for heavily dynamic inference servers.
