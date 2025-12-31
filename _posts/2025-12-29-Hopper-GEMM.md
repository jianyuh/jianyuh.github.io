---
layout: post
title: "High-Performance Matmul Kernels on NVIDIA Hopper"
date: 2025-12-29
categories: [GEMM]
tags: [GEMM]
---

Reading the following blog:
- [Inside NVIDIA GPUs: Anatomy of high performance matmul kernels](https://www.aleksagordic.com/blog/matmul)

#### **1. Architectural Foundations: The Hierarchy of Constraints**
To understand kernel performance, one must internalize the "physics" of the hardware. The H100 architecture dictates that performance is a function of managing the memory hierarchy and maximizing compute throughput under power constraints.

*   **Memory Hierarchy & Latency:** The system is organized by a tradeoff between capacity and speed. Data moves from high-capacity/high-latency Device Memory (HBM) $\to$ L2 Cache $\to$ L1/Shared Memory (SMEM) $\to$ Registers (RMEM). The goal is to keep frequent data as close to the compute units (SMs) as possible.
*   **The Streaming Multiprocessor (SM):** The fundamental unit of compute. The H100 (SXM5) exposes 132 SMs. Crucially, while an SM can host up to 2,048 concurrent threads (hiding latency), it can only issue instructions for 128 threads (4 warps) simultaneously per cycle.
*   **Speed of Light (SoL):** The theoretical ceiling of performance is volatile. While calculated as `freq * num_tc * flops_per_tc`, the actual clock frequency fluctuates based on power and thermal throttling, meaning the effective SoL is dynamic.

#### **2. The Software Layer: CUDA, PTX, and SASS**
Software abstractions map directly to hardware resources. A **Thread Block** must contain at least 4 warps (128 threads) to keep the SM’s 4 warp schedulers busy.

*   **ISA Nuances (PTX vs. SASS):**
    *   **PTX** is a virtual, forward-compatible ISA.
    *   **SASS** is the native ISA. Performance engineers must analyze SASS because it reveals exactly how the compiler lowers instructions (e.g., loop unrolling, register banking). The last few percent of performance—critical at scale—often resides in SASS optimizations.
*   **Quantization Effects:**
    *   **Tile Quantization:** If the matrix size isn't divisible by the tile size, threads at the edge do no useful work.
    *   **Wave Quantization:** If the number of thread blocks isn't a multiple of the GPU's resident block capacity, the final "wave" of execution may leave SMs idle, nearly doubling execution time for marginal work.

#### **3. Evolution of Matrix Multiplication Kernels**

**A. The Baseline: Naive & Warp-Tiling (Synchronous)**
The naive kernel illustrates the importance of **global memory (GMEM) coalescing**. Simply swapping loop indices (changing from row-major to column-major access) can cause a 13x slowdown due to the physics of DRAM row activation.

To optimize, we use **Warp-Tiling**:
*   **Concept:** Matmul is computed as a sum of partial outer products. Blocks of data are loaded into Shared Memory (SMEM) to maximize reuse.
*   **Bank Conflicts:** SMEM is composed of 32 banks. If multiple threads in a warp access different addresses in the same bank, accesses are serialized. Swizzling is required to avoid this.
*   **Arithmetic Intensity:** Performance is memory-bound until the ratio of FLOPs to Bytes loaded exceeds the hardware's "ridge point" (approx. 410 for H100 PCIe). Using square tiles maximizes this intensity.

**B. The Hopper Standard: Asynchronous & Tensor Cores**
Achieving State-of-the-Art (SOTA) on H100 requires moving from synchronous execution to asynchronous pipelines using specific hardware features.

*   **TMA (Tensor Memory Accelerator):** A hardware engine that asynchronously copies data between GMEM and SMEM. It relieves threads from managing loads and automatically handles **swizzling** (using XOR masks) to map data into SMEM such that rows and columns can be accessed without bank conflicts.
*   **WGMMA (Warpgroup MMA):** Hopper introduces asynchronous matrix multiply instructions that operate on "warp groups" (128 threads). The `wgmma.mma_async` instruction requires fencing and "commit groups" to manage data dependency, but allows the SM to queue matrix multiplications while doing other work.

#### **4. Designing the SOTA Pipeline**
The transition from 32 TFLOP/s (naive) to ~760 TFLOP/s (SOTA) involves strict pipeline orchestration.

1.  **Producer-Consumer Model:**
    *   Work is split between **Producer Warps** (driving TMA to load data into a circular buffer in SMEM) and **Consumer Warps** (driving Tensor Cores).
    *   Synchronization is handled via **mbarriers** (memory barriers) tracking "full" and "empty" states of the buffer.
2.  **Persistent Kernels:** Instead of launching a block per tile, launch a fixed number of blocks (one per SM) that loop over the workload. This hides the latency of the kernel epilogue (writing results back to GMEM) by overlapping it with the compute of the next tile.
3.  **Thread Block Clusters:** Grouping blocks allows SMs to access each other's SMEM (Distributed Shared Memory). This reduces L2 cache traffic by enabling data reuse across the cluster.
4.  **Swizzling Logic:** The hardware uses specific XOR masks (e.g., `Swizzle<3,4,3>`) to permute address bits. This ensures that logically contiguous vectors (rows or columns) are physically distributed across different memory banks.
5.  **Scheduling:** Using a **Hilbert Curve** traversal order for processing tiles improves L2 cache locality compared to naive row-major scanning.
