---
layout: post
title: "NCCL GIN & MSCCL++: Rethinking GPU Communication for Low-Latency AI"
date: 2026-06-24
categories: [Systems, Distributed Training, Inference]
tags: [NCCL, GIN, MSCCL++, GPU Networking, RDMA, NVSHMEM, DeepEP, MoE, Collective Communication, One-Sided Communication]
---

Reading notes on two works that, from different vendors, are converging on the same idea — **take the CPU off the critical path of GPU communication**:
- **NCCL GIN** — [*GPU-Initiated Networking for NCCL*](https://arxiv.org/pdf/2511.15076) (GPU-Initiated Networking in NVIDIA NCCL 2.28's new Device API)
- **MSCCL++** — [*MSCCL++: Rethinking GPU Communication Abstractions for AI Inference*](https://arxiv.org/pdf/2504.09014) (Changho Hwang et al., Microsoft Research & Microsoft Azure) · [code](https://github.com/microsoft/mscclpp)

Large Language Models (LLMs) and Mixture-of-Experts (MoE) architectures have exposed a critical bottleneck in GPU communication: **the CPU control path**. (For background on the MoE expert-parallel communication these libraries accelerate, see [MoE Parallel Folding]({% post_url 2026-01-18-EP %}) and [MoE training with Megatron Core]({% post_url 2026-03-11-MoE-Megatron %}); for the broader GPU mental model, [How to Think About GPUs for LLM Scaling]({% post_url 2026-05-10-Scaling-Book-GPUs %}).) Traditional GPU networking relies on the host CPU to orchestrate transfers, introducing kernel-launch overhead and explicit host-device synchronization that hinder fine-grained, low-latency execution. In real-world inference, communication kernels account for **10%–40%** of total execution time.

Both projects attack this from complementary angles. **NCCL GIN** pushes RDMA control all the way down into CUDA kernels so GPU threads drive the NIC directly. **MSCCL++** exposes a layered, composable set of channel abstractions that decouple data transfer from synchronization, letting developers build asynchronous, overlap-friendly collectives without hand-writing non-portable stacks. This post collects the complete technical notes from both.

---

## Part I — NCCL GIN: GPU-Initiated Networking

### Motivation: Killing the CPU Control Path

Traditional GPU networking models rely heavily on the host CPU to orchestrate communication, introducing kernel-launch overhead and explicit host-device synchronization. With the release of **NCCL 2.28**, NVIDIA introduced the **Device API**, bringing three operation modes directly to the device:

*   **LSA (Load/Store Accessible):** for intra-node PCIe/NVLink.
*   **Multimem:** for hardware multicast.
*   **GIN (GPU-Initiated Networking):** for network RDMA.

GIN completely eliminates CPU coordination overhead, allowing GPU threads to control InfiniBand and RoCE network operations autonomously from within CUDA kernels.

### The Three-Layer Architecture and Dual Backends

GIN introduces minimal overhead via compile-time optimizations and direct hardware access. It operates on a three-layer model:

1.  **NCCL Core (Host-Side):** Manages communicator initialization, GIN resource management, and collective memory-window registration.
2.  **Device-Side API:** Exposes one-sided primitives (e.g., `put`, `signal`) directly callable from CUDA kernels.
3.  **Pluggable Network Backends:** GIN handles varied deployment scenarios via dual semantics.

**GDAKI Backend (Direct GPU-to-NIC).** GDAKI provides fully autonomous device-driven networking. It leverages **DOCA GPUNetIO**, allowing executing GPU threads to construct RDMA work queue entries (WQEs) in device memory and directly ring the NIC's doorbell registers to trigger DMA transfers. Hardware autonomously executes RDMA transactions and updates completion queues, **bypassing the CPU entirely**. This requires modern hardware (**ConnectX-6 Dx+** NICs) and **CUDA 12.2+**.

**Proxy Backend (CPU-Assisted).** Designed for universal hardware portability, this trades peak latency for compatibility. GPU threads enqueue **64-byte descriptors** into lock-free queues. A pinned CPU proxy thread continuously polls these queues, extracts the fields, and posts standard RDMA network operations via network plugins. This backend guarantees GIN semantics on any RDMA-capable NIC or legacy infrastructure.

### Programming Model, Math, and Semantics

GIN shifts away from two-sided handshaking and traditional PGAS (Partitioned Global Address Space) in favor of **Window-based One-Sided Communication**. Buffers are registered across all ranks to establish memory windows that are symmetric in addressability while supporting asymmetric capacity.

**Asynchronous Completion Tracking.** Operations execute asynchronously, and developers track completions using two **ID-based** resources rather than memory addresses:

*   **Counters (Local Completion):** Track when source buffers can be safely reused by the sender.
*   **Signals (Remote Completion):** Symmetric objects that guarantee data arrival and visibility at the destination.

**Ordering Guarantees (The Logic Model).** To maximize network efficiency, GIN operations are fundamentally **unordered by default**. The API provides ordering guarantees *only* between a `put` and a `signal` mapping to the same context and the same peer. When a `signal` (or a `put` with an attached `SignalInc`/`SignalAdd` action) completes remotely, it guarantees that **all preceding put operations to that peer on the same context have completed and are visible to remote GPU threads**. This lets applications batch multiple `put`s and fire a single terminating `signal`, ensuring ordered remote visibility without explicit memory fences.

### Case Study: Integrating GIN into DeepEP for MoE

DeepEP is a highly specialized MoE communication library for dynamic token routing, originally built for [DeepSeek-V3]({% post_url 2024-12-26-deepseek-v3 %}) (whose [hardware-aware design]({% post_url 2025-05-15-DeepSeek-V3-ISCA %}) leans heavily on it). Translating DeepEP's native NVSHMEM backend (pointer-based addressing and memory atomics) to NCCL GIN (window offsets and signal atomics) highlights GIN's customizability.

**Mathematical Mapping for Multi-Communicator Setup.** DeepEP requires high Queue Pair (QP) parallelism (up to **24 QPs** for High-Throughput kernels). Because NCCL GIN provides exactly **4 contexts per communicator**, DeepEP satisfies its QP requirements by deriving the necessary communicators and a deterministic distribution:

*   **Required Communicators** = $\lceil \text{QPs} / 4 \rceil$.
*   **Deterministic Dispatch** = `comm_id = id / 4` and `ctx_id = id % 4`.

**High-Throughput (HT) Kernels.** Optimized for training and prefill (batches of 4096 tokens), HT uses hierarchical symmetric RDMA over remote nodes, forwarded via NVLink. GIN handles circular-buffer flow control (head/tail pointers) by emulating release-acquire semantics: bulk transfers use single-threaded `put()` without immediate signaling, followed by a zero-byte `put()` combined with `SignalAdd` to atomically update remote tail counters only after prior transfers are visible.

**Low-Latency (LL) Kernels.** Optimized for inference-decode (1–128 tokens) using a full all-to-all RDMA mesh. The math behind Streaming Multiprocessor (SM) allocation dynamically distributes experts across warps:

*   **Warp Groups per SM** $G = \lceil N / S \rceil$, where $N$ is total experts and $S$ is available SMs.
*   **Expert Mapping** = `expert_idx = sm_id * G + warp_group_id`.

LL kernels dynamically verify NVLink availability; if absent, GIN's `put()` immediately performs RDMA transfers, delivering per-expert token counts securely using the zero-byte signal strategy.

### Performance Insights

Extensive benchmarks on an NVIDIA **EOS DGXH100** cluster yield highly competitive results for GIN against standalone NVSHMEM libraries:

*   **Microbenchmarks (Latency):** For small messages (4–128 bytes), the GIN **GDAKI** backend achieves **16.7 µs** round-trip latency, outperforming NVSHMEM IBGDA (24.3 µs) and effectively matching the optimized NVSHMEM IBRC (16.0 µs). The **Proxy** backend introduces minimal overhead at **18.0 µs**.
*   **HT Kernel Throughput at Scale:** Across 2 nodes (16 GPUs), GIN sustains **84.36 GB/s** RDMA bandwidth for BF16 dispatch operations, nearly identical to NVSHMEM (84.97 GB/s). At 8 nodes (64 GPUs), this scales to ~53–54 GB/s, matching baseline speeds.
*   **LL Kernel Low Latency:** With pure RDMA (NVLink disabled), GIN dispatches at **160.82 µs** on 1 node, and **219–225 µs** across 8 nodes. Under a hybrid RDMA+NVLink config, GIN consistently delivers lower latency than NVSHMEM (e.g., **9% lower at 2 nodes: 142.51 µs vs 157.00 µs**).

**The Bottom Line:** NCCL GIN bridges the gap between hardware-level device networking and large-scale AI runtime orchestration. By allowing direct GPU-NIC interactions through GDAKI or graceful CPU-assisted degradation via Proxy, GIN natively integrates tightly coupled computation-communication fusion right into NCCL 2.28. LLM practitioners no longer need distinct networking backends for collectives and fine-grained MoE dispatch—it can all reside efficiently in NCCL.

---

## Part II — MSCCL++: Rethinking GPU Communication Abstractions

> **Title:** *MSCCL++: Rethinking GPU Communication Abstractions for AI Inference*
> **Authors:** Changho Hwang et al. (Microsoft Research & Microsoft Azure)

### The Problem with the Status Quo

Vendor libraries like NCCL provide **synchronous** primitives (`send`, `recv`, `copy`, `reduce`) that work over internal buffers, incurring local memory copies and wasting GPU cycles in busy-wait loops. Because they are synchronous, they strictly limit the ability to overlap computation and communication. In an era where AI inference demands a delicate balance of latency (for token decoding) and bandwidth (for prompt prefilling), a one-size-fits-all library forces developers to hand-write custom `AllReduce` kernels—a notoriously complex, hardware-specific task.

### The MSCCL++ Solution: Multi-Layered Abstraction

MSCCL++ solves this via a hierarchical separation of concerns across three layers:

1.  **Primitive API:** A low-level C++/CUDA interface providing zero-copy, one-sided, and asynchronous data-transfer mechanisms mapped directly to hardware capabilities.
2.  **DSL API:** A Python-based Domain-Specific Language allowing users to specify complex communication schedules from a global view of all GPUs. The DSL tracks data dependencies, fuses operations, and removes redundant synchronizations during lowering.
3.  **Collective API:** A drop-in C++ replacement for NCCL/RCCL, bundling heavily optimized algorithms written using the MSCCL++ DSL.

### Deep Dive: The Channel Abstractions

At the core of the Primitive API is the **Channel**, an abstraction mapping to distinct physical interconnect methods. MSCCL++ decouples data transfer from synchronization to enable asynchronous execution.

#### 1. PortChannel (Port-Mapped I/O)

Designed for DMA engines (NVLink) or RDMA NICs (InfiniBand), PortChannel relies on the CPU to initiate data transfers via a lockless request queue shared with the GPU.

*   **Mechanics:** When a GPU thread calls `put`, it writes to the queue head and increments it. A dedicated CPU thread polls the tail, executes the hardware request (e.g., `ibv_post_send` for RDMA), and immediately returns.
*   **Synchronization:** A `signal` request pushes an atomic increment to a receiver's semaphore via CPU (e.g., `ibv_atomic_add`). The receiver calls `wait` (a busy-wait loop) without involving its local CPU.

#### 2. MemoryChannel (Memory-Mapped I/O)

MemoryChannel handles direct thread-copy mechanisms between peer GPUs (PCIe, NVLink). It features two distinct mathematical protocols to balance bandwidth and latency:

*   **High-Bandwidth (HB) Protocol:** Transfers large chunks of data and synchronizes once. Amortizes synchronization overhead but suffers high latency.
*   **Low-Latency (LL) Protocol:** Instead of chunk-level synchronization, it interleaves synchronization flags directly into the data stream. For every $N-1$ elements written to the receiving GPU, `put` writes an integer flag. The receiving GPU uses `read` to check if the flag at index $N$ is set before consuming the $N-1$ elements.
*   **Constraint Derivation:** Because GPUs use a weak memory consistency model (where concurrent writes can land out of order), $N$ cannot be arbitrary. MSCCL++ restricts $N$ to the exact byte sizes of single memory-access instructions: **4, 8, or 16 bytes**.

#### 3. SwitchChannel (Switch-Mapped I/O)

SwitchChannel abstracts in-network capabilities, such as NVIDIA's NVSwitch NVLS multicast and aggregation.

*   **Reduce:** Uses the PTX instruction `multimem.ld_reduce` to fetch values from peer virtual addresses, reduce them in the switch, and return the value to the local GPU.
*   **Broadcast:** Writes a register to a `multimem` address using `multimem.st`; the physical switch pushes it to all peers simultaneously.

### Algorithmic Insight: Overlapping Compute and Communication

The zero-copy, asynchronous nature of MSCCL++ allows mathematical splitting of communication rings to overlap bandwidth with reduction computation.

In a standard Ring ReduceScatter, a buffer of size `sz` is split across $N$ GPUs into chunks of size `csz = sz / N`. In MSCCL++, the chunk offsets are calculated iteratively over $N$ steps:

$$\text{offset} = ((rank + N - step) \bmod N) \times csz$$

To hide latency, MSCCL++ slices this `csz` chunk further into two halves:

1.  The GPU initiates a one-sided asynchronous `put` for the **first half**.
2.  While the first half travels the network, the GPU locally `reduce`s the **second half** from the *previous* step.
3.  The GPU waits for the first half to arrive, flushes, and initiates a `put` for the **second half**.
4.  It simultaneously `reduce`s the **first half** of the current chunk.

This creates a tight, fully pipelined overlap of compute and communication within a single thread block.

### Key Takeaways for Domain Experts

**1. Latency vs. Bandwidth Dominance in AI Inference.** MSCCL++ proves that AI inference requires different algorithmic topologies depending on message size. For small messages (e.g., token decoding in LLMs), MSCCL++ uses a **One-phase All-pairs (1PA)** algorithm with the LL Protocol, maximizing parallel dispatches and relaxing synchronization to cut **1KB AllReduce latency by 47%** vs. MSCCL (from 9.5 µs to 5.0 µs). For larger messages (e.g., prompt prefilling), a **Two-phase Hierarchical (2PH)** algorithm minimizes cross-node traffic.

**2. Drastic Reductions in Code Complexity and Portability.** Because the MSCCL++ Primitive API hides consistency models but preserves physical hardware abstractions, porting logic to new architectures is trivialized. While AMD's RCCL requires **~35,480 lines** of diverged code, MSCCL++ was ported to the **AMD MI300x in just 7 weeks with fewer than 10 lines** of AMD-specific low-level code. Adding support for **DeepEP** (Expert Parallelism for DeepSeek-V3), which natively relied on proprietary NVIDIA InfiniBand GPUDirect Async (**IBGDA**), required simply routing it through MSCCL++ `PortChannel`s to achieve hardware parity without proprietary lock-in.

**3. Real-World Speedups.** In end-to-end benchmarks using **Llama3-70b** (via [vLLM]({% post_url 2025-11-30-vLLM %})) and **DeepSeek-V3** (via SGLang), MSCCL++ delivered up to **1.11×** and **1.31×** faster decode latencies over NCCL baselines.

**Conclusion:** MSCCL++ represents a critical paradigm shift. By exposing raw data-transfer methods without locking developers into monolithic, synchronous APIs, it transforms collective communication from a black-box vendor bottleneck into a highly tunable, composable execution graph.

---

## Putting It Together: Two Roads to the Same Destination

NCCL GIN and MSCCL++ are independent answers to the same diagnosis—**the CPU control path is the enemy of fine-grained GPU communication**—and they share a striking amount of design DNA:

| Dimension | NCCL GIN | MSCCL++ |
|---|---|---|
| **Vendor / origin** | NVIDIA (NCCL 2.28 Device API) | Microsoft Research & Azure |
| **Core abstraction** | Window-based one-sided `put`/`signal` from CUDA kernels | Channels (Port / Memory / Switch) decoupling transfer from sync |
| **Direct GPU→NIC path** | GDAKI (DOCA GPUNetIO, ring NIC doorbell) | — (RDMA via PortChannel CPU proxy) |
| **CPU-assisted fallback** | Proxy backend (64-byte descriptors, polling thread) | PortChannel (lockless queue + polling thread) |
| **Low-latency trick** | Zero-byte `put` + `SignalAdd` for ordered visibility | LL protocol — flags interleaved in data stream (N = 4/8/16 B) |
| **In-network offload** | Multimem multicast (sibling Device API mode) | SwitchChannel (`multimem.ld_reduce` / `multimem.st`) |
| **MoE proof point** | DeepEP ported off NVSHMEM | DeepEP ported off IBGDA |
| **Portability story** | Dual backend (GDAKI vs Proxy) | <10 LoC to MI300x; drop-in NCCL/RCCL replacement |

The shared threads worth internalizing:

1.  **One-sided, asynchronous, zero-copy is the new default.** Both reject synchronous two-sided handshaking in favor of `put` + completion signals, so the GPU never stalls on the host.
2.  **Decouple data movement from synchronization.** GIN's counters/signals and MSCCL++'s separate sync flags both let you batch bulk transfers and fire one terminating signal—exactly the pattern that enables ordered remote visibility without fences.
3.  **DeepEP is the canonical stress test.** Both papers prove their model by re-hosting DeepSeek's MoE all-to-all dispatch (off NVSHMEM for GIN, off IBGDA for MSCCL++)—a sign that **dynamic MoE token routing** is now the workload defining the frontier of GPU communication.
4.  **Latency and bandwidth want different algorithms.** GIN splits into LL vs HT kernels; MSCCL++ splits into 1PA (LL protocol) vs 2PH. Message size dictates topology in both worlds.

The convergence is the real story: device-initiated, composable communication is moving from bespoke research backends into the mainstream runtimes (NCCL itself, plus a portable NCCL-compatible layer), so LLM practitioners can finally get tightly fused compute-communication overlap without hand-rolling non-portable stacks.
