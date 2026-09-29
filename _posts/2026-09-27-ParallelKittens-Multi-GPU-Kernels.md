---
layout: post
title: "ParallelKittens: Three Principles and Eight Primitives for Multi-GPU Kernels"
date: 2026-09-27
categories: [Systems, Kernels]
tags: [ParallelKittens, ThunderKittens, NVLink, NVSwitch, TMA, Overlap, TensorParallelism, RingAttention, Ulysses, ExpertParallelism, Hopper, Blackwell]
---

Reading notes on:
- [ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels](https://arxiv.org/pdf/2511.13940)

Inside a single GPU, the memory wall is mostly under control. IO-aware algorithms like FlashAttention and tile DSLs like ThunderKittens made HBM traffic something you design around. **Between GPUs, the wall is getting taller.** Inter-GPU communication over NVLink/NVSwitch now accounts for **over 50% of execution time** in LLM training and inference, because the hardware generations have scaled unevenly:

| Resource (A100 → B200) | Scaling |
| :--- | ---: |
| BF16 Tensor Core compute | **7.2×** |
| HBM bandwidth | **5.1×** |
| NVLink bandwidth | **3×** |
| PCIe / InfiniBand bandwidth | **2×** |

For reference, NVLink is 450 GB/s unidirectional per GPU on H100 and 900 GB/s on B200. Each generation, compute per byte of interconnect gets larger, so more of a kernel's wall-clock time goes to communication unless it is hidden behind compute. The existing ways of hiding it each have a flaw:

1. **Bespoke operators** (FLUX, Comet, FlashDMoE) hand-tune one operation in low-level CUDA, CUTLASS, or NVSHMEM. They are fast, but they can't be reused and they break when the architecture changes.
2. **Compiler frameworks** (Triton Distributed) don't adapt to new accelerators. Triton Distributed was developed on H800s and sometimes runs *below* the non-overlapped baseline on H100s.
3. **Communication libraries** (NCCL, NVSHMEM) add synchronization handshakes and intermediate buffering that a fused kernel does not need.

**ParallelKittens (PK)** turns multi-GPU kernel design into **three principles**, then encodes them in a small set of C++ primitives on top of ThunderKittens. Kernels written this way take **under 50 lines of device code** and land within about 10% of hand-tuned state of the art, often ahead of it, across data, tensor, sequence, and expert parallelism. It is the human-written counterpart to the agent- and compiler-driven kernel work in [Cake]({% post_url 2026-08-29-Cake-Compiler-Agent-Co-Design %}), and a general-purpose relative of the NVL72 MoE megakernel in [Mixture-of-Kittens]({% post_url 2026-08-05-Mixture-of-Kittens-MoE-Megakernel %}). It also appeared briefly in my [MLSys 2026 poster notes]({% post_url 2026-05-24-MLSys-2026 %}).

![ParallelKittens: pick the transfer mechanism by message size and SM budget, check the overlap threshold K >= sR/2B, choose intra- or inter-SM scheduling, and strip library overheads; results across tensor, sequence and expert parallelism](/assets/images/parallelkittens_principles.svg)

---

## 1. A Cost Model for Multi-GPU Kernels

PK models a kernel's wall-clock time as

$$T_{\text{kernel}} = T_{\text{launch}} + \max(T_{\text{comp}}, T_{\text{mem}}, T_{\text{comm}}) + T_{\text{non-overlap}} + T_{\text{sync}}$$

- $T_{\text{launch}}$: per-kernel launch cost, meaning host-side latency plus per-thread-block setup and teardown (tensor memory allocation, pipeline fill and drain).
- $T_{\text{comp}}, T_{\text{mem}}, T_{\text{comm}}$: the Tensor Core, local HBM, and interconnect pipelines, with $T_{\text{comm}} = S_{\text{comm}} / B_{\text{comm}}$.
- $T_{\text{non-overlap}}$: communication that could *not* be hidden behind compute.
- $T_{\text{sync}}$: inter-SM or inter-device synchronization latency.

A perfect kernel is a single $\max$. The design goals follow directly: drive $T_{\text{non-overlap}} \to 0$, shrink $T_{\text{sync}}$, and saturate $B_{\text{comm}}$. Each of the three principles targets one of those terms.

---

## 2. Principle 1: Choose the Transfer Mechanism by Granularity

Datacenter GPUs can move data over NVLink in three ways, and they behave very differently:

| Mechanism | Peak BW achieved | Message size to saturate | SMs to saturate | In-fabric reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Copy Engine (CE)** | 81–82% | $\ge$ 256 MB | 0 (host-driven) | No |
| **TMA** | 74–78% | ~2 KB | ~15 | No |
| **Register ops (`ld`/`st`)** | 70–76% | 128 B | ~76 | **Yes** (NVSwitch) |

Three takeaways:

1. **Copy engines win on bulk transfers and lose on tiles.** They reach the highest aggregate throughput (**368.82 GB/s on H100, 726.13 GB/s on B200**), but only with contiguous payloads of $\ge 256$ MB. For tile-granular traffic such as MoE token routing, CE efficiency collapses. Device-initiated methods reach comparable utilization at **2 KB messages**. This is why Triton Distributed, Flux, and CUTLASS, which all use the copy engine for intra-node all-gather GEMM, fall behind the non-overlapped baseline at small matrix sizes.
2. **SM cost differs by 3.2–5.1×.** Register instructions are synchronous and consume registers, so it takes **~76 SMs** issuing concurrently to saturate the link. TMA is asynchronous and issued by a single thread, so **~15 SMs** are enough. Every SM spent on communication is taken away from the GEMM.
3. **Only register ops can reduce in the fabric.** `multimem.ld_reduce` and `multimem.red` are the *only* way to use NVSwitch's hardware-accelerated in-network reduction.

So the choice is not about peak bandwidth, which is within ~10% across all three. It is about **message size, SM budget, and whether you need a reduction**. Taking the CPU off the critical path is the same idea behind [NCCL GIN and MSCCL++]({% post_url 2026-06-24-NCCL-GIN-and-MSCCLpp-GPU-Communication %}). PK applies it inside a fused compute kernel.

---

## 3. Principle 2: Schedule the Overlap, and Know When It Is Possible

### When can communication hide behind compute?

Take a fused $M \times N \times K$ GEMM + reduce-scatter on $P$ GPUs, processing $m \times n \times k$ tiles. One $m \times n$ output tile needs $K/k$ sub-GEMMs, so at sustained Tensor Core throughput $R$:

$$T_{\text{comp,tile}} = \frac{2mnk}{R} \cdot \frac{K}{k} = \frac{2mnK}{R}$$

Sending that tile (element size $s$ bytes) over NVLink bandwidth $B$:

$$T_{\text{comm,tile}} = \frac{s \cdot mn}{B}$$

Overlap is possible when compute per tile exceeds communication per tile:

$$\frac{2mnK}{R} \ge \frac{s \cdot mn}{B} \;\implies\; K \ge \frac{s \cdot R}{2B}$$

The tile dimensions cancel, so **only the reduction dimension $K$ matters**. For H100 in BF16 ($s = 2$, $R = 989$ TFLOP/s, $B = 450$ GB/s):

$$K \ge \frac{2 \times 989 \times 10^{12}}{2 \times 450 \times 10^{9}} \approx 2198$$

The paper checks this with a fused GEMM + RS against a standalone GEMM ($M = N = 32768$):

| $K$ | GEMM (ms) | GEMM + RS (ms) | Non-overlapped comm |
| ---: | ---: | ---: | ---: |
| 512 | 2.071 | 6.483 | 68% |
| 1024 | 2.918 | 6.613 | 56% |
| 2048 | 5.567 | 7.531 | 26% |
| 4096 | 11.780 | 11.828 | **< 1%** |
| 8192 | 23.285 | 25.325 | 8% |

At $K = 2048$, just under the bound, the exposed fraction roughly halves. The authors attribute the remainder to the atomic additions used for output-tile accumulation. At $K = 4096$ communication is essentially fully hidden. The 8% at $K = 8192$ is in the paper's table but isn't discussed in the text. The bound also explains the scaling table above: $R$ grows faster than $B$ each generation, so the threshold $K$ rises, and shapes that overlapped cleanly on the previous GPU stop overlapping on the next one. It is the same roofline reasoning as [designing GEMM shapes for the GPU]({% post_url 2026-08-09-LLM-Hardware-Co-Design-GEMM-Shapes %}), extended with an interconnect ceiling.

### Intra-SM vs. inter-SM overlap

- **Intra-SM:** within one SM, one thread group drives the MMAs while another issues async TMA. Synchronization uses a hardware `mbarrier` at **~64 ns**.
- **Inter-SM:** partition the SMs into a compute group running the local GEMM and a communication group. Synchronization goes through HBM at **~832 ns**, 13× slower.

Intra-SM overlap keeps every SM's Tensor Cores busy and syncs cheaply, so it is the default when $K$ clears the bound. Inter-SM overlap is needed when you want **in-network reduction**. With intra-SM overlap, peer writes serialize over per-port NVLink links. Dedicated communication SMs can instead accumulate partials in HBM and hand one all-reduce to NVSwitch, which cuts $T_{\text{comm}}$ by roughly a factor of the GPU count. The measurements (8× H100, $N = 32768$) show both sides of the trade-off:

| Kernel | No overlap | Intra-SM | Inter-SM |
| :--- | ---: | ---: | ---: |
| GEMM + RS | 510.1 | **743.7** | 618.1 |
| GEMM + AR | 450.9 | 172.3 | **623.9** |

Intra-SM wins for reduce-scatter. For all-reduce, intra-SM is *worse than no overlap*, and inter-SM is **3.62×** faster than intra-SM.

---

## 4. Principle 3: Remove Software Overheads

General-purpose communication libraries pay for generality with latency, and a fused kernel doesn't need that generality:

1. **NCCL handshakes and staging.** NCCL requires two-way sender/receiver synchronization and routes payloads through pre-allocated intermediate channels. PK pre-allocates *destination* buffers on the peer and does **one-way, asynchronous, zero-copy** writes directly into them. On an all-reduce kernel this gives up to **1.79× over NCCL on B200** (1.32× on H100), both at the smallest size tested. The gain shrinks to about 1.04× at the largest size, where NCCL's overheads are amortized.
2. **NVSHMEM address resolution.** The public NVSHMEM API does a global load (`ldg`) on every peer access to resolve the remote virtual pointer, and forces CTA-wide `__syncthreads`. PK keeps **pre-resolved peer pointers in registers**, cutting element-wise NVLink access latency by up to **4.5×** and raising bandwidth utilization by about 20 GB/s.

---

## 5. The Abstractions

### Memory hierarchy, extended across GPUs

| Abstraction | Where it lives | Capacity / bandwidth (H100) |
| :--- | :--- | :--- |
| Register tile `rt<M, N>` | registers | 64K 32-bit registers per SM (the paper writes "64 KB") |
| Shared tile `st<M, N>` | SMEM | 227 KB/SM, ~33 TB/s |
| (L2 cache) | on-chip, shared by all SMs | 50 MB, ~12 TB/s |
| Global layout `gl` | HBM | 80 GB, ~3 TB/s |
| **Parallel global layout `pgl`** | identically allocated tiles on every GPU | NVLink |

The `pgl` is the new piece: memory regions of identical shape and size allocated on every GPU, so a tile coordinate refers to the same logical tile on every peer. P2P transfers, broadcasts, in-fabric multicast and reductions then become ordinary tile operations.

### The LCSC template

Kernels follow a **Load-Compute-Store-Communicate** template with four worker roles:

- **Compute SMs:** a *loader* (TMA reads from local or peer HBM), a *consumer* (Tensor Core or CUDA-core compute), and a *storer* (writes to local or peer HBM). When the loader or storer touches peer HBM, that is intra-SM overlap.
- **Communication SMs:** a *communicator* that occupies one or more SMs exclusively for dedicated communication. That is inter-SM overlap.

The user writes the four functions. The template handles kernel configuration, shared memory and TMA setup, semaphores and barriers, and the split of SMs between compute and communication (`num_comm_sms`), which it auto-tunes.

### Eight primitives

| Primitive | What it does |
| :--- | :--- |
| `store_async(dst, src, idx)` | Async TMA store from an `st` tile to multicast `pgl` memory |
| `store_add_async(dst, src, idx)` | Async atomic-add reduction from `st` to `pgl` |
| `reduce<TILE_ROWS, TILE_COLS, OP>(dst, dst_idx, src, src_idx)` | In-network reducing load (sum, max, or min) from `pgl` into local `gl` |
| `all_reduce<TILE_ROWS, TILE_COLS, OP>(dst_and_src, idx)` | In-place in-network all-reduce over multicast memory |
| `signal(bar, idx, dev_idx, val)` | Atomically add `val` to a specific device's barrier counter |
| `signal_all(bar, idx, val)` | Multicast atomic add to every device's barrier counter in one operation |
| `wait(bar, idx, dev_idx, expected)` | Spin-wait with relaxed ordering until the barrier reaches `expected` |
| `barrier(bar, idx, dev_idx)` | Global multi-device barrier |

Four primitives move data and four synchronize. All of them work on tiles, from 16×16 up to the shared-memory limit (about 256×256), addressed by `int4` coordinates. The P2P stores are asynchronous and issued by a single thread, so they fuse with compute. The network-accelerated reductions need at least a warp for full throughput. That small surface is how a fused MoE dispatch + grouped GEMM fits in under 40 lines of device code.

---

## 6. Under the Hood: CUDA VMM and Multicast Handles

Plain CUDA IPC (`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`) works on existing tensors but cannot use NVSwitch acceleration. PK therefore uses the virtual memory management (VMM) API:

1. Allocate physical memory with **`cuMemCreate`**, setting `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`, then reserve and map a local virtual address with **`cuMemAddressReserve`** + **`cuMemMap`**.
2. Export the physical allocation as a Linux file descriptor with **`cuMemExportToShareableHandle`**. File descriptors are process-local, so the FD is sent as a control message over a **Unix domain socket**.
3. The receiving process imports it with **`cuMemImportFromShareableHandle`**, then reserves and maps its own virtual address.
4. For in-network acceleration, create a **multicast object** with **`cuMulticastCreate`**, register devices with **`cuMulticastAddDevice`**, and bind each device's physical memory with **`cuMulticastBindMem`**. The multicast object is itself exported as a POSIX FD and mapped into every process the same way.

Each process ends up with two addresses: a local one for ordinary loads and stores, and a multicast one. A write to the multicast address is broadcast by the NVSwitch fabric, and PTX `multimem.red` / `multimem.ld_reduce` on it perform in-fabric reductions. A plain *read* from the multicast address is undefined behavior.

VMM allocations carry a 2 MB granularity requirement on H100 and B200, so ordinary `cudaMalloc`-backed PyTorch tensors can't be shared directly. PK needs a custom tensor class with its own VMM allocation. This one-time setup is also what makes the pre-resolved register pointers from Principle 3 possible.

---

## 7. Results

Benchmarks run on 8× H100 SXM5 80GB and 8× B200, connected by NVLink/NVSwitch.

| Parallelism | Baseline | PK relative performance | Residual non-overlapped comm |
| :--- | :--- | :---: | :--- |
| Data / tensor | cuBLAS + NCCL | 1.06–1.68× | < 1% at large $K$ |
| | Triton Distributed | 1.07–5.63× | |
| | Flux | 0.97–2.33× | |
| | CUTLASS | 0.90–7.39× | |
| Sequence (Ring Attention) | xDiT (NCCL P2P + FlashAttention-3) | 1.07–4.08× (paper text) | ~9% |
| Sequence (Ulysses) | YunChang | 1.01–1.39× | |
| Expert (MoE) | Comet | 0.92–1.22× | ~15% |

Ranges below 1× mean a hand-tuned baseline wins at some sizes. The headline "up to 2.33×" for tensor parallelism is the maximum over Flux across all sizes, not a typical speedup.

### Tensor parallelism ($N = 32768$, 8× H100)

- **AG + GEMM:** PK **727** TFLOP/s vs. Flux 662, CUTLASS 628, Triton Distributed 528, cuBLAS + NCCL 493. At $N = 16384$, Flux leads (699 vs. 675).
- **GEMM + RS:** PK **744** TFLOP/s vs. Triton Distributed 602, cuBLAS + NCCL 510, Flux 431, but **CUTLASS is faster at 793**. CUTLASS also leads at $N = 16384$ (640 vs. 575).
- **GEMM + AR** (no Flux or CUTLASS kernel exists): PK **624** vs. cuBLAS + NCCL 451 and Triton Distributed 317.
- **B200 GEMM + RS:** PK **1,409** TFLOP/s vs. cuBLAS + NCCL 960 (1.47×).

So PK is not uniformly fastest. It gets ~1.47× over the non-fused cuBLAS + NCCL pipeline on both generations, beats Flux by 10% on AG + GEMM, and trails CUTLASS by 6% on GEMM + RS. The authors point out that AG + GEMM and GEMM + RS usually run back-to-back, and no single baseline beats PK on the pair. The real claim is consistent near-best performance from a few dozen lines of code, rather than a dedicated kernel project per operator.

### Sequence parallelism

- **Ring Attention:** blockwise attention fused with async KV tile transfers, using inter-SM overlap, reaches **623 TFLOP/s** at sequence length 393,216, 1.28× over xDiT's 488. The gap is largest at the shortest sequence, 12,288: 434 vs. 167, or **2.6×**. There, xDiT's coarse stream-level overlap has too little compute per KV block to hide the transfer. The paper's text states 1.07–4.08×, but no point in its Ring Attention figure (Figure 10) exceeds 2.6×. The 4.08× must come from a configuration that isn't plotted.
- **DeepSpeed-Ulysses:** a fine-grained multi-dimensional all-to-all removes the tensor-reshape overheads, reaching **661 TFLOP/s on H100** (vs. YunChang's 652) and **1,336 TFLOP/s on B200** (vs. 1,297) at sequence length 393,216. The larger gains (up to 1.39×) come at shorter sequences.

Both are the sequence-parallel schemes behind [long-context video training]({% post_url 2026-06-14-Scaling-Video-Training-SP %}). Here the communication is fused into the attention kernel instead of wrapped around it.

### Expert parallelism

The benchmark covers the first half of an MoE layer, token dispatch overlapped with the first expert MLP, at a DeepSeek-V3-like shape (top-8 of 256 experts, $H = 7168$, expert hidden size 2048). PK adds **fewer than 40 lines** of device code to a grouped GEMM kernel:

| Total tokens | 8,192 | 16,384 | 32,768 | 65,536 | 131,072 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| cuBLAS + NCCL | 66 | 136 | 149 | 147 | 150 |
| Comet | 245 | 329 | **425** | **424** | **462** |
| PK | **298** | **374** | 411 | 413 | 426 |

PK leads at small token counts (1.22× at 8K) and trails Comet by 3–8% from 32K tokens up. The paper summarizes this as "matches or surpasses", which is generous: at the largest size, 426 vs. 462 is a 0.92× loss. MoE is also where the residual non-overlapped communication is highest (~15%): data-dependent routing makes the traffic irregular and harder to schedule. That is why dedicated work like [Mixture-of-Kittens]({% post_url 2026-08-05-Mixture-of-Kittens-MoE-Megakernel %}) and [UltraEP]({% post_url 2026-08-12-UltraEP-Exact-Load-Balancing-Rack-Scale-MoE %}) keeps going after the MoE layer specifically.

---

## Takeaways

1. **Don't use copy engines for fine-grained overlap.** They need >256 MB transfers to saturate NVLink. Use async TMA for tile-granular point-to-point traffic, with ~15 SMs and 2 KB messages.
2. **Check the overlap bound before writing the kernel.** If $K \ge sR/2B$ (≈2.2K for BF16 on H100; fully hidden by $K = 4096$ in practice), use intra-SM overlap and keep every Tensor Core busy. If you need in-network reduction, switch to inter-SM overlap and let NVSwitch do the reduction (3.62× on GEMM + AR). The bound rises every generation because $R$ outgrows $B$.
3. **Bypass generic runtimes on the hot path.** Pre-allocated VMM handles and one-way zero-copy writes replace NCCL's handshakes (1.79×). Register-cached peer pointers replace NVSHMEM lookups (4.5×).
4. **A small set of primitives gets close to hand-tuned.** Eight primitives and one LCSC template cover TP, Ring and Ulysses SP, and EP within about 10% of the best hand-tuned kernel everywhere, and ahead of it in many cases. CUTLASS still wins GEMM + RS at large sizes, and Comet wins large-batch MoE dispatch.
