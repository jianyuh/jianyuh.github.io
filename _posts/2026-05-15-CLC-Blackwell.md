---
layout: post
title: "Cluster Launch Control: Hardware-Driven Dynamic Tile Scheduling on Blackwell"
date: 2026-05-15
categories: [CUDA]
tags: [CUDA, NVIDIA, Blackwell, GEMM, Scheduling, CuTe, PTX]
---

Reading notes based primarily on:
- [Dynamic Persistent Tile Scheduling with Cluster Launch Control (CLC) on NVIDIA Blackwell GPUs](https://research.colfax-intl.com/dynamic-persistent-tile-scheduling-with-cluster-launch-control-clc-on-nvidia-blackwell-gpus/)

**Cluster Launch Control (CLC)** is the new Blackwell PTX mechanism that pushes dynamic GEMM tile scheduling from a software atomic counter into hardware.

This connects directly to the broader Blackwell story from [SM100: TMEM, TMA, and the New Tensor Core Roofline]({% post_url 2026-04-12-blackwell-sm100 %}). CLC is the third leg of "asynchronous clusters as the execution unit": after TMEM (where accumulators live) and TPC-scoped 2CTA MMA (how a cluster issues math), CLC is **how clusters get more work**.

---

## 1. The GEMM Tile Scheduling Problem

A GEMM `C = A·B` of shape `(M, N, K)` decomposes into work tiles of shape `(bM, bN, bK)`. Each output tile is

$$C_{[i,j]} = \sum_k A_{[i,k]} B_{[k,j]}$$

Tile scheduling is the policy that maps the grid of `(M/bM, N/bN)` output tiles onto cluster thread blocks (CTAs). Historically there are two competing approaches, and both have a structural flaw on heterogeneous workloads.

### Single-tile scheduling

Launch a CUDA grid of shape `(M/bM, N/bN)` and compute exactly one output tile per cluster. Load balancing is trivial — the GPU's hardware scheduler picks up the next cluster as soon as an SM is free. But every cluster pays the full startup cost: pipeline construction, descriptor setup, prologue TMA loads. There is also no way to overlap the **epilogue of tile N** with the **mainloop of tile N+1**, which is critical at small K where the epilogue is a non-trivial fraction of cycles.

### Static persistent scheduling

Launch exactly as many clusters as the GPU can hold concurrently — on a B200 with 148 SMs and a 2-CTA cluster, that's 74 clusters. Statically partition the tile list into 74 chunks. Each cluster loops over its chunk, paying startup cost once and overlapping consecutive tiles within the loop.

This is great on uniform GEMMs. It is **catastrophic** on grouped GEMMs where K varies per problem. With a tile shape of `(128, 128, 128)`:

| Problem | K     | FLOPs/tile |
| :------ | :---: | :--------: |
| A       | 128   | 2²² ≈ 4M   |
| B       | 2048  | 2²⁶ ≈ 64M  |

If the static partition gives every cluster the same number of tiles, the clusters holding Problem B do **16× the work** of those holding Problem A. The tail is set by the slowest cluster, and most of the SMs sit idle waiting.

---

## 2. Software Dynamic Scheduling: An Atomic Counter

The standard fix is a persistent kernel with a **global atomic fetch-and-increment counter**. A cluster finishes a tile, atomically increments the counter, and uses the returned index as its next tile. Light clusters churn through more tiles, heavy clusters churn through fewer, and the FLOP distribution evens out across SMs.

The costs:

- Every tile boundary becomes a global atomic — serialized through L2, with a round trip to DRAM in the worst case.
- The counter must be zeroed in device memory before every launch, which complicates CUDA graphs and adds host-side bookkeeping.
- Cluster-to-cluster contention scales linearly with cluster count.

It works, but it leaves performance on the table — especially for short kernels where the per-tile atomic latency is a non-trivial fraction of the per-tile compute.

---

## 3. Cluster Launch Control: Move It to Hardware

CLC inverts the launch model. The kernel is **physically launched at the single-tile-scheduler grid size**, e.g., `(M/bM, N/bN, 1)` — far larger than the GPU can run concurrently. The hardware scheduler launches the first wave (74 clusters on B200). Then those clusters do something new: they **steal** the grid coordinates of clusters that haven't launched yet, and cancel those launches.

Stealing is an asynchronous, hardware-arbitrated atomic. Conceptually it's the same fetch-and-increment as the software path, but the counter lives in the GPU's grid scheduler, not in global memory.

### PTX Surface

Two new instructions:

- **`clusterlaunchcontrol.try_cancel`** — issued by one thread per cluster, asynchronously. It requests a cancel-and-fetch, writes an opaque 16-byte response into shared memory, and tracks completion through a transaction barrier (mbarrier). The shape is intentionally TMA-like: one thread issues, the whole cluster waits on the mbarrier.

- **`clusterlaunchcontrol.query_cancel`** — decodes the 16-byte response. It exposes:
  - `.is_canceled` — a predicate. `true` means the steal succeeded and you own a new tile. `false` means **the grid is exhausted, or a higher-priority kernel pre-empted you**, and the cluster must exit gracefully. Issuing another `try_cancel` after `false` is undefined behavior.
  - `.get_first_ctaid` — the `(x, y, z)` grid coordinates of the stolen tile.

The contract is: try, query, do work, repeat until query returns false, exit.

---

## 4. CuTe DSL Implementation

NVIDIA's CuTe DSL wires CLC into the standard warp-specialized GEMM pipeline. The interesting bits are the warp layout and the proxy fence.

### Warp specialization

Within a cluster, work is split:

- 4 epilogue warps (TMEM → SMEM → GMEM)
- 1 MMA warp (issues `tcgen05.mma`)
- 1 TMA warp (issues loads)
- **1 scheduler warp** (issues `try_cancel`)

The scheduler is its own warp because it has to run independently of the math pipeline — it should be requesting tile N+2 while the MMA warp is still grinding through tile N.

### CLC pipeline

A `PipelineClcFetchAsync` synchronizes producer (scheduler) with consumers (TMA, MMA, epilogue, and the scheduler itself, which has to read its own `.is_canceled` to decide whether to exit). The pipeline is multi-stage so multiple tiles can be in-flight at once.

### The proxy fence

The CuTe implementation places a shared-async proxy fence

```
cute.arch.fence_proxy("async.shared", space="cta")
```

**strictly after** the 16-byte CLC response has been decoded. The hazard it prevents is subtle: the scheduler warp's *next* `try_cancel` would overwrite the same shared-memory response buffer. Without the fence, that overwrite can race against consumer warps still reading the *current* tile coordinates. The fence makes the "I am done reading the response" ordering visible to the async proxy that owns the shared-memory write.

This is the kind of thing that's easy to miss in a hand-written kernel and hard to debug — it manifests as occasional wrong-tile work, not as a hang.

---

## 5. Benchmarking on B200

The CLC vs. static-persistent comparison on a B200 (148 SMs) is more nuanced than "dynamic always wins."

### Uniform GEMMs

For uniform K, CLC and static-persistent are roughly tied. Both vastly outperform single-tile scheduling because both amortize startup cost and both overlap epilogue with mainloop. This matters most at small K, where epilogue cost is a real fraction of the tile.

### Heterogeneous GEMMs (grouped, varied K)

CLC wins, by exactly the margin you'd predict from the FLOP imbalance. This is the headline use case.

### Pipeline depth has a trap

Increasing the CLC pipeline depth (e.g., from 2 stages to 3) hides scheduling latency by always having a stolen tile ready. But deeper pipelining is also **functionally closer to static scheduling**: once tiles are queued into the SM, they're committed to that SM. On heterogeneous workloads, deep CLC pipelines re-introduce the load imbalance that motivated CLC in the first place. There is a sweet spot, and it depends on the workload shape.

### Tail-drop vs. L2 hit rate

Two quirks:

1. **Static persistent suffers a tail dropoff.** Near the end of the kernel, some SMs finish their assigned tiles earlier than others and go idle. Tensor-pipe utilization sags before the kernel exits. CLC fixes this completely — every SM stays fed until the very last tile.

2. **CLC takes an L2 hit on huge problems.** On `M=N=32768, K=2048`, CLC's steady-state tensor-pipe throughput is *lower* than static's, and its L2 hit rate drops from ~52% (static) to ~35% (CLC). The likely cause is that dynamic stealing breaks the spatial locality that static partitioning naturally produces — neighboring tiles in the static schedule share rows/columns of A and B, and that reuse hits L2. CLC's stolen tiles can come from anywhere in the grid. This is currently not fully explained by published profiling.

### Hardware asymmetry

Even on a perfectly uniform GEMM, CLC's histogram of tiles-per-SM is **not flat**. Some SM pairs naturally end up with up to 5% more tiles than others before the grid drains. This is a strong hint that the GPU's physical scheduler — bank arbitration, GPC boundaries, warp-slot occupancy — does not actually want a mathematically uniform per-SM workload. Forcing one (static persistent) can be slightly suboptimal even on the "easy" case.

---

## 6. Takeaways

- CLC replaces a software global atomic with hardware-arbitrated grid stealing. The programming model is `try_cancel` / `query_cancel`, with shape similar to TMA.
- The win is largest on heterogeneous workloads (grouped GEMMs, varied K) and on the tail of any kernel where static partitioning leaves some SMs idle early.
- Pipeline depth is a tuning parameter with a real trap: too deep and CLC degenerates back toward static scheduling.
- L2 reuse can be worse under CLC on very large uniform problems, because spatial locality is no longer implicit in the schedule. This is an open area.
- The CuTe DSL implementation factors CLC into its own scheduler warp and gates the response buffer with a shared-async proxy fence — a pattern worth copying in hand-written Blackwell kernels.

CLC is part of the same Blackwell shift visible in [TMEM and 2CTA MMA]({% post_url 2026-04-12-blackwell-sm100 %}): the cluster, not the warpgroup, is the unit of work, and the hardware is doing more of the scheduling that software used to own.
