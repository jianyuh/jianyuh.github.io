---
layout: post
title: "UltraEP: Near-Optimal MoE Load Balancing on Rack-Scale Nodes"
date: 2026-08-12
categories: [Systems, LLM]
tags: [MoE, Expert Parallelism, Load Balancing, EPLB, Rack-Scale, NVLink, All-to-All, DeepEP, Megatron-LM]
---

Reading notes on:
- [UltraEP: Unleash MoE Training and Inference on Rack-Scale Nodes with Near-Optimal Load Balancing](https://arxiv.org/pdf/2606.04101)

Mixture-of-Experts architectures are moving decisively toward **fine-grained** designs — hundreds of computationally light experts instead of a handful of fat ones. The specialization is better, but the routing distribution becomes hyper-dynamic, and at that point traditional load balancing stops being a proactive optimization and becomes a *reactive failure*. Placing experts from historical statistics is fundamentally mismatched to non-stationary traffic whose popularity shifts sharply across microbatches, layers, and data domains.

UltraEP's argument is that the rack-scale node has changed the economics enough to do the thing everyone previously assumed was too expensive: solve the balancing problem **exactly, on the critical path, every microbatch**. This continues themes from [MoE Parallel Folding]({% post_url 2026-01-18-EP %}) and [Mixture-of-Kittens]({% post_url 2026-08-05-Mixture-of-Kittens-MoE-Megakernel %}), where the recurring lesson is that once the EP group fits inside one high-bandwidth domain, work that used to belong to the host or to an offline planner can move onto the device and into the hot path.

![UltraEP hot path: skewed per-microbatch load, quota-driven solver, balanced execution](/assets/images/ultraep_hotpath.svg)

---

## 1. The EP Straggler Problem

At 32-way or 64-way expert parallelism, minor routing fluctuations amplify into system-level stalls. The failure shows up in four coupled ways:

- **Device-level load skew.** Inter-rank imbalance ratios routinely reach **1.30–4.01x** without intervention.
- **Compute stragglers.** The overloaded GPUs hold the global synchronization barrier hostage.
- **Token all-to-all bottlenecks.** Congestion on the interconnect during dispatch/combine.
- **Memory spikes.** Receive-side hotspots create activation memory spikes up to **11x** higher than ideal.

### Why periodic and predictive balancing fails

Historical methods like **EPLB** adjust placement from stale statistics. In fine-grained MoE, popularity drifts fast enough that a stale plan doesn't merely leave residual imbalance — it frequently *introduces new stragglers*, because it optimizes for a traffic pattern that has already vanished. The only way to reach near-ideal throughput is exact-load balancing computed from the load matrix of the microbatch you are about to run.

### The RSN connectivity enabler

What makes this affordable is the **Rack-Scale Node (RSN)**. Standard RDMA clusters do packet-based networking at tens of GB/s; RSNs extend *scale-up* connectivity across dozens of GPUs at hundreds of GB/s with load/store memory semantics — the same shift analyzed in [NCCL GIN & MSCCL++]({% post_url 2026-06-24-NCCL-GIN-and-MSCCLpp-GPU-Communication %}) and [Inside TPU and GPU Clusters]({% post_url 2026-07-16-Collective-Communication-TPU-GPU-Clusters %}). Concretely, it buys a latency budget of about **0.3 ms** for hot-path balancing (planning plus replication). With the entire EP group inside one high-bandwidth domain, real-time per-microbatch balancing becomes physically viable for the first time.

---

## 2. Architecture and Memory Co-Design

Balancing gains are easy to erase with system overhead, so UltraEP co-designs the memory layout with the execution pipeline.

### Logical vs. physical experts

UltraEP separates **logical experts** (model identity) from **physical experts** (on-device instances). Each rank reserves fixed **main slots** for persistent logical experts and transient **redundant slots** for replicas, giving a deterministic one-to-many mapping.

### The "replication-only" mandate

UltraEP deliberately refuses to reorder experts. At large EP, where a rank hosts only **2–4 main experts**, full state migration is prohibitive: reordering means updating global routing tables *and* migrating persistent optimizer states. Replication needs only a weight copy to expand capacity, leaving the home instance and its optimizer state untouched. This is the same asymmetry that makes [Megatron Core's MoE training path]({% post_url 2026-03-11-MoE-Megatron %}) conservative about placement changes mid-run.

### Cross-layer buffer reuse

Redundant slots store **no optimizer state**; weight and gradient buffers are shared across layers. For a model like **Qwen3-235B**, this shrinks a redundant slot from gigabytes to **36 MB of weights / 72 MB of gradients** — a **30x to 90x** reduction. Redundancy becomes cheap enough to use aggressively.

### Execution pipeline and the ring buffer

The forward pass does eager planning immediately after gating, then weight distribution. The backward pass re-materializes weights and reduces gradients from replicas back to main experts. UltraEP uses `torch.autograd` to carry a **virtual layer ID** that indexes a ring buffer whose size is matched to the maximum number of in-flight microbatches. That single trick makes the scheme compatible with pipeline parallelism and virtual PP without the balancer needing to know anything about the PP schedule.

---

## 3. The Optimization Formulation

| Symbol | Meaning |
| :--- | :--- |
| $R$ | All ranks in one EP group |
| $E$ | All logical experts |
| $h(e)$ | Home rank of logical expert $e$ |
| $H(e)$ | Set of ranks hosting physical instances of $e$ |
| $N_{\text{slot}}$ | Redundant slots per rank |
| $\Lambda = \{\lambda_{r,e}\}$ | Global load matrix (tokens from rank $r$ to expert $e$) |
| $U = \{u_{e,r}\}$ | Solved load quota table |
| $\beta$ | Target balancing coefficient (default $1.01$) |

The objective is to minimize the **exposed path** latency of the busiest ranks:

$$T_{\text{fwd}} = \text{solve_rep} + \max(T_{\text{reroute}}, T_{\text{w_distr}}) + T_{\text{tok_a2a}} + T_{\text{moe}}$$

$$T_{\text{bwd}} = T_{\text{tok_a2a}} + T_{\text{moe}}$$

Compute latency is set by the most loaded rank:

$$T_{\text{moe}} \propto \max_{r \in R} \sum_{e \in E} u_{e,r}, \qquad T_{\text{bwd,moe}} \approx 2\,T_{\text{fwd,moe}}$$

with the factor of two accounting for Wgrad and Dgrad. Token all-to-all is bounded by the worse of a rank's send and receive volume:

$$T_{\text{tok_a2a}} \propto \max_{r \in R} \max\left(\sum_{e \in E} \lambda_{r,e},\ \sum_{e \in E} u_{e,r}\right)$$

And weight distribution captures the fan-out bottleneck at whichever rank happens to host the hottest main experts:

$$T_{\text{w_distr}} \propto \max_{r \in R} \sum_{e \in E_r} \left(|H(e)| - 1\right)$$

That last term is the one people forget: making a hot expert *very* replicated is itself a communication cost concentrated on one sender, which is exactly what Section 5 goes after.

---

## 4. The Quota-Driven Planning Solver

The control plane is simplified by decoupling token-level routing from combinatorial search. Instead of assigning tokens, the solver assigns **quotas**.

### The greedy feasibility oracle

Binary-search the smallest load threshold $\tau$ that can be satisfied. For each rank define excess and slack:

$$\text{exc}_r = \max(\ell_r - \tau,\ 0), \qquad \text{slk}_r = \max(\tau - \ell_r,\ 0)$$

Crucially, a replica is materialized **only when it carries useful load**, governed by a quota floor $u_{\min}$. This is what prevents the pathology of creating replicas that satisfy a placement heuristic while resolving no bottleneck.

### Quota decomposition and locality

Once $U$ is solved, source ranks consume their **own local expert quotas first** before dispatching to remote replicas — cutting cross-rank traffic without violating the solved threshold $\tau$.

### Algorithm 1: joint solving

```
INPUT: Load Matrix Λ, Slot Budget N_slot, Quota Floor u_min, Target β
INITIALIZE: Binary search range [τ_lo, τ_hi]
WHILE τ_lo < τ_hi:
    τ = (τ_lo + τ_hi) / 2
    Identify exc_r (excess) and slk_r (slack) for all ranks.
    FOR each overloaded rank r (descending exc_r):
        FOR each hot expert e on rank r (descending λ_e):
            t* = rank with max slk that does not host e
            δ  = min(exc_r, slk_t*, remaining_load_of_e)
            IF δ >= u_min:
                update exc_r, slk_t*, transfer quota to t*
    IF all exc_r == 0: τ is FEASIBLE; search lower.
    ELSE:              search higher.
OUTPUT: Slot assignment X and Load Quota Table U
```

### GPU-native implementation

To hit the sub-**0.3 ms** budget, the solver runs as a **single-SM kernel**: the load matrix is staged in shared memory and reductions are done at warp level, so routing metadata never round-trips to the CPU. This is the same "get the host off the critical path" discipline that dominates modern MoE kernels — see the on-device scheduling in [Mixture-of-Kittens]({% post_url 2026-08-05-Mixture-of-Kittens-MoE-Megakernel %}) and [SonicMoE]({% post_url 2025-12-19-SonicMoE %}).

---

## 5. RSN-Native Balancing Communication

The solver emits a stream of device-resident transfer tasks; communication kernels consume it and saturate RSN bandwidth.

### Persistent tile streaming

A persistent device-side kernel executes the weight/gradient transfers. Experts are split into fixed-size **tiles**; thread blocks pull tiles from the task stream and use **one-sided peer-memory access** to store directly into remote GPUs. Double buffering folds task lookup and synchronization into the data movement itself.

### Chunk streaming relay for hotspot fan-out

When an expert's replica count exceeds a threshold (set to **4**), a naive one-to-many broadcast makes the source rank the bottleneck — precisely the weight-distribution term above. UltraEP builds a two-stage relay tree:

- **Relay frontier.** The source seeds a relay set of size $\approx \sqrt{|H(e)| - 1}$.
- **Streaming logic.** Relays forward *chunks* of tiles immediately on arrival, with no global barrier.
- **Scheduling.** A global greedy heuristic assigns relay roles to ranks with spare sending capacity, so no single rank bottlenecks replication.

The $\sqrt{\cdot}$ frontier is the standard two-level broadcast trade-off: it balances the source's fan-out against relay depth, turning an $O(|H(e)|)$ serial send into roughly $O(\sqrt{|H(e)|})$ exposed time.

---

## 6. Evaluation

Measured at production scale (up to **256 GPUs**):

- **Throughput.** **94.3%** of ideal on average; **1.42x** over Megatron-LM, and **1.49x** over no balancing overall.
- **Balance.** Inter-rank imbalance flattened from **1.30–4.01** down to **1.01–1.04**.
- **Replication speed.** **3.1x to 5.5x** faster than PyTorch Distributed and DeepEP.
- **Production.** A **RefMoE-288B** run sustained **92%** of ideal throughput with stable convergence.

### Memory is the underrated result

Removing receive-side hotspots yields up to an **11x reduction in MoE activation memory** for serving prefill. That is arguably more strategically valuable than the throughput number: it is headroom against OOM, and it buys larger batch sizes. In training it lowers activation peaks, letting models scale without paying the recompute tax of frequent activation checkpointing.

---

## Takeaways

1. **Fine-grained MoE turned balancing into a real-time problem.** Statistics-driven placement is not merely suboptimal at this granularity — it actively manufactures stragglers.
2. **Rack-scale connectivity is what makes exactness affordable.** A ~0.3 ms hot-path budget only exists because the EP group sits in one load/store domain.
3. **Replicate, don't reorder.** Sidestepping optimizer-state migration is the design decision that keeps hot-path balancing tractable, and cross-layer buffer reuse (GB → tens of MB) is what makes redundant slots cheap enough to spend freely.
4. **Quotas beat token-level search.** Binary search over a threshold plus a greedy oracle with a quota floor $u_{\min}$ gives exact balance without combinatorial placement search, and fits in a single SM.
5. **The memory win may outlast the throughput win.** 11x lower activation peaks changes what batch sizes and context lengths are reachable at all.
