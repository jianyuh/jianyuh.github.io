---
layout: post
title: "Cake: Compiler–Agent Co-Design for Frontier GPU Kernels"
date: 2026-08-29
categories: [GPU, Compilers, Agents]
tags: [Cake, CUDA, Blackwell, TMEM, KernelGeneration, Agents, FlashInfer, CUTLASS, Triton]
---

Reading notes on:
- [CAKE: Compiler–Agent Co-Design for Frontier Kernel Evolution](https://arxiv.org/pdf/2608.12629)

Writing frontier GPU kernels has been a dark art practiced by a small number of people who hold both the math and the microarchitecture in their heads simultaneously. Demand for those kernels has exploded; the supply of those people has not. The obvious fix — point an LLM agent at the problem — has run into a stack that fractured into two bad options:

1. **High-level DSLs like Triton** hide warp specialization, barrier choreography, and memory-tier placement behind coarse tile abstractions. You cannot express the hand-tuned synchronization that expert kernels depend on, because the abstraction has already decided for you.
2. **CUDA/PTX or CUTLASS with CuTe** give total control, and hand you a rigorous layout calculus in exchange. One miscalculated stride produces brittle code, silent corruption, or a GPU hang that is nearly impossible to attribute.

Meanwhile, state-of-the-art AI kernel agents treat the environment as a **fixed black box**: propose CUDA, compile, test numerics, measure latency, retry on a sparse timing signal. When the code hangs, the harness returns a timeout. The agent learns nothing about the pipeline stall, the data-consistency mismatch, or the synchronization hazard that actually caused it.

**Cake** inverts the assumption. Instead of adapting agents to a compiler built for humans, it designs the compiler and its IR *to be authored by agents* — and then lets the compiler co-evolve with the kernels. This is the compiler-side counterpart to the RL-based approach in [CUDA Agent]({% post_url 2026-03-03-cuda-agent %}).

![The Cake co-evolutionary loop: agents author Cake IR, a static verifier gates candidates before compilation, and distilled failures upgrade the compiler itself](/assets/images/cake_coevolution.svg)

---

## 1. Cake IR: Declarative Resources, No Layout Algebra

Cake IR is a typed, hardware-explicit **schedule representation**. Agents write *what* the machine should do — roles, pipelines, barriers, memory staging — and the backend derives *how* it maps to physical resources: barrier addresses, phase bits, descriptor encodings, register offsets.

### Four pillars

1. **Type-checked vocabulary.** Compute (matrix math, elementwise, reductions), memory movement (global→shared, shared→register), and synchronization all come from a fixed, statically typed vocabulary — not arbitrary inline C or PTX. There is no escape hatch, which is the point: an escape hatch is where static analysis dies.
2. **Declared resources.** SMEM views, TMEM registers, barriers, and software pipelines are explicitly declared, so the compiler knows every buffer's shape, dtype, and hardware lifetime.
3. **Explicit warp roles.** Thread blocks are partitioned into named warp groups (`load`, `mma`, …), making every cross-role handoff visible to static analysis rather than implicit in control flow.
4. **Auto-derived metadata.** TMA coordinates, transaction bytes, barrier phase bits, TMEM offsets — all mechanically derived by the lowering engine instead of hand-tracked.

### A schedule fragment

A pipelined FMHA forward pass on Blackwell (`sm_100a`):

```python
@cake.schedule()
def fmha_fwd(lm, Q: LM.tma3d, O: LM.tma2d, seqlen_q: LM.i32):
    # 1. Declarative resource allocation
    pool = lm.smem(98304)                                   # 96 KB shared memory
    smem_q = pool.view(offset=0, shape=(128, 128), dtype=lm.bf16, stage=3)
    tmem_acc = lm.tmem(cols=0, width=128, shape=(128, 128), dtype=lm.f32)

    # 2. Warp-role allocation and pipelining
    load = lm.role(warps=[0])       # warp group 0: data movement
    mma  = lm.role(warps=[1])       # warp group 1: tensor-core math
    pipe = lm.pipeline(stages=3)    # 3-stage software pipeline

    # 3. Explicit synchronization barrier
    q_full = lm.barrier(count=3, prod=[load], cons=[mma], init_count=1, pipeline=pipe)

    # 4. Asynchronous execution loops
    with load:
        for stage in lm.range(0, 3):
            smem_q.tma_load(Q, coords=(0, 0, stage), stage=stage, barrier=q_full)

    with mma:
        for stage in lm.range(0, 3):
            lm.wait(q_full, stage=stage)   # block until load warp finishes this stage
            lm.fence_proxy()               # guarantee on-chip visibility
            lm.mma(tmem_acc, smem_q[stage], smem_q[stage], init=(stage == 0))
```

Everything an expert would reason about — three-stage pipeline, producer/consumer warp split, TMEM accumulator, proxy fence before the MMA reads SMEM the TMA wrote — is present and named. Nothing about swizzle bit patterns or descriptor layouts is. If you've read [NVIDIA Blackwell SM100: TMEM, TMA, and the New Tensor Core Roofline]({% post_url 2026-04-12-blackwell-sm100 %}) or the [Hopper matmul kernels]({% post_url 2025-12-29-Hopper-GEMM %}) note, this fragment reads as the *schedule* those posts describe, with the bookkeeping deleted.

### The layout bet

This is Cake's most contrarian design choice. Modern GPU compilers — CUTLASS 3/4 with CuTe, Triton with linear layouts over $\mathbb{F}_2$ — treat layout as a **first-class algebraic coordinate mapping**. Agents must then manipulate stride and swizzle equations to avoid bank conflicts and coalescing violations.

Cake says: **layout is not a first-class citizen.** Agents declare concrete local commitments — an SMEM view offset, a swizzle tag (`swizzle=128b`), TMA coordinate bindings, TMEM column ranges — and the **static analysis engine carries the entire burden of checking legality**. It traces dataflow producer→consumer, verifying representation compatibility against the target's instruction contracts. An invalid layout is rejected *before* any GPU compile or run cycle, with a pointer to the mismatched instruction or buffer.

The trade is deliberate. Layout algebra is more expressive and lets a human reason compositionally; declarative commitments plus a verifier are far more *checkable*, and checkability is what an agent actually needs. An agent doesn't want the freedom to derive a novel swizzle; it wants an immediate, localized error when the swizzle it guessed doesn't match what `tcgen05.mma` expects.

---

## 2. Two Frontier Workloads

### A. Flash-KMeans (assign)

Lloyd's iteration is dominated by exact nearest-centroid assignment:

$$a_i = \operatorname{argmin}_{j \in \{1, \dots, K\}} \|x_i - \mu_j\|_2^2$$

with $x_i \in \mathbb{R}^D$ the $i$-th token feature, $\mu_j \in \mathbb{R}^D$ the $j$-th centroid. Computed directly out of global memory it is bandwidth-bound, sweeping the centroid matrix redundantly. **Flash-KMeans** expands the square:

$$\|x_i - \mu_j\|_2^2 = \|x_i\|_2^2 + \|\mu_j\|_2^2 - 2 \langle x_i, \mu_j \rangle$$

which turns a distance computation into a GEMM plus two cheap correction terms. Mapping to Cake IR:

1. **GEMM reduction.** The $-2\langle x_i, \mu_j \rangle$ term becomes a tensor-core GEMM of $X \in \mathbb{R}^{N \times D}$ against $M \in \mathbb{R}^{K \times D}$. At the evaluation shape ($B{=}32$, $N{=}65536$, $K{=}1024$, $D{=}128$) this is solidly compute-bound: BF16 inputs, FP32 accumulate.
2. **On-chip epilogue.** $\|\mu_j\|_2^2$ (precomputed once per iteration) and $\|x_i\|_2^2$ (accumulated row-wise on the fly) are added inside TMEM or registers — the same epilogue-fusion argument as [CODA]({% post_url 2026-07-14-CODA-GEMM-Epilogue-Programming %}).
3. **Fused argmin.** The critical move: rather than writing the $N \times K$ distance matrix to HBM, the kernel does a **threadblock-level argmin reduction on-chip**, returning only the index tensor $a \in \mathbb{I}^N$ and its distances. The intermediate never exists in global memory.

### B. Kimi Delta Attention and the recurrent-state problem

Softmax attention materializes an $S \times S$ matrix at $O(S^2)$. Linear attention — KDA, Gated DeltaNet — reaches $O(S)$ by replacing softmax with an associative recurrence over a state matrix $S_t \in \mathbb{R}^{D \times D}$:

$$S_t = S_{t-1} + \beta_t (v_t - S_{t-1} k_t) \otimes q_t$$

or, in chunked linear-attention form:

$$S_t = S_{t-1} + (q_t \beta_t - S_{t-1} k_t) \otimes v_t$$

with $q_t, k_t, v_t \in \mathbb{R}^D$ and $\beta_t$ a dynamic gate acting as a learning rate on the update. The math is covered in [Linear Attention: Kimi Delta Attention]({% post_url 2025-12-13-KDA %}) and appears again as GDN in [Qwen3.8-Flash-Next]({% post_url 2026-08-27-Qwen3.8-Next-GDN-QSA-Gated-Residual %}).

**Why this is hard to schedule.** Unlike a GEMM with an elementwise epilogue, KDA carries a $D \times D$ state that **must stay live across chunks**. An expert kernel needs a two-phase pipeline:

1. **Intra-chunk** — parallel local updates within a chunk (block size 64 or 128) on tensor cores.
2. **Inter-chunk recurrence** — thread $S_t$ across chunk boundaries. Spilling to global memory at every boundary destroys the performance you came for. The agent must allocate dedicated TMEM or SMEM to keep the state warm, and coordinate handoff with cluster-scoped barriers or warp-specialized async pipelines.

The result worth noting: the agent **discovered this physical choreography without ever seeing the CUDA or SASS reference** for FlashKDA. That is the strongest evidence in the paper that the IR is expressive enough — the search found expert structure, not just expert-adjacent parameters.

---

## 3. The Co-Evolutionary Loop

Traditional compilers are static. If your code pattern isn't supported or lowers badly, *you* rewrite the code. Cake flips it: when a frontier workload exposes a gap in the IR or the analyzer, **the compiler is upgraded**.

```
┌─────────────────────────────────────────────────────────────┐
│                     1. Kernel Evolution                     │
│  [Agent] Proposes Cake IR Candidates ──► Static Verifier    │
│                                                │            │
│  GPU Benchmarking ◄── On-Device Run ◄── Lowering to CUDA    │
└───────────────────────▲────────────────────────│────────────┘
                        │                        │
                        │ Dynamic Feedback       │ Syntax / Legality
                        │ (Compute Sanitizer)    │ Failures
                        │                        ▼
┌───────────────────────┴─────────────────────────────────────┐
│                 2. Compiler-Harness Evolution               │
│  Distill failures into static checks / verifier hard gates  │
│  Add hardware primitives (e.g., Blackwell TMEM / tcgen05)   │
└─────────────────────────────────────────────────────────────┘
```

Two coupled pathways:

1. **Hardware-informed feature addition.** Agents read hardware documentation (Blackwell manuals) and propose IR vocabulary extensions. Blackwell-native `tcgen05` variants, TMEM allocations, and cluster-level synchronization primitives were added this way.
2. **Failure distillation.** When a candidate triggers a runtime crash, a silent numerical mismatch, or a Compute Sanitizer violation, the agent doesn't just patch the kernel. The **root cause becomes a compile-time static check**. An opaque GPU hang is converted into a structured verifier gate that blocks the whole class of invalid candidates forever after.

This is the load-bearing idea. A search loop's throughput is set by how fast it rejects bad candidates, and a GPU hang is the slowest possible rejection: minutes of wall clock and zero attributable signal. Every distilled failure moves rejections from the milliseconds-to-minutes regime into the microseconds regime, *and* converts them from "something went wrong" into "line 14's swizzle doesn't match the MMA contract." Both matter; the second matters more. It is the same principle as [Harness Engineering for Self-Improvement]({% post_url 2026-07-09-Harness-Engineering-Self-Improvement %}) — improving the environment beats improving the policy — applied to a compiler.

### The three pre-compile gates

Before any CUDA is generated, candidate IR passes:

- **Program safety** — synchronization hazards (producer/consumer races between warp roles), memory-safety violations, barrier phase-bit mismatches.
- **Hardware conformance** — the selected instructions are natively supported on the target SKU (no Blackwell TMEM ops lowering to `sm_80`).
- **Data consistency** — mathematical types and physical representations traced across dataflow; SMEM/TMEM swizzle configurations must match what the tensor-core instruction expects.

---

## 4. Results

Evaluated on B200 and H100 against tuned baselines.

### Clean-start Flash-KMeans

Agents authored the compute-bound `assign` kernel with low-level references withheld:

| Arm | Median vs. tuned FlashML Triton | Best run | Plateau exit | Active evolve time |
| :--- | :---: | :---: | :---: | :---: |
| Direct CUDA/PTX | $0.928\times$ | — | never plateaued | 3.73 h |
| **Cake IR** | $\mathbf{1.144\times}$ | $1.205\times$ | **3/3 runs** | **1.89 h** |

Over an 80M-token budget the raw CUDA/PTX arm never even reached the Triton baseline, let alone plateaued. Cake IR beat it, converged in every run, and did so in half the time. The gap is about brittleness, not about intelligence: the same agent, given a substrate where illegal candidates are caught statically, can afford aggressive exploration.

### Frontier workloads

| Kernel | SKU | Baseline | Gain | Deployment |
| :--- | :--- | :--- | :--- | :--- |
| **Kimi Delta Attention prefill** | B200 | Official FlashKDA | **$2.05\times$** geomean | Verified in end-to-end [Kimi-K3]({% post_url 2026-07-18-Kimi-K3 %}) serving under SGLang; FlashInfer PR #42621 |
| **KDA decode** | B200 | Upstream FlashInfer | **$1.14\times$** geomean | 30 public-API shapes; FlashInfer PR #42792 |
| **TinyGEMM** | B200 / GB300 | TRT-LLM / FlashInfer | **18–23%** latency reduction | 35 shapes; FlashInfer PR #42743 |
| **Alpha-MoE W8A8 megakernel** | B200 | TRT-LLM (prerouted API) | **$6.204\times$** at $N{=}256$<br>**$4.025\times$** at $N{=}512$ | Fuses gather, projections, activations, requantization, accumulation; FlashInfer PR #42874 |

The Alpha-MoE number needs unpacking. Isolating pure GPU execution span gives only $1.215\times$ and $1.170\times$. The $6.2\times$ API-level figure is **launch and schedule fusion**: TRT-LLM launches five separate GPU activities, while the Cake-generated kernel uses an output reset plus a single fused megakernel and skips the global-memory round trips between them. At small $N$ the kernels aren't the cost — the gaps between them are. Same lesson as [Mixture-of-Kittens]({% post_url 2026-08-05-Mixture-of-Kittens-MoE-Megakernel %}) and [event tensors and dynamic megakernels]({% post_url 2026-05-23-Event-Tensor %}). It also means the honest headline is "$1.2\times$ on compute, $6.2\times$ on the API," and the paper says so.

### From single shapes to dispatch-backed portfolios

The standard objection to auto-tuned kernels is that they overfit one shape, while production serving sees $(B, S, H)$ vary continuously. Cake adds a **generalization and dispatch stage**: specialized seed kernels from the inner loop are grouped into shape buckets with an explicit fallback hierarchy. On GB200, dispatcher-*inclusive* gains over FlashLib 0.2.0:

- **KNN build** — $1.418\times$ overall $G_{\text{span}}$ across 112 shapes (8 outer families, 80 distinct routes).
- **KNN search** — $2.116\times$ across 198 shapes.
- **Flash-KMeans portfolio** — $1.803\times$ across 124 shapes, dispatched over 12 final routes.

Reporting dispatcher-inclusive numbers is the right call; it's the metric a serving stack actually experiences.

---

## 5. Target Matrix

Cake enforces SKU-level constraints at compile time:

| Architecture | Key features | Notably absent |
| :--- | :--- | :--- |
| **Ampere** `sm_80` | `mma.sync`, `ldmatrix`, `cp.async` | TMA, clusters, TMEM |
| **Ada Lovelace** `sm_89` | FP8 tensor cores | TMA, TMEM |
| **Hopper** `sm_90a` | WGMMA, TMA, clusters, async barriers | TMEM |
| **Blackwell** `sm_100a` | `tcgen05.mma`, TMEM, 2-CTA MMA (`cta_group::2`), `tcgen05.{ld,cp,shift}` | — |
| **Blackwell** `sm_103a` | adds `tcgen05.ld.red`, $K{=}96$ block-scaled MMA | — |
| **Consumer Blackwell** `sm_120a`/`sm_121a` | Hopper-style `mma.sync` + `ldmatrix`, TMA, clusters | `tcgen05` surface, hardware TMEM |

That last row is the trap the conformance gate exists for: consumer Blackwell shares a generation name with `sm_100a` but not its tensor-core instruction surface. A kernel that silently assumes otherwise fails at runtime, on someone else's machine. Related: [Cluster Launch Control]({% post_url 2026-05-15-CLC-Blackwell %}) and [FlashAttention-4]({% post_url 2026-03-06-FA4 %}).

---

## 6. Takeaways

- **The compiler/author interface is the bottleneck, not the model.** Same agent, same budget: $0.928\times$ in raw CUDA/PTX, $1.144\times$ in Cake IR. The substrate decided the outcome.
- **Checkability beats expressiveness for agent authors.** Dropping first-class layout algebra in favor of declared commitments plus a verifier is a real loss of compositional power, traded for immediate, localized, pre-compilation errors. For a search loop that is clearly the right trade.
- **Distill failures into the compiler, not the kernel.** A patched kernel fixes one candidate; a new verifier gate fixes every future candidate in that class, and converts an unattributable hang into a pointed diagnostic.
- **Measure what the API measures.** $6.2\times$ end-to-end from $1.2\times$ of kernel speedup is a statement about launch overhead, and small-$N$ MoE is dominated by it.
- **Ship the dispatcher.** Single-shape wins don't survive contact with production; the portfolio-plus-fallback stage is what turns a benchmark result into a FlashInfer PR.

Cake's kernels are now upstream in FlashInfer and running in Kimi-K3 serving. That, more than any speedup table, is the claim: agent-authored frontier kernels are in production, and the thing that made it possible was rebuilding the compiler for a non-human author.

The logical endpoint of that argument is a chip designed on the same assumption. [OpenAI's Jalapeño]({% post_url 2026-08-30-OpenAI-Jalapeno-Inference-Chip-Teardown %}) deletes the compiler cost model entirely and replaces it with agent-driven empirical search against a cycle-accurate simulator — same diagnosis (human-authored predictive models don't survive real silicon), different feedback channel.
