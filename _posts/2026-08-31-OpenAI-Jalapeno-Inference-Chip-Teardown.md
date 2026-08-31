---
layout: post
title: "Beyond the GPU: A First-Principles Teardown of OpenAI Jalapeño"
date: 2026-08-31
categories: [Hardware, Inference]
tags: [Jalapeno, OpenAI, ASIC, Inference, NUMA, SpatialProgramming, Roofline, Compilers, MXFP4]
---

Reading notes on:
- [Redesigning the Inference Chip: From Nvidia GPU's Flaws to OpenAI Jalapeño](https://zartbot.github.io/blog/arch/jalapeno/en.html)
- [Jalapeño's first results show industry-leading speed and efficiency in AI inference](https://openai.com/index/jalapeno-first-results/)

For a decade, the GPU has won by **hiding** latency: throw thousands of threads at the machine so that while one warp waits on memory, another computes. That bargain is excellent for training and increasingly terrible for real-time agentic inference, where the workload is batch-1-ish, the dependency chain is serial, and there simply aren't enough independent warps to hide behind.

Jalapeño is OpenAI's clean-sheet answer, and its thesis is blunt: **stop hiding latency and start eliminating it.** Out-of-order cores instead of high occupancy. NUMA instead of UMA. A two-level memory hierarchy instead of five. Static spatial mapping instead of dynamic hardware scheduling. And — the part most relevant to everything else I've been writing about — **empirical search by LLM agents instead of a compiler cost model.**

![Jalapeño: 64 NUMA core slices, dual NoC, and an agent-driven empirical search compiler stack](/assets/images/jalapeno_arch.svg)

---

## 1. Why SIMT Fails at Batch 1

The NVIDIA SIMT model maximizes throughput by saturating compute pipelines with concurrent warps. At batch 1 it hits a **latency-hiding failure**: there aren't enough warps to mask memory access, so the machine stalls in a state that reaches *neither* the compute ceiling *nor* the memory ceiling. Execution units sit idle waiting for operands that arrive dozens of cycles late. On a roofline plot you are not on either roof — you are underneath both, which is the one position the roofline model doesn't have a name for.

### The bound nobody is hitting

The theoretical output token rate for a memory-bound decode is:

$$\text{Throughput} \leq \frac{N \times M}{K}$$

with $N$ the scale-up domain size (number of chips), $M$ the HBM bandwidth per chip, and $K$ the model parameter scale.

Run the numbers for a 1T-parameter model and the theory says **1,000–2,000 tokens/s per user**. Blackwell and B300 in practice deliver **100–200 tokens/s**. That is a **10× gap between the bandwidth bound and reality** — and the important consequence is that *buying more HBM bandwidth does not close it.* You are not bandwidth-limited; you are limited by everything between the bandwidth and the ALU. The roofline framing of the same problem is in [The Economics of a Token]({% post_url 2026-05-17-token-economics %}) and [Efficiency in LLMs]({% post_url 2026-06-26-Efficiency-in-LLMs-Fast-Inference-Memory-Bandwidth %}).

### Where the cycles go: UMA contention

Two structural costs in the Hopper/Blackwell/Rubin lineage:

- **Cross-partition L2 latency.** The L2 is partitioned to sustain aggregate bandwidth, but maintaining a *unified* view across partitions costs **200–400 cycles** on a cross-partition access.
- **Global fence polling.** SMs are independent and unsynchronized, so a global memory fence requires the hardware to poll every core. The queuing delay dwarfs the actual data transfer.

Both are the price of general-purpose flexibility, and both are charged per synchronization — which is exactly the operation a serial decode loop performs constantly. For the counterpoint on what that hierarchy buys you, see [NVIDIA Blackwell SM100: TMEM, TMA, and the New Tensor Core Roofline]({% post_url 2026-04-12-blackwell-sm100 %}).

---

## 2. Four Phases, and the Fallacy of PD Disaggregation

Agentic inference isn't two workloads, it's four, with genuinely different arithmetic intensities:

| Phase | Arithmetic intensity | Primary bottleneck | Roofline position |
| :--- | :--- | :--- | :--- |
| **Prefill** | High: $O(L \cdot d^2)$ projection | Matrix compute | Compute ceiling |
| **Decode** | Low: GEMV | Memory bandwidth | Memory (weights / KV) |
| **Draft model** | Variable (step-dependent) | Bandwidth & sync | Memory / latency bound |
| **Verify** | Medium (tree-dependent) | Network / weight traffic | Hybrid (MoE-dependent) |

The industry response has been [prefill–decode disaggregation]({% post_url 2025-03-30-prefill-decoding-disagg %}) — and GLM-5.3-Flash extends it to three stages with [EPD]({% post_url 2026-08-28-GLM-5.3-Flash-Hybrid-Attention-Vision-In-The-Loop %}). Jalapeño's argument is that this direction is a dead end, for two reasons.

### Resource-ratio lock-in

Define the serviceable rate $R$ of a cluster with capacities $P, D, S, V$ for the Prefill, Decode, Speculate, and Verify pools, and per-request weights $w_\bullet$:

$$R = \min\left(\frac{P}{w_p},\ \frac{D}{w_d},\ \frac{S}{w_s},\ \frac{V}{w_v}\right)$$

A `min` over four terms is only efficient if the four ratios stay matched. They don't: the weights depend on the speculative **acceptance rate $\alpha$**, which fluctuates per request with prompt complexity (see [speculative decoding]({% post_url 2024-12-15-speculative-decoding %}) and [DFlash & DSpark]({% post_url 2026-06-29-DFlash-DSpark-Diffusion-Speculative-Decoding %})). Fixed pools of dedicated hardware therefore guarantee **dark silicon** — idle capacity in three pools while the fourth throttles the cluster. Every disaggregation boundary you add is another `min` term, and another opportunity for the ratio to drift.

### The transfer tax

Moving KV cache across discrete physical chips over a scale-out network degrades TTFT and burns substantial I/O energy. Disaggregation trades a utilization problem for a data-movement problem, and at rack scale the data movement is the expensive one.

Jalapeño's alternative is to make one chip **homogeneous and balanced** enough to run all four phases well, keeping data local. That is a bet that the utilization win from fungible hardware beats the specialization win from dedicated pools — and it only pays if a single chip can be good at both GEMM-heavy prefill and GEMV-heavy decode.

---

## 3. The NUMA Microarchitecture

Jalapeño abandons unified memory. Compute cores are **paired directly with HBM slices** — Non-Unified Memory Access — maximizing locality and shortening the physical data path. You lose the convenience of a global address space with uniform cost; you gain a memory latency you can actually reason about statically.

### The core slice

**64 core slices**, each built on an **out-of-order superscalar core**. This is the deepest departure from the GPU. Rather than hiding latency with occupancy (thousands of threads), each slice extracts **instruction-level parallelism from a single instruction stream**, filling pipeline gaps with independent operations from the same thread. OoO logic is expensive in area and power — it's what GPUs deliberately deleted to buy more ALUs. Jalapeño buys it back, because at batch 1 there is no second warp to spend those ALUs on anyway.

### Peak compute, derived

**13.4 PFLOPS (MXFP4)**, and the derivation shows it's a yield decision as much as a performance target. On a max-reticle die, each of the 64 slices holds **16 physical tensor units, 15 active and 1 held as redundancy**:

$$1.7\ \text{GHz} \times 64\ \text{slices} \times (15 \times 64 \times 64 \times 2)\ \text{ops/cycle} = 13.4\ \text{PFLOPS}$$

The derived BF16 peak is **835 TFLOP/s** — a 16:1 ratio to MXFP4, which tells you the tensor units are natively narrow and BF16 is reserved for attention precision rather than being a first-class datapath. That's a legitimate call for inference given the [NVFP4]({% post_url 2025-11-20-NVFP4-Train %}) and [MXFP8]({% post_url 2025-12-07-MXFP8-Train %}) results, but it is a much sharper precision cliff than a GPU presents.

### A deliberately thin hierarchy

NVIDIA's stack is Register File → SMEM → TMEM → L2 → HBM. Jalapeño collapses it to **512 KB L1 per slice**, connected by a **1536-bit bus** to sliced HBM4 at **240 GB/s of local bandwidth** per slice (≈15.4 TB/s aggregate).

No global L2 means no cross-partition penalty, no cache contention between slices, and no jitter — at the cost of requiring software to **explicitly orchestrate every byte of data movement**. That is a hard requirement to place on a human programmer, and it is the reason section 5 exists.

---

## 4. System and Network

The system design is notably conservative about supply chain and aggressive about determinism.

- **Vindaloo & Katsu.** Compute and control are physically separated: **Vindaloo** compute trays hold 8 Jalapeño ASICs; **Katsu** CPU trays hold 2× AMD Turin x86 with 1.5 TB of memory. Connectivity is PCIe DAC — either 8× PCIe-Gen5×8 or 4× PCIe-Gen5×16.
- **Dual NoC.** The on-chip network is split by purpose. A general-purpose mesh handles global HBM access for odd jobs; a dedicated **8×8 two-stage collective NoC** handles All-Reduce and All-Gather, wired **directly into the slice L1 caches**. Putting collectives on their own fabric that terminates in L1 is the on-chip analogue of the argument in [NCCL GIN & MSCCL++]({% post_url 2026-06-24-NCCL-GIN-and-MSCCLpp-GPU-Communication %}) and [Inside TPU and GPU Clusters]({% post_url 2026-07-16-Collective-Communication-TPU-GPU-Clusters %}): collectives have a latency profile that general traffic will always contend with if you let it.
- **Rack topology.** Tensor parallelism runs in-rack over **600 GB/s copper**; expert parallelism runs cross-rack over **200 GB/s optical**. Switching uses 1U **"Chana"** trays on Broadcom **Tomahawk 6** silicon — mature Ethernet scale-up rather than a proprietary high-risk interconnect. Matching the TP/EP split to the copper/optical split is exactly the placement problem attacked in [MoE Parallel Folding]({% post_url 2026-01-18-EP %}), [UltraEP]({% post_url 2026-08-12-UltraEP-Exact-Load-Balancing-Rack-Scale-MoE %}), and [RoutePack]({% post_url 2026-08-21-routepack %}).

Choosing Tomahawk 6 Ethernet over a bespoke fabric is the most telling decision in the whole design. It says the team believes their advantage is in the chip and the compiler, not in owning the interconnect.

---

## 5. Killing the Cost Model

Everything above is only usable if something can solve the spatial mapping problem. A thin hierarchy plus NUMA plus static placement means the *software* decides where every tensor lives and when every byte moves — a search space no human wants to enumerate.

Traditional compilers handle this with a **cost model**, and cost models fail on complex silicon because they cannot predict real contention. OpenAI's answer is to **delete the cost model and replace it with empirical search driven by LLM agents.**

### The six-layer stack

| Layer | Role | GPU counterpart |
| :--- | :--- | :--- |
| **Teacup** | Request / serving layer | [vLLM]({% post_url 2025-11-30-vLLM %}) |
| **Gigakernel** | Persistent resident kernel holding the entire decode loop | CUDA Graph |
| **Gluon** | Triton-family tile SPMD kernel language | Triton |
| **Linear Layouts + TensorInfo** | Algebra for layout and physical placement | CuTe / linear layouts |
| **Assembly-like kernels** | ~3,000-line low-level kernels with hardware sanitizers | PTX / SASS |
| **Chilisim** | Cycle-accurate simulator, <5% error | Nsight + hardware |

Two entries deserve emphasis. **Gigakernel** is a persistent kernel containing the *complete* decode loop — the megakernel idea from [Mixture-of-Kittens]({% post_url 2026-08-05-Mixture-of-Kittens-MoE-Megakernel %}) and [event tensors]({% post_url 2026-05-23-Event-Tensor %}), promoted from optimization to architectural assumption. If the decode loop never leaves the chip, launch overhead and host round-trips stop existing.

**Chilisim** is the one that makes the whole approach work. A cycle-accurate simulator with **<5% error** is the fitness function. Agents can evaluate millions of spatial mappings without touching silicon, which is the difference between a search loop bounded by hardware availability and one bounded by CPU time.

### The MLA trajectory

The approach was validated on DeepSeek MLA optimization ([DeepSeek-V3]({% post_url 2024-12-26-deepseek-v3 %}), [V3.2]({% post_url 2025-12-01-DeepSeek-V3.2 %})), reaching **88.9% of theoretical peak over a 40-hour agent trajectory** through compiler passes P2 (spatial mapping / slice placement) and P4 (tiling / prefetch search):

| Milestone | % of peak |
| :--- | :---: |
| Functional correctness | 0.31% |
| FP8 matrices | 31.7% |
| Blocked look-ahead rescaling | 59.2% |
| V-matmul scheduling | 77.1% |
| K-tile prefetch + coalesced access | 88.9% |

The shape of that curve is the interesting part. The jump from 0.31% to 31.7% is datatype selection — the kind of thing a cost model handles fine. The climb from 59.2% to 88.9% is scheduling and prefetch choreography, which is precisely where cost models break down and empirical search wins. The last 30 points of peak are bought by measurement, not by prediction.

This is the same conclusion [Cake]({% post_url 2026-08-29-Cake-Compiler-Agent-Co-Design %}) reached from the opposite direction — Cake keeps a compiler and makes its *verifier* the agent's feedback channel; Jalapeño keeps a simulator and makes *measured cycles* the feedback channel. Both delete the human-authored predictive model. Related: [CUDA Agent]({% post_url 2026-03-03-cuda-agent %}), [Harness Engineering for Self-Improvement]({% post_url 2026-07-09-Harness-Engineering-Self-Improvement %}), and [Jeff Dean at YC 2026]({% post_url 2026-08-06-Jeff-Dean-Self-Improving-AI %}).

---

## 6. Takeaways

- **The 10× gap is a synchronization gap, not a bandwidth gap.** 1,000–2,000 tok/s theoretical vs 100–200 tok/s actual on a 1T model is the single number that justifies this whole architecture. More HBM does not fix it.
- **OoO cores are the anti-GPU bet.** Occupancy-based latency hiding requires parallel work that batch-1 decode does not have. Spending area on ILP extraction instead of ALU count is only correct if you believe interactive inference is the dominant workload — which is the bet.
- **Every disaggregation boundary adds a `min` term.** With four phases and a fluctuating acceptance rate $\alpha$, fixed pools guarantee dark silicon. Homogeneous-and-balanced is a coherent alternative, not a compromise.
- **A thin hierarchy is a compiler problem disguised as a hardware simplification.** Deleting the global L2 removes jitter and hands the entire orchestration burden to software. That trade is only available now because agents plus a <5% simulator can carry it.
- **Chilisim is the real product.** A fast, accurate fitness function is what converts "agents optimize kernels" from a demo into a compilation strategy. Cake reached the same place with a static verifier: the bottleneck in agentic optimization is always the speed and specificity of the feedback signal.
- **Mature Ethernet, bespoke silicon.** Tomahawk 6 over a proprietary fabric is a statement about where OpenAI thinks its moat is.

Jalapeño's most interesting claim isn't the 13.4 PFLOPS. It's that a chip can now be designed *assuming* its programming model is too hard for humans — because the search will be done by agents against a simulator. That inverts the last forty years of the hardware–software contract, in which the ISA existed to be humane.
