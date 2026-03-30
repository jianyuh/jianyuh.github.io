---
layout: post
title: "Nvidia Inference: Disaggregated Decode, LPU Integration, and Datacenter Macro-Architectures"
date: 2026-03-30
categories: [AI]
tags: [NVIDIA, GTC, Inference, LPU, Groq, Rubin, NVLink, SemiAnalysis]
---

Reading notes based on:
- [Nvidia – The Inference Kingdom Expands (SemiAnalysis)](https://newsletter.semianalysis.com/p/nvidia-the-inference-kingdom-expands)

At GTC 2026, Nvidia aggressively expanded its hardware and inference ecosystem to address the emerging bottlenecks of memory-bound LLM decode phases, massive CPU demands in reinforcement learning, and KV cache storage limits. The company announced three entirely new system architectures: **Groq LPX** (integrating the LP30 chip), **Vera ETL256**, and **STX**, alongside major updates to the **Kyber rack architecture (NVL144, NVL576, and NVL1152)**.

---

## 1. The Groq "Acquisition" and LP30 Architecture

Nvidia functionally acquired Groq via a \$20B IP licensing and acqui-hire deal, bypassing drawn-out antitrust regulations. Groq's hardware, specifically the LPU (Language Processing Unit), is utilized to create a disaggregated decode system.

### LP30 (Groq 3 LPU) Hardware Details

*   **Silicon & Manufacturing:** Designed on **Samsung's SF4X node**, skipping the failed LPU 2 which suffered from 112G SerDes malfunctions. Using Samsung SF4X allows Nvidia to bypass TSMC N3 logic constraints and HBM allocation constraints, enabling incremental revenue scale-up.
*   **Memory Hierarchy:** Features a single-level memory hierarchy with **500MB of on-chip SRAM** (up from 230MB in Gen 1) providing an ultra-fast **150 TB/s memory bandwidth**. It lacks HBM entirely.
*   **Compute:** Dedicated to tensor-first compute with **1.2 PFLOPs of FP8**, which is a fraction of standard GPU compute but highly optimized for deterministic execution. The chip pumps instructions vertically and streams data horizontally across functional slices (VXM, MEM, SXM, MXM).
*   **Form Factor:** Deployed in the **LPX Compute Tray**, featuring a belly-to-belly PCB design (8 LPUs on top, 8 on bottom) to minimize X/Y trace distances, alongside **2 Altera "Fabric Expansion Logic" FPGAs**, 1 Intel Granite Rapids CPU, and a BlueField-4 module.

---

## 2. Decoding Acceleration Techniques

The integration of the LPU is designed to accelerate the **latency-sensitive, memory-bounded decode phase** of LLM inference, leaving the **compute-intensive prefill phase to GPUs**.

### Attention-FFN Disaggregation (AFD)

**The Problem:** During decode, GPU utilization for the Attention mechanism barely improves as batch sizes scale because it is bounded by loading KV cache. Conversely, Feed Forward Network (FFN) utilization scales effectively with larger batch sizes. In state-of-the-art sparse Mixture-of-Expert (MoE) models, utilization drops further as tokens route to a larger pool of experts.

**The Solution:** Attention operations are **stateful** (relying on dynamic KV cache) and are thus mapped to HBM-heavy Rubin GPUs. FFN operations are **stateless** (depending only on token inputs) and are mapped to the SRAM-heavy, deterministic LPUs.

**Network Optimization:** To hide the communication latency of dispatching tokens from GPU to LPU experts and combining them back, the system relies on **ping-pong pipeline parallelism**, allowing tokens to continuously bounce between GPUs and LPUs over Spectrum-X Ethernet.

The key insight here is a further decomposition beyond [prefill-decode disaggregation]({% post_url 2025-03-30-prefill-decoding-disagg %}). Within the decode phase itself, attention and FFN have fundamentally different computational profiles:

| Property | Attention (Decode) | FFN (Decode) |
|----------|-------------------|--------------|
| State | Stateful (KV cache) | Stateless |
| Bottleneck | Memory bandwidth | Compute |
| Batch scaling | Poor (KV-bound) | Good |
| Best hardware | HBM-heavy GPU | SRAM-heavy LPU |

### Speculative Decoding & Memory Management

*   LPUs can host draft models or Multi-Token Prediction (MTP) layers to predict $k$ new tokens, which the main model verifies in a single "warm prefill" step.
*   Unlike stateless FFNs, draft models/MTP layers require tens of gigabytes of dynamic KV cache. To support this, **the Altera FPGAs provide up to 256GB of additional DDR5 memory per FPGA** for the LPUs to access.

This is a natural evolution of the [speculative decoding]({% post_url 2024-12-15-speculative-decoding %}) paradigm: instead of running the draft model on the same GPU as the verifier, offloading it to a dedicated LPU eliminates the resource contention entirely.

---

## 3. Networking Topologies and Bandwidth Math

Nvidia's systems push the physical limits of copper to keep TCO down, orchestrating incredibly dense electrical networks before resorting to optical interconnects.

### LPX Rack Network Math

*   **Intra-Tray:** 16 LPUs connect via an all-to-all PCB mesh. Each LPU routes to 15 others via $4 \times 100G$ C2C links.
*   **Intra-Rack (Inter-Node):** Each LPU routes $2 \times 100G$ to 15 other nodes. With FPGAs connecting at 25G/50G, a node features 1,020 differential pairs. Across 16 nodes, the **copper backplane supports 8,160 differential pairs** ($16 \times 1020 / 2$).
*   **Total intra-rack scale-up bandwidth:**

$$
\text{BW} = 256 \text{ LPUs} \times 90 \text{ lanes} \times 112\text{ Gbps} / 8 \times 2 \text{ directions} = 645 \text{ TB/s}
$$

### Rubin Ultra NVL144 (Kyber Rack) Scale-up Math

The Kyber rack fits 144 Rubin Ultra GPUs and 72 NVLink 7 switches.

*   **GPU Bandwidth:** Each GPU uses 72 Differential Pairs (DPs). At $200\text{ Gbit/s bi-di}$ per channel, each GPU achieves **14.4 Tbit/s uni-directional** scale-up bandwidth.
*   **Switch Bandwidth:** Each NVSwitch 7 uses 144 lanes of 200G, totaling **28.8 Tbit/s uni-directional** bandwidth. Connecting these requires midplanes and copper flyover cables.
*   **NVL288 Constraints:** Scaling to 288 GPUs across two racks via copper would require 20,736 additional DPs, acting as a massive upper bound on cable content, unless higher radix switches are introduced.

---

## 4. Co-Packaged Optics (CPO) vs. Copper Roadmap

A key architectural insight from GTC 2026: **Nvidia uses copper where it can, and optics where it must**.

| System | Generation | Scale-up Interconnect | CPO? |
|--------|-----------|----------------------|------|
| NVL72 | Rubin | Intra-rack copper | No |
| NVL144 | Rubin Ultra | Intra-rack copper (Kyber) | No |
| NVL576 | Rubin Ultra | Intra-rack copper + inter-rack CPO | Partial |
| NVL1152 | Feynman | Full rack-to-rack CPO | Yes |

*   **Rubin / Rubin Ultra:** Scale-up within NVL72 and NVL144 Kyber racks remains strictly copper.
*   **NVL576 (Rubin Ultra):** This 8-rack system will be the first introduction of **CPO scale-up**, utilizing a two-tier all-to-all network between racks, though intra-rack networking stays copper.
*   **Feynman NVL1152:** Will fully adopt CPO for rack-to-rack scale-up, overcoming the physical reach/shoreline limits of bumping electrical SerDes from 224G to 448G.

The transition is driven by physics: as SerDes rates double, the reach of copper decreases and the shoreline (pin count per die edge) becomes the binding constraint. CPO sidesteps both by converting electrical signals to photons at the package boundary.

---

## 5. Ancillary Infrastructure: Vera ETL256 and STX

To prevent non-GPU components from bottlenecking system performance, Nvidia released auxiliary racks:

### Vera ETL256

A standalone, liquid-cooled rack packing **256 Vera CPUs** to handle the surging preprocessing and simulation demands of Reinforcement Learning workloads. It utilizes a single-tier Spectrum-6 multiplane topology.

This reflects a growing reality: RL training pipelines (reward model evaluation, environment simulation, data preprocessing) impose enormous CPU demands that steal GPU cycles if co-located. Dedicated CPU racks eliminate this contention.

### CMX and STX (Context Memory Storage)

To combat the exponential growth of KV Cache, Nvidia introduced **Tier G3.5 NVMe storage**. The STX reference rack utilizes **BlueField-4 DPUs** (featuring a Vera CPU, 2x CX-9 NICs, and 2x SOCAMM modules) to offload "warm" KV cache from expensive GPU HBM and system DRAM, optimizing inference efficiency.

This creates a multi-tier memory hierarchy for KV cache:

| Tier | Medium | Capacity | Bandwidth | Use Case |
|------|--------|----------|-----------|----------|
| Hot | GPU HBM | ~192 GB/GPU | ~8 TB/s | Active decoding |
| Warm | System DRAM / FPGA DDR5 | ~512 GB–1 TB | ~200-400 GB/s | Recent context, draft models |
| Cold | NVMe (STX) | Multi-TB | ~50-100 GB/s | Long context, session persistence |

---

## 6. Strategic Synthesis

Nvidia is fundamentally transitioning from selling standalone AI accelerators to orchestrating entire **datacenter macro-architectures**. Several strategic threads emerge:

1. **Supply chain arbitrage:** By disaggregating decode tasks to SRAM-heavy LPUs manufactured on Samsung's SF4X (not TSMC N3), Nvidia bypasses critical industry supply constraints (TSMC N3 capacity and HBM allocation) while preserving high-margin GPU allocations strictly for compute-heavy prefill.

2. **Copper maximalism:** The aggressive densification of copper flyover cables and midplanes in Kyber and LPX architectures proves that TCO optimization remains paramount. Nvidia delays the transition to expensive CPO interconnects until physical electrical bounds absolutely mandate it in the Feynman generation.

3. **Full-stack lock-in:** With dedicated GPU racks (Kyber), CPU racks (Vera ETL256), storage racks (STX), and decode accelerator racks (LPX), plus the networking fabric (Spectrum-X, NVLink 7) tying them together, Nvidia is selling complete datacenter blueprints rather than individual chips.

4. **Inference economics:** The LPU integration directly addresses the economic pain of decode. During decode, GPUs are massively underutilized on compute but bottlenecked on memory bandwidth. Offloading FFN to cheap, deterministic LPUs improves the cost-per-token by utilizing the right silicon for the right workload.
