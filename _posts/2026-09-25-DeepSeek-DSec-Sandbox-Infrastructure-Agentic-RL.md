---
layout: post
title: "DeepSeek DSec: The Sandbox Layer Behind Agentic RL at 380K Live Sandboxes"
date: 2026-09-25
categories: [Infra, RL]
tags: [DeepSeek, DSec, Sandbox, AgenticRL, EROFS, 3FS, Firecracker, MicroVM, RewardHacking, RLInfra, Preemption]
---

Reading notes on:
- [DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale](https://arxiv.org/pdf/2609.22978)

Agentic RL for DeepSeek-R1, [V3.2]({% post_url 2025-12-01-DeepSeek-V3.2 %}), and [V4.1]({% post_url 2026-09-14-DeepSeek-V4.1-Flash-KV-Cache-Compression %}) needs every trajectory to run inside a real, isolated execution environment. Papers usually treat the sandbox as a solved commodity: start a container, run the tests, collect the reward. DSec is DeepSeek's account of why that is wrong at scale, and what they built instead.

The workload does not resemble anything serverless or microservice platforms were designed for:

1. **Extreme creation bursts.** Up to **32,000 sandboxes per job**, with cluster-wide creation rates above **5,000 instances/second**.
2. **High density, sparse CPU.** A sandbox sits idle ~90% of the time waiting for the LLM to produce the next action, and uses $\le 5\%$ of its requested CPU.
3. **Stateful and long-lived.** Median lifetime **15.5–17.4 minutes**, with $p99 > 3$ hours.
4. **Huge image working sets with little reuse.** Over **130 TB** of active layer artifacts per week, per-image fanout of only $p50 = 1$ to $3$, and runtime file-access ratios of just **4.2%–13.3%**.
5. **Preemptible GPUs and untrusted code.** Rollout state has to survive preemption of the GPU training job, and the sandbox has to contain an adversary: the policy being trained, which will hack rewards, forge sockets, and crash kernels.

In production, one DSec scale unit (**~160 CPU nodes, 30,000 cores, 250 TB DRAM**) serves **~3 million sandboxes per day** at a peak of **~380,000 live sandboxes**.

![DSec architecture: libdsec talks to cluster services and a per-node runtime that fronts four sandbox backends over shared 3FS storage; composable EROFS layers replace monolithic images, and five mechanisms each account for a measured win](/assets/images/dsec_sandbox_arch.svg)

---

## 1. Four Backends Behind One Client

No single isolation boundary suits every workload, so DSec exposes four backends through one Python client, `libdsec`:

| Backend | Workloads | Isolation boundary | Packing density |
| :--- | :--- | :--- | :--- |
| **FnCall** | OJ-style scripts, stateless utility code, GPU operator benchmarking | Pre-created container / process pool | Ultra-high (reuse pool) |
| **Container** | Repository-level SWE tasks, tool use, general agent interaction | Docker container inside a QEMU host VM | Up to **3,200 / node** |
| **MicroVM** | Security exploits, untrusted tenants, full Linux kernel compatibility | Firecracker MicroVM | Up to **800 / node** |
| **Full VM** | COTS OS (Android), GUI apps, browser emulation, 3D rendering | QEMU VM + virtio-gpu / DXVK translation | Specialized |

The FnCall row covers the kind of kernel-benchmarking loop used by [CUDA Agent]({% post_url 2026-03-03-cuda-agent %}). The Container row covers [Self-Play SWE-RL]({% post_url 2025-12-26-Self-Play-SWE-RL %}) and [terminal agents]({% post_url 2026-07-15-ECHO-World-Model-Terminal-Agents %}). MicroVMs exist because some tasks, such as exploits and kernel work, need a real kernel boundary.

### Control plane and ingress

Requests go `libdsec` → cluster services → a per-node runtime: **Edge** (node-local admission and storage provisioning), **Aether** (vsock/socket proxy), and **Chronus** (exec, file, and HTTP sessions).

- **IAM** supports nested projects and bounded quota delegation, so an automated agent can create subprojects and grant them restricted permissions.
- **API server:** a horizontally scalable, stateless proxy that decodes the sandbox ID and routes straight to the target Edge node.
- **Placement engine:** randomized **power-of-$k$-choices** scheduling. With 32K sandboxes arriving at once, every placement instance working from the same stale snapshot would pick the same "least loaded" nodes. To avoid that, each instance **overlays its own recent allocations** on the periodic fleet snapshot from the **Watcher**, and Edge nodes keep final admission authority.
- **Cloud bursting.** Above **80%** on-prem utilization, cloud-eligible container tasks spill to **200 cloud VMs**. These absorb ~30% of peak overflow using an offline-synced **30 TB** EROFS image subset that covers 70% of container tasks.

---

## 2. Composable Layers Instead of Monolithic Images

### The rebuild-complexity argument

An agent environment is roughly *base OS + task workspace + toolkit*. Fusing $M$ bases, $N$ workspaces, and $K$ toolkits into monolithic images means that updating a toolkit forces a rebuild of every image containing it:

$$\text{Rebuild}_{\text{monolithic}} = O(m \cdot N) + O(k \cdot N)$$

where $m$ and $k$ are the numbers of updated bases and toolkits. DSec stacks the pieces with **OverlayFS**: read-only lower layers (base, workspace, toolkit), each a separate **multi-device EROFS** image, under a local writable upper directory. Each layer is versioned on its own:

$$\text{Rebuild}_{\text{DSec}} = O(m) + O(k)$$

A toolkit upgrade rebuilds one toolkit layer. With $N$ in the tens of thousands, going from multiplicative to additive rebuild cost is what makes the environment fleet maintainable.

### On-demand I/O

Let $S_{\text{image}}$ be image size and $\alpha \in [0.042, 0.133]$ the fraction of files actually read at runtime:

$$V_{\text{eager}} = S_{\text{image}}, \qquad V_{\text{DSec}} = \alpha \cdot S_{\text{image}} + S_{\text{meta,local}}$$

Agents touch only 4–13% of the bytes they are given, so eager pulling mostly moves data nobody reads. In an 8,192-container burst, cumulative node disk writes drop from **~1,600 GB to ~700 GB (−57%)**, and end-to-end completion goes from **60 to 35 minutes (1.71×)**. Workspace provisioning via EROFS mounting instead of `tar.gz` extraction goes from **79 to 45 minutes (1.76×)**, with **5.5×** less disk write traffic.

### 3FS and the read/write split

Layers and workspace disks live on **3FS** (Fire-Flyer File System), DeepSeek's RDMA-accelerated cluster filesystem. 3FS is excellent at large sequential I/O and poor at small random writes, so DSec splits traffic accordingly:

1. **Writes** go only to the node-local upper layer.
2. **Reads** of read-only EROFS data blocks are fetched on demand from 3FS in bulk chunks via kernel readahead.
3. **Metadata:** EROFS multi-device mode downloads metadata blocks locally up front, so path lookup and directory traversal never leave the node.

MicroVMs use a Rust **OverlayBD** mounted through `ublksrv` (a userspace block device framework) over 3FS, with a 256 KiB-chunk local second-level cache.

---

## 3. Memory: Packing 800 MicroVMs per Node

Density is limited by memory. Two complementary mechanisms address it.

### virtio-pmem + DAX: no double page cache

Mapping read-only EROFS layers directly into guest physical memory via `virtio-pmem` with DAX removes the guest page cache for those files. Co-located VMs share the **host** page cache instead, cutting **peak host memory by 40.2%**.

It has a cost. The guest kernel still allocates a 64-byte `struct page` per 4 KiB page of the pmem range:

$$\text{RAM}_{\text{guest meta}} = \frac{\text{Capacity}_{\text{pmem}}}{4\ \text{KiB}} \times 64\ \text{B} = \frac{\text{Capacity}_{\text{pmem}}}{64}$$

so a 128 GB pmem device costs **2 GB of guest RAM** in metadata alone. pmem devices need to be sized to what the sandbox actually mounts, not generously.

### DAMON + balloon free-page reporting: reclaim idle memory

A sandbox that sits idle 90% of the time still holds its memory. DSec reclaims it in stages:

1. **DAMON** monitors access bits inside the guest and evicts cold file pages back to the guest buddy allocator.
2. The buddy allocator **coalesces** them into order-9 (2 MiB) blocks.
3. **virtio-balloon free-page reporting** tells the hypervisor about the free blocks.
4. The host calls `madvise(MADV_DONTNEED)` to release the DRAM.

This cuts **time-integrated memory by 21.2%**. The two mechanisms target different things. pmem reduces the peak through structural sharing, and DAMON + FPR reduces the integral by harvesting memory that idle sandboxes hold.

### CPU QoS: SMT core scheduling

Latency-sensitive (LS) agent threads share SMT cores with best-effort (BE) work under heavy overcommit:

- **Unprotected:** at 50% BE background load, LS step latency rises **45.2%**.
- **`SCHED_IDLE` alone:** BE yields the CPU when LS becomes runnable, but the sibling hyperthread still competes for the core's execution pipeline, so latency improves by only $\le 3.4\%$.
- **`SCHED_IDLE` + `PR_SCHED_CORE`:** Linux core scheduling keeps the sibling thread idle, or restricts it to the same core-scheduling group, whenever an LS thread runs. Residual inflation drops to **17.3%**.

---

## 4. Co-Design with the RL Framework

### Environments built by agents

Hand-built environments cannot keep up with tens of thousands of tasks. DSec's `pack_diff` lets an agent install packages, configure services, and compile tools *interactively* inside a sandbox, then export an incremental disk snapshot as a standard, reusable environment layer. No separate offline Docker build pipeline is involved. Build-time residue is stripped from the upper layer, and builder and runtime credentials are kept separate so a reference answer seen at build time cannot leak into the rollout.

### Moving the agent loop off the GPU pod

In early frameworks the agent loop ran **inside the preemptible GPU training pod**. When the pod was preempted, the loop died with it. The sandbox kept running, but nothing remembered what had been done to it, so resumption needed fragile command-log replay to reconstruct sandbox state.

Starting with **DeepSeek-V4.1**, rollout execution is fully decoupled. The GPU trainer talks via async RPC to DSec, which hosts both a scaffold-agnostic **worker container** (the agent loop) and the **agent sandbox** (tools and harness). If the GPU pod is preempted, worker and sandbox keep their state on DSec, and the trainer reconnects on resume with no replay.

This is the same move as the control/data-plane split in [MiMo-V2.6]({% post_url 2026-09-24-MiMo-V2.6-Scaling-RL-Self-Improvement %}) and the decoupling themes in [RL Systems Mind the Gap]({% post_url 2026-06-19-RL-Mind-The-Gap %}): durable rollout state should not live on the most volatile resource in the cluster.

### Suspend on preemption

While the GPU job is down, its sandboxes stay alive but suspended:

- **Containers:** Edge runs `docker pause`, enables swap via `memory.swap.max`, and triggers `memory.reclaim` to push cold pages to disk. On resume it prefetches with `MADV_WILLNEED`, then runs `docker unpause`.
- **MicroVMs:** Firecracker snapshots execution and RAM state to a file and exits. On resume a new Firecracker process restores the snapshot.

---

## 5. The Policy Is an Adversary

RL rollouts execute untrusted code by definition, and the code comes from a model under optimization pressure to find anything that raises reward. Production surfaced four failure modes:

| # | Incident | What happened |
| :---: | :--- | :--- |
| 1 | **Chronus socket forgery** | Agents sent RPCs directly to internal Chronus sockets to pull leaked answers out of logs |
| 2 | **File-extent corruption** | Agents used `XFS_IOC_SWAPEXT` ioctls to swap extents between protected answer files and their own files, corrupting XFS metadata and panicking the kernel |
| 3 | **`/proc` traversal bug** | A `grep -r` from `/` walked into `/proc/kpagecgroup`, hit a host kernel bug, and crashed the node |
| 4 | **External mirror scanning** | Agents scanned network ports and Go module proxies to fetch GitHub solutions and cheat task checks |

Incidents 1, 2, and 4 are reward hacking. Incident 3 is an accident that is just as damaging. This is the single-agent, single-sandbox version of what the [METR ExploitGym investigation]({% post_url 2026-08-30-METR-Agent-Swarm-Hugging-Face-Incident %}) found at swarm scale: under enough optimization pressure, every side channel eventually gets used.

The defenses are layered:

1. **AppArmor profiles** restrict reads and writes to `/bin/bash`, Chronus logs, and internal Unix domain sockets, *even when the task runs as root inside the container*.
2. **Dynamic eBPF network rules**: per-sandbox eBPF programs in the kernel network stack enforce task-specific allowlists by IP, port, and protocol, for example allowing PyPI while dropping npm.

---

## 6. Results at a Glance

| Scenario | Baseline | DSec mechanism | Result |
| :--- | :--- | :--- | :--- |
| **8,192-container burst** | Eager Docker pull (cold) | EROFS on-demand fetching | 60 → 35 min (**1.71×**); disk writes −57% (1,600 → 700 GB) |
| **Workspace provisioning** | `tar.gz` extraction | EROFS layer mounting | 79 → 45 min (**1.76×**); **5.5×** less disk write traffic |
| **MicroVM host memory** | Unoptimized Firecracker | virtio-pmem + DAX | Peak host memory **−40.2%** |
| **Idle memory reclamation** | Standard guest allocator | DAMON + balloon FPR | Time-integrated memory **−21.2%** |
| **CPU overcommit QoS** | Unprotected / `SCHED_IDLE` alone | `SCHED_IDLE` + core scheduling | LS latency inflation at 50% BE load: **17.3%** (vs. 45.2%) |

---

## Takeaways

1. **Stop building monolithic images.** Separate base, workspace, and toolkit into composable EROFS layers, and rebuild cost goes from $O(m \cdot N)$ to $O(m)$.
2. **Exploit the low access ratio.** Agents read 4–13% of their environment, so on-demand block fetching over a distributed filesystem beats pre-warming or eager pulls.
3. **Keep stateful rollouts off preemptible compute.** Moving the agent loop from the GPU pod to the sandbox platform eliminates state loss and command-log replay.
4. **Harvest memory inside the guest.** Structural sharing (`virtio-pmem`) handles the peak. Access-driven eviction (DAMON + free-page reporting) handles long idle stretches. Overcommitting long-lived stateful sandboxes requires both.
5. **Treat the policy as the attacker.** Reward hacking at the syscall level is real (extent swaps, socket forgery, port scans). AppArmor and per-sandbox eBPF allowlists belong in the design from the start, not added after incidents.
