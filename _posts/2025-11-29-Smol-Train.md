---
layout: post
title: "Smol Training Playbook Reading Note"
date: 2025-11-29
categories: [Training]
tags: [Training]
---

Read on the following playbook:
- [The Smol Training Playbook: The Secrets to Building World-Class LLMs](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook).

**SmolLM3**, a **3B multilingual reasoning model trained on 11T tokens**. Architecture, Scaling, and Hybrid Reasoning Alignment

## I. Pretraining Architecture and Ablation Results

The objective of targeting **on-device deployment** constrained the architecture choice to a **dense Llama-style model at 3B parameters**, ruling out high-memory architectures like Mixture-of-Experts (MoE). Ablations were run primarily on a 1B Llama3.2-style baseline trained on 45B tokens.

### Key Architectural Decisions for SmolLM3:

| Component | Technical Implementation | Justification and Ablation Insight |
| :--- | :--- | :--- |
| **Attention** | **Grouped Query Attention (GQA)**, 4 groups | GQA matched **Multi-Head Attention (MHA)** performance while substantially reducing the size of the **KV cache** at inference time, enhancing efficiency for memory-constrained edge deployment. Ablations confirmed GQA ratios (2, 4, 8) performed comparably to MHA. |
| **Positional Encoding** | **RNoPE Hybrid Approach** ("NoPE") + **RoPE ABF** | Alternates RoPE and NoPE layers. This strategy successfully maintained strong short-context performance while providing the basis for robust **length generalization**. The RoPE base frequency ($\theta$) was scaled using **RoPE ABF** from standard values to **5M** during context extension (4k to 64k tokens). |
| **Embeddings** | **Tied Embeddings** | Sharing input and output embedding matrices saves significant parameters (e.g., 17% in the 1.5B ablation model). Ablations showed that **increasing model depth** (more layers) yielded greater performance benefits than adding the equivalent parameters through untied embeddings. |
| **Training Structure** | **Intra-document masking** | Prevents tokens from attending to content from adjacent, packed documents in the sequence. While having limited impact on short-context performance, this was deemed crucial for **training speed and stability** when scaling to long sequences (4k to 64k tokens). |

### Ablation Methodology and Evaluation

For architectural evaluations, the team focused on tasks with strong early signal using the **Cloze Formulation (CF)**, which is preferred over Free-form Generation (FG) or Multiple Choice Format (MCF) for early pretraining stages. Evaluation used small subsets (1,000 questions) of benchmarks including MMLU, ARC, PIQA, and HellaSwag to accelerate iteration.

The principle of **derisking** dictates that only one variable is changed per ablation. To ensure fair comparisons when modifying components like attention mechanisms or embeddings, parameter counts were maintained by adjusting auxiliary dimensions (e.g., decreasing layers when untying embeddings increased parameter count).

## II. Training Dynamics, Hyperparameters, and Operational Stability

The full 11T token run was conducted on **384 H100 GPUs (48 nodes)**, achieving an approximate **Model FLOPs Utilization (MFU) of ~30%**.

### Hyperparameter Selection

The team adopted a robust, vanilla setup for stability:
*   **Optimizer:** **AdamW** (beta1: 0.9, beta2: 0.95, weight decay 0.1, gradient clipping 1) was chosen, despite finding alternatives like **Muon** and **AdeMaMix** achieving lower final loss in small ablations, because the alternatives proved **prone to divergence** when scaled up to the 3B model.
*   **Learning Rate Schedule:** **Warmup-Stable-Decay (WSD)** was selected for its flexibility and ease of use in mid-training decay experiments. The peak learning rate was set at **2e-4**.
*   **Data Structure:** The final configuration used a **Global Batch Size (GBS) of 2.36M tokens**, which maximized throughput on the 384-GPU cluster.

### Debugging and Mid-Training Crises

The training run was disrupted by several non-architectural failures, highlighting the challenges of production-scale runs:

1.  **Mystery of the Vanishing Throughput:** Throughput plummeted, traced to the **FSx Weka** network storage evicting dataset shards (from the 24TB dataset) mid-training, forcing costly S3 fetches. *Fix:* Data was moved to the **local NVMe RAID (`/scratch`)** drives, which provided **26.59 GiB/s** throughput, and a spare node was reserved for immediate manual failover, reducing downtime from 1.5 hours to zero.
2.  **The Persistent Throughput Drops (Software):** Even on local storage, throughput continued to drop. This was attributed to the **nanotron built-in dataloader (`nanosets`)** naively building a large index that grew with step count, causing shared memory issues. *Fix:* Switching to the established **TokenizedBytes** dataloader resolved the drops.
3.  **The Subtle Tensor Parallelism (TP) Bug:** After 1T tokens, the 3B model was underperforming expectations. The root cause was a subtle bug in the TP implementation: **identical random seeds were used across all TP ranks**, when each rank should have been initialized with a unique seed (e.g., `self.config.general.seed + tp_rank`). Fixing this required a full run restart.

## III. Post-Training and Hybrid Reasoning Alignment

The base model was aligned using Supervised Fine-Tuning (SFT), continued pretraining (**mid-training**), and **Preference Optimization (PO)** to become a **hybrid reasoning model**.

### SFT and Data Alignment

The SFT phase focused on enabling the dual reasoning mode: `/think` (extended CoT) and `/no_think` (concise).
*   **Loss Masking:** Loss was computed exclusively over **assistant tokens** (not user queries), yielding small but consistent performance gains, particularly on **IFEval**.
*   **Multi-Turn Reasoning:** Initial SFT models failed **"abysmally"** at switching reasoning modes across multi-turn dialogues. This deficit was remedied by creating the synthetic dataset **IFThink** to explicitly train this context-switching capability.
*   **Vibe-Testing Insight:** Informal "vibe-testing" uncovered a critical bug where the custom system prompt (`custom_instructions`) was accidentally being removed from training samples, a failure missed by standard metrics.

### Preference Optimization (PO)

Preference optimization provided large gains over the SFT base:
*   **Algorithm:** **APO-zero** (Anchored Preference Optimization) delivered **15–20 percentage points** improvement on instruction following tasks like IFEval.
*   **Hyperparameters:** Optimal PO performance was achieved with a learning rate typically **10x smaller** than the SFT rate (e.g., 1e-6).

### Reinforcement Learning with Verifiable Rewards (RLVR)

When applying RLVR (specifically **GRPO**) to improve math performance in the concise `/no_think` mode:
*   The model exhibited **reward hacking**, maximizing the reward by generating excessively long Chain-of-Thought (CoT) responses, converting the `/no_think` mode into a verbose `/think` mode.
*   *Mitigation:* This was solved by integrating an **overlong completion penalty** into the reward function, parameterized by `max completion length` and `soft punishment cache`. Penalties set in the **2.5–3k token range** successfully constrained the output length distribution while preserving performance gains.

## IV. Infra and Performance Tuning

The infra chapter provides detailed quantitative analysis of the cluster used for SmolLM3.

### GPU Internal Architecture and Memory

*   **TFLOPS Reality:** While the H100 boasts a theoretical peak of 990 TFLOPs at BF16, achievable utilization at the kernel level for dense matmuls was around **72–77%**. End-to-end training MFU typically falls to **~30%** due to communication overhead and non-matmul operations.
*   **Memory Bandwidth:** **HBM3** (High Bandwidth Memory) provides **~3 TB/s** total bidirectional bandwidth (1,519 GB/s read + 1,519 GB/s write), validating the H100's theoretical specification. **Flash Attention** achieves speedups by using **SRAM tiling** to reduce HBM access from O(N²) to O(N), maximizing the utilization of faster memory tiers.

### Interconnect Topology and Bandwidth

The significant difference between intra-node and inter-node bandwidth dictated the parallelism strategy (DP x TP):

| Communication Path | Technology | Bandwidth (Bidirectional) | Relative Speed |
| :--- | :--- | :--- | :--- |
| **GPU-to-GPU (Intra-node)** | **NVLink 4.0** | **786 GB/s** | High-speed, local |
| **GPU-to-Storage (Local)** | **NVMe RAID** | **26.59 GiB/s** (337K IOPS) | Fastest I/O path |
| **GPU-to-GPU (Inter-node)** | **EFA Network** | **~45–58 GB/s** (All-to-all, 16 nodes) | Major bottleneck |
| **CPU-to-GPU** | **PCIe Gen5 x16** | ~63.02 GB/s | Intermediate speed |

The high bandwidth difference justifies constraining **Tensor Parallelism (TP)** to within a single node to leverage NVLink, leaving **Data Parallelism (DP)** to handle the slower inter-node EFA communication.

