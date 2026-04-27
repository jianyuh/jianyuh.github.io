---
layout: post
title: "DeepSeek-V4 Architecture & Training: Hybrid Attention, Muon, On-Policy Distillation"
date: 2026-04-26
categories: [LLM]
tags: [DeepSeek, MoE, Long Context, Muon, On-Policy Distillation, RL Infra]
---

Paper: [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf).

Companion to [DeepSeek-V4 Infra]({% post_url 2026-04-25-DeepSeek-V4-Infra %}). Where that note covered Section 3 (General Infrastructures), this one walks through Sections 2, 4, and 5: the model architecture (residuals, hybrid attention, MoE, Muon), pre-training stability, and the post-training pipeline. V4 keeps the [DeepSeek-V3]({% post_url 2024-12-26-deepseek-v3 %}) backbone but replaces nearly every load-bearing component above it.

---

## 1. Manifold-Constrained Hyper-Connections (mHC)

V4 replaces conventional residual connections with **mHC**, which expands the residual stream by $n_{hc}$ but constrains the residual mapping matrix $B_l$ to the **Birkhoff polytope** (doubly stochastic matrices):

$$B_l \in \mathcal{M} := \{M \in \mathbb{R}^{n \times n} \mid M \mathbf{1}_n = \mathbf{1}_n,\ \mathbf{1}_n^\top M = \mathbf{1}_n^\top,\ M \ge 0\}$$

This bounds $\|B_l\|_2 \le 1$, so the residual transform is non-expansive, and because $\mathcal{M}$ is closed under multiplication, stability is preserved across deep stacks. Projection uses 20 iterations of Sinkhorn–Knopp; the input/output mappings $A_l, C_l$ are bounded by Sigmoid to prevent signal cancellation. See the standalone [mHC note]({% post_url 2026-01-06-mHC %}) for the full derivation.

---

## 2. Hybrid Attention: CSA + HCA + SWA

To break the quadratic bottleneck for million-token contexts, V4 interleaves **Compressed Sparse Attention (CSA)** and **Heavily Compressed Attention (HCA)**.

### Compressed Sparse Attention

CSA shrinks sequence length by a factor of $m$, then applies DeepSeek Sparse Attention (DSA) on a top-$k$ subset of the compressed entries.

*   **Compression with overlap:** every $m$ tokens are consolidated into a single KV entry via Softmax; compression indices overlap so each compressed entry is built from $2m$ tokens, preserving continuity.
*   **Lightning indexer:** queries are produced low-rank, and the index uses a ReLU-aggregated score to select compressed blocks:

$$I_{t,s} = \sum_{h=1}^{n_h^I} w^I_{t,h} \cdot \text{ReLU}\!\left(q^I_{t,h} \cdot K^{IComp}_s\right)$$

### Heavily Compressed Attention

HCA pushes compression further: $m'$ tokens (with $m' \gg m$) collapse into a single KV entry **without** overlap, then dense Multi-Query Attention runs over the highly compressed set.

### Shared KV MQA + grouped projection

Both CSA and HCA share an MQA format where each compressed entry is *both* key and value. To bound the cost of projecting concatenated head outputs back to dimension $d$, heads are split into $g$ groups, each projected to a smaller intermediate $d_g$, then mapped to the final output.

### Three correctness fixes

1.  **SWA branch for local context.** CSA/HCA queries only see preceding compressed blocks, so they lose immediate locality. A supplementary Sliding Window Attention branch retains $n_{win}$ uncompressed recent KV entries.
2.  **Partial RoPE on last 64 dims.** Because KV entries act as both keys *and* values, raw attention outputs inadvertently carry absolute positions. V4 applies a **reverse RoPE** (position $-i$) directly on the attention outputs to recover relative positions.
3.  **RMSNorm + attention sinks.** RMSNorm is applied to queries and KV entries *before* core attention to prevent logit explosion, and a learnable attention sink is added to the Softmax denominator to absorb attention when no strong matches exist.

---

## 3. MoE Updates

*   **Affinity score:** routing affinity in the MoE layers switches from Sigmoid to $\text{Sqrt}(\text{Softplus}(\cdot))$.
*   **Hash routing in early layers:** the first several blocks replace dense FFNs with MoE layers using **deterministic hash routing** (by token ID) instead of learned routing.
*   **Load balancing:** auxiliary-loss-free balancing is retained, supplemented by a small sequence-wise balance loss to prevent extreme intra-sequence skew.

---

## 4. Muon Optimizer

V4 adopts **Muon** for the vast majority of weight matrices (embeddings, prediction heads, and norms stay on AdamW).

### Hybrid Newton–Schulz orthogonalization

Muon needs to orthogonalize the gradient update $M$. V4 runs a 10-step hybrid Newton–Schulz iteration:

$$M_k = a M_{k-1} + b\,(M_{k-1} M_{k-1}^\top) M_{k-1} + c\,(M_{k-1} M_{k-1}^\top)^2 M_{k-1}$$

*   First **8 steps**: aggressive coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$ for rapid convergence.
*   Final **2 steps**: stabilizing coefficients $(a, b, c) = (2, -1.5, 0.5)$ that lock singular values at 1.

### No QK-Clip needed

Other Muon-trained models often add QK-Clip to prevent attention logits from exploding. V4 omits this entirely — the explicit RMSNorm on hybrid-attention queries and KV entries already prevents logit explosion, making QK-Clip redundant.

---

## 5. Pre-Training: Stability Above All

### Configuration

| Model | Layers | $d$ | Total params | Activated / token | Tokens |
|---|---|---|---|---|---|
| V4-Flash | 43 | 4096 | 284B | 13B | 32T |
| V4-Pro | 61 | 7168 | 1.6T | 49B | 33T |

Context curriculum: **4K → 16K → 64K → 1M**. Sparse attention (CSA/HCA) is introduced at the 64K threshold.

### The outlier–routing vicious cycle

Trillion-parameter MoE training reliably produces loss spikes. V4's diagnosis: **loss spikes track outliers in MoE layers, and the routing mechanism mathematically amplifies them into a vicious cycle.** Two targeted fixes:

*   **Anticipatory routing.** Decouple synchronous updates of the backbone and routing networks: feature compute at step $t$ uses current $\theta_t$, but routing indices are computed using historical $\theta_{t-\Delta t}$. To avoid doubling parameter-loading overhead, the system pre-fetches data at $t-\Delta t$ and caches the routing indices for future use, bounding wall-clock overhead to ~20%. The mechanism is triggered dynamically only when loss spikes are detected.
*   **SwiGLU clamping.** The linear branch of SwiGLU is hard-clamped to $[-10, 10]$ and the gate's upper bound is capped at 10 — surgical suppression of anomalies without hurting quality.

---

## 6. Post-Training Phase I: Specialists + Generative Reward Models

Post-training begins by cultivating domain specialists (math, code, agents) via SFT followed by GRPO-style RL.

*   **Generative Reward Model (GRM).** For hard-to-verify tasks where RLHF traditionally needs a separate scalar reward model, V4 forces the **actor itself to act as the GRM**. By jointly optimizing generative and evaluative proficiency, the model's reasoning capability is fused into its judging process — robust trajectory evaluation with minimal human annotation.
*   **Interleaved thinking across rounds.** Older models flushed reasoning traces on each new user prompt. V4 preserves the full `<think>...</think>` history across rounds, enabling a cumulative chain of thought over long-horizon agentic tasks.
*   **Quick instruction tokens.** For auxiliary chatbot subroutines (intent classification, web-search routing), V4 appends specialized tokens like `<|action|>` or `<|query|>` directly to the input. The model emits parallel classifications by reusing the already-computed KV cache, replacing external classifier models and cutting TTFT.

---

## 7. Post-Training Phase II: Multi-Teacher On-Policy Distillation

Instead of weight merging or mixed RL, V4 consolidates $N > 10$ specialists into a single base model via **On-Policy Distillation (OPD)**. (See the standalone [On-Policy Distillation note]({% post_url 2026-04-19-On-Policy-Distillation %}) for broader context.)

### Objective

The student learns from teacher distributions on its *own* generated trajectories under reverse KL:

$$L_{OPD}(\theta) = \sum_{i=1}^N w_i \cdot D_{KL}\!\left(\pi_\theta\,\|\,\pi_{E_i}\right)$$

### Full-vocabulary logit distillation

Prior distillation often collapsed the KL into a token-level advantage estimate, which suffers from high gradient variance. V4 computes the **exact full-vocabulary KL** against teacher logits — faithful knowledge transfer at the cost of memory.

### Memory hack for trillion-scale teachers

Materializing $\lvert V \rvert > 100k$ logits across multiple trillion-parameter teachers is infeasible. V4's trick: **cache only the last-layer teacher hidden states** in a centralized buffer during the forward pass, then reconstruct full logits on the fly during loss computation using a custom TileLang kernel.

---

## 8. Million-Token RL Infrastructure

Two system innovations in the long-context RL stack:

### Write-Ahead Log to prevent length bias

In a preemptible rollout cluster, regenerating an interrupted request from scratch sounds harmless but introduces **severe length bias**: shorter responses are mathematically more likely to survive an interruption uninterrupted, so the resulting output distribution skews short.

V4 fixes this with a **token-granular WAL** persisting each generated token. On interruption, the WAL is replayed to reconstruct the KV cache exactly where it left off — preserving the output distribution.

### DSec agent sandbox

For agentic trajectory evaluation, V4 introduces **DeepSeek Elastic Compute (DSec)**, supporting hundreds of thousands of concurrent sandboxes. It unifies four execution substrates — Function Call, Container, microVM, fullVM — under a single API, with layered EROFS / overlaybd storage enabling millisecond-scale environment resumption and deterministic trajectory replay.

---

## 9. Takeaways

1.  **Stability constraints have moved into the architecture.** mHC's Birkhoff projection, RMSNorm-before-attention, and SwiGLU clamping all bake stability into model structure rather than into ad-hoc training tricks like QK-Clip.
2.  **Long context is a hierarchy of compressions, not a single trick.** CSA + HCA + SWA, plus reverse-RoPE on shared-KV outputs, is a *system* — each piece exists to fix a specific failure mode of the others.
3.  **Outliers, not loss curves, are the right signal for MoE instability.** Anticipatory routing only triggers when spikes are detected, costing ~20% wall-clock — a far better trade than running it always or chasing spikes after the fact.
4.  **OPD with full-vocabulary KL replaces mixed RL.** Cache last-layer hidden states, reconstruct logits with TileLang, distill against the exact teacher distribution. Cleaner than weight merging and lower variance than token-level RL distillation.
5.  **Length bias is a correctness bug.** Naively regenerating preempted rollouts looks like a robustness fix but distorts the data distribution. WAL replay turns preemption from a statistical hazard into pure overhead.
