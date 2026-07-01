---
layout: post
title: "IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse"
date: 2026-06-27
categories: [LLM, Inference, Long Context]
tags: [Sparse Attention, DSA, IndexCache, IndexShare, Long Context, Distillation, KL Divergence, GLM-5, DeepSeek]
---

Reading notes on [**"IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse"**](https://arxiv.org/pdf/2603.12201).

As long-context agentic workflows and retrieval-augmented generation scale, large language models (LLMs) are hitting a fundamental bottleneck: the quadratic complexity of the self-attention mechanism. While sparse attention models like **DeepSeek Sparse Attention (DSA)** offer a principled workaround, they introduce their own computational hurdles as sequence lengths grow.

IndexCache introduces a highly elegant and mathematically grounded approach to optimizing sparse attention. By exploiting **cross-layer redundancy**, IndexCache removes up to **75% of indexer computation**, yielding up to a **1.82× prefill speedup at 200K contexts** without sacrificing reasoning capabilities.

> **Relation to GLM-5.2.** IndexCache is the general form of the **IndexShare** trick used in [GLM-5.2]({% post_url 2026-06-21-GLM-5.2 %}). GLM-5.2 reuses one indexer per fixed block of 4 layers — i.e., *uniform interleaving*. IndexCache shows that uniform interleaving is actually the *weak* baseline, and replaces it with either a data-driven greedy layer search (training-free) or a multi-layer distillation objective (training-aware). It even validates the training-free variant directly on the 744B GLM-5 model.

Here is a deep dive into the technical details, mathematical derivations, and core insights.

---

## 1. The Bottleneck in Production-Grade Sparse Attention

In DSA, attention is decomposed into two stages: **selection** and **computation**. A lightweight "lightning indexer" scores all preceding tokens to select the top-$k$ most relevant ones (typically $k = 2048$), reducing the core attention complexity from $O(L^2)$ to $O(Lk)$.

**The Catch:** The indexer itself still operates at $O(L^2)$ per layer. While it uses cheap operations (few heads, low-rank projections, FP8 arithmetic), computing it independently across $N$ layers results in an $O(NL^2)$ total cost. Profiling a **30B DSA model** reveals that at a 200K context length, this indexer accounts for **81% of the prefill time**. Reducing this overhead is the key to accelerating long-context inference.

## 2. The Core Premise: Cross-Layer Stability

The fundamental insight behind IndexCache is that the sets of top-$k$ tokens selected by the indexer are remarkably **stable across consecutive transformer layers**. Adjacent layers share **70% to 100%** of their selected tokens, often forming distinct functional "blocks" with mutually high overlap.

IndexCache leverages this by partitioning the $N$ transformer layers into two types:

*   **Full (F) Layers:** Retain their indexers, compute fresh top-$k$ indices $T_t^{(\ell)}$ over preceding tokens, and perform sparse core attention.
*   **Shared (S) Layers:** Skip the indexer forward pass entirely. They inherit and reuse the cached index tensor from the nearest preceding F layer.

This modification requires only a single conditional branch during inference and **zero additional GPU memory**.

---

## 3. Method I: Training-Free IndexCache and the Greedy Search

If we want to apply IndexCache to an off-the-shelf DSA model without weight updates, how do we decide which layers should be F layers and which should be S layers?

**Why Uniform Interleaving Fails.** The naive approach is uniform interleaving (e.g., retaining every 4th layer: `FSSSFSSS...` — exactly GLM-5.2's IndexShare). However, the authors found that indexer importance varies significantly; early and transitional layers are highly sensitive to indexer removal. A uniform pattern often drops critical indexers, causing severe long-context degradation (e.g., a **7.2 point drop** on the Long-Context average at 1/4 retention).

**The Solution: Greedy Layer Selection.** Instead of guessing, the authors use a data-driven greedy search algorithm. Starting with all layers as F layers, the algorithm incrementally converts layers to S layers, using the **language modeling (LM) loss** on a calibration set as a proxy for downstream quality.

*   **Algorithm:** Iterate over all current F layers, tentatively flip each to S, evaluate the LM loss, and permanently commit the flip that yields the lowest loss.
*   **Efficiency:** A full search requires $N(N-1)/2$ forward passes, which can be accelerated by parallelizing the search across pipeline stages.

**Expert Insight — Why Local Similarity Fails.** The researchers initially tried a cheaper alternative: measuring the cosine similarity of attention outputs when an indexer is reused across layers. However, similarity-optimal patterns performed just as poorly as uniform interleaving. *Why?* Because cosine similarity is a "local" metric. Two layers might have nearly identical attention outputs, but miss a few critical reasoning tokens. These subtle errors cascade through downstream layers, wrecking global performance. The greedy search succeeds because **LM loss is a "global" metric** that naturally captures end-to-end error propagation, allowing it to identify truly critical layers.

---

## 4. Method II: Training-Aware IndexCache (Multi-Layer Distillation)

While training-free search works well, each indexer in the base model was only trained to serve its own layer. If we train the model from scratch (or via continued pre-training), we can explicitly optimize the F-layer indexers to serve multiple S-layers simultaneously.

### The Mathematics of Multi-Layer Distillation

In standard DSA, an indexer at layer $\ell$ is trained via KL divergence to match its own layer's aggregated full attention distribution $p_t^{(\ell)}$. The standard loss is $L_I = \sum_t D_{KL}(p_t^{(\ell)} \parallel q_t^{(\ell)})$, where $q_t^{(\ell)}$ is the indexer's output distribution.

IndexCache generalizes this. If layer $\ell$ is an F layer serving subsequent S layers $\ell+1, \dots, \ell+m$, the multi-layer distillation loss becomes:

$$L_I^{multi} = \sum_{j=0}^{m} \frac{1}{m+1} \sum_t D_{KL}\left(p_t^{(\ell+j)} \parallel q_t^{(\ell)}\right)$$

### Derivation: Gradient Equivalence

A potential concern is whether summing KL divergences introduces chaotic gradient interactions. The authors mathematically prove that this multi-layer loss is **exactly equivalent** to distilling the indexer against a single, averaged attention target.

Let the averaged target distribution across the served layers be $\bar{p}_t = \sum_{j=0}^{m} \frac{1}{m+1} p_t^{(\ell+j)}$. The single-target loss would be:

$$L_I^{avg} = \sum_t D_{KL}\left(\bar{p}_t \parallel q_t^{(\ell)}\right)$$

**Proof of Equivalence.** Because $q_t^{(\ell)}$ is the only parameter-dependent term in the KL divergence, the entropy of the target $p$ vanishes when calculating the gradient:

$$\nabla_\theta D_{KL}(p \parallel q) = -\nabla_\theta \sum_s p(s) \log q(s)$$

Applying this to the multi-layer loss:

$$\nabla_\theta L_I^{multi} = -\sum_{j=0}^{m} \frac{1}{m+1} \sum_t \nabla_\theta \sum_s p_t^{(\ell+j)}(s) \log q_t^{(\ell)}(s)$$
$$= -\sum_t \nabla_\theta \sum_s \left( \sum_{j=0}^{m} \frac{1}{m+1} p_t^{(\ell+j)}(s) \right) \log q_t^{(\ell)}(s)$$
$$= -\sum_t \nabla_\theta \sum_s \bar{p}_t(s) \log q_t^{(\ell)}(s)$$
$$= \nabla_\theta L_I^{avg}$$

**Expert Insight — The Power of Re-Training.** This mathematical equivalence proves that the indexer learns to predict a **"consensus top-k"** that covers important tokens across all served layers. Interestingly, once the model is trained with this objective, the severe layer sensitivity seen in the training-free approach completely vanishes. With multi-layer distillation, a **simple uniform interleaving pattern performs just as well** as the complex greedy-searched pattern. (In other words, re-training is precisely what makes GLM-5.2's IndexShare-style uniform reuse safe.)

---

## 5. Performance and Scaling Results

The empirical results on a 30B DSA model are highly compelling:

*   **Latency Speedup:** At a 200K context length, retaining only 1/4 of the indexers reduces prefill latency from **19.5s to 10.7s (a 1.82× speedup)**. Per-request decode throughput also jumps from **58 tok/s to 86 tok/s (a 1.48× speedup)**.
*   **Maintained Quality:** Evaluated across nine benchmarks, the 1/4 retention models (both training-free greedy search and training-aware) showed negligible degradation in long-context reasoning capabilities. Remarkably, removing redundant indexers even acted as a mild regularizer, slightly improving scores on **AIME 2025** and **GPQA-Diamond**.
*   **Scale:** Preliminary experiments applying the training-free IndexCache to the production-scale **744B GLM-5** model confirmed these findings, yielding a **1.2× end-to-end speedup** while matching the base model's performance on the Artificial Analysis index.

## Conclusion

IndexCache brilliantly demonstrates that the cross-layer representation stability of LLMs can be exploited even when full attention is entirely stripped from the network. By using either an intelligent global search or a mathematically sound multi-layer distillation loss, we can aggressively prune redundant token selection mechanisms. As sparse attention paradigms (like those in [DeepSeek-V3]({% post_url 2024-12-26-deepseek-v3 %}), [DeepSeek-V3.2]({% post_url 2025-12-01-DeepSeek-V3.2 %}), and [GLM-5]({% post_url 2026-02-20-GLM5 %})) become industry standard, structural optimizations like IndexCache will be mandatory for cost-effective, long-horizon AI agents. For where sparse attention sits in the broader inference-efficiency stack, see the [LLM efficiency notes]({% post_url 2026-06-26-Efficiency-in-LLMs-Fast-Inference-Memory-Bandwidth %}).
