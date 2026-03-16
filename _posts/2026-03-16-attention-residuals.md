---
layout: post
title: "Attention Residuals (AttnRes) – Generalizing Depth-wise Information Flow in LLMs"
date: 2026-03-16
categories: [Residual]
tags: [Residual]
---

Paper: [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf)

This paper asks a simple question: if attention replaced recurrence along the sequence dimension, why are we still using a fixed additive recurrence along the depth dimension?

## Motivation: Residuals Behave Like Depth-Wise Recurrence

Standard residual connections can be written as:

$$h^l = h^{l-1} + f_{l-1}(h^{l-1})$$

Unrolling over depth gives:

$$h^l = h^1 + \sum_{i=1}^{l-1} f_i(h^i)$$

That view makes the paper's core observation clear: every layer sees a uniformly weighted sum of all earlier layer updates. The authors call this a **time-depth duality**. In an RNN, all past tokens are compressed into a single hidden state over time. In a Transformer with standard residuals, all past layer outputs are compressed into a single hidden state over depth.

The paper argues that this becomes especially problematic in PreNorm LLMs. Because the residual stream keeps accumulating unweighted updates, hidden-state magnitudes tend to grow with depth, leading to **PreNorm dilution**. Later layers then have to produce larger and larger updates just to maintain the same influence on the final representation.

## Full Attention Residuals

The proposed fix is to replace fixed accumulation with learned softmax attention over previous layers:

$$h^l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot v_i$$

Here, the values are the embedding and prior layer outputs, while the attention weights are:

$$\alpha_{i \to l} = \frac{\phi(q^l, k_i)}{\sum_{j=0}^{l-1} \phi(q^l, k_j)}$$

Instead of deriving the query from the current hidden state, each layer uses a learned parameter vector:

$$q^l = w^l \in \mathbb{R}^d$$

This detail matters. The query is static, so depth-wise attention can still be computed in parallel across layers. The keys are RMS-normalized before scoring,

$$\phi(q, k) = \exp(q^\top \mathrm{RMSNorm}(k))$$

which prevents large-magnitude activations from dominating the attention distribution.

Conceptually, Full AttnRes turns the residual path into a learned depth mixer. A layer is no longer forced to treat all earlier layers equally; it can emphasize whichever layers are most useful.

## Why the Full Version Does Not Scale Cleanly

From an arithmetic perspective, Full AttnRes is manageable. The paper quotes $O(L^2 d)$ work, which is acceptable for realistic layer counts. The real issue is systems cost: all previous layer outputs must remain available, so activation storage and cross-stage communication grow as $O(Ld)$. Under activation recomputation or pipeline parallelism, that becomes the bottleneck.

## Block Attention Residuals

To make the idea practical, the paper introduces **Block AttnRes**. The $L$ layers are partitioned into $N$ blocks of size $S$, and each block is summarized by a simple sum:

$$b_n = \sum_{j \in B_n} f_j(h^j)$$

A layer then attends to:

- completed block summaries $b_0, b_1, \dots, b_{n-1}$
- the running partial sum of its current block

This is the key compression step. Instead of exposing every earlier layer explicitly, the model exposes a smaller set of block summaries, which reduces memory and communication overhead from $O(Ld)$ to $O(Nd)$. The reported scaling results suggest that a small number of blocks, roughly $N \approx 8$, captures most of the gains of Full AttnRes.

## Training and Inference Optimizations

The paper also does the systems work needed to make Block AttnRes usable at scale.

For training, the main trick is **cross-stage caching** under pipeline parallelism. Rather than repeatedly sending already-seen block summaries across virtual stages, each rank caches what it has received and only transmits the incremental blocks. That avoids redundant communication and reduces the peak communication cost enough to overlap it with computation.

For inference, the authors use a **two-phase computation**:

1. compute inter-block attention for all layers in a block together
2. compute intra-block attention sequentially and merge it with an online softmax

Because the layer queries are static parameters, this organization lowers memory traffic substantially. The paper reports total residual-path memory I/O of about $5.5d$ reads per layer, compared with $3d$ for standard residuals and much higher cost for multi-stream alternatives such as mHC.

## Residual Connections as a Mixing Matrix

One of the nicest parts of the paper is the matrix view. If we define a depth mixing matrix $M \in \mathbb{R}^{L \times L}$, where $M_{i \to l}$ is the weight assigned by layer $l$ to layer $i$'s output, several architectures fall into the same template:

- **Standard residuals:** an all-ones lower-triangular matrix
- **Highway networks:** input-dependent scalar gates, but still low-complexity depth mixing
- **Multi-stream methods such as mHC:** depth-wise linear attention with a higher-rank structured state
- **Attention residuals:** full depth-wise softmax attention

Under this lens, Block AttnRes smoothly interpolates between standard residuals and Full AttnRes by changing how many block summaries are exposed.

## Takeaways

The empirical story is straightforward:

- **PreNorm dilution is reduced.** Activation magnitudes stay bounded more cleanly, and gradients are distributed more evenly over depth.
- **The model learns nontrivial skip patterns.** Attention maps remain strongly local, but deeper layers sometimes jump back to very early layers or even the embedding.
- **The preferred architecture shifts.** In the paper's iso-compute and iso-parameter sweeps, AttnRes favors deeper, narrower models than standard residual designs do.

That last point is especially interesting. If depth is no longer handicapped by uniform residual accumulation, then making a model deeper becomes a better tradeoff than it is in a baseline Transformer.

My main takeaway is that this paper treats residual connections as an architectural choice rather than a fixed law. Standard residuals hard-code a very specific depth mixing rule: every earlier layer contributes equally through simple addition. AttnRes replaces that rule with learned attention, then introduces a blockwise approximation that keeps the idea deployable at scale.

Whether this becomes a standard recipe will depend on implementation complexity and robustness across model families, but the framing is strong: residual paths are not just optimization scaffolding, they are a depth-wise information routing mechanism that can be redesigned.
