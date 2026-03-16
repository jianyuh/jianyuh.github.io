---
layout: post
title: "Attention Residuals (AttnRes) – Generalizing Depth-wise Information Flow in LLMs"
date: 2026-03-16
categories: [Residual]
tags: [Residual]
---

Reading the following paper:
- [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals/blob/master/Attention_Residuals.pdf)

**1. The Motivation: Time-Depth Duality and PreNorm Dilution**
Standard residual connections are mathematically formalized as $h^l = h^{l-1} + f_{l-1}(h^{l-1})$. When unrolled over depth, this equates to $h^l = h^1 + \sum_{i=1}^{l-1} f_i(h^i)$, revealing that a layer receives a uniformly-weighted sum of all preceding layer outputs. 

The authors note a striking **"Time-Depth Duality"**: just as Recurrent Neural Networks (RNNs) compress all historical sequence tokens into a single hidden state over time, standard residuals compress all preceding layer outputs into a single hidden state over depth. 

In modern LLMs, the standard PreNorm architecture exacerbates this limitation. **Because PreNorm uses unweighted accumulation, hidden-state magnitudes grow as $O(L)$ with depth**. This phenomenon, known as "PreNorm dilution," progressively shrinks each layer’s relative contribution, forcing deeper layers to learn increasingly large outputs from fixed-scale inputs just to maintain influence. Prior efforts to solve this, such as scaled residuals or multi-stream recurrences, remained fundamentally constrained to the additive recurrence paradigm. 

**2. Full Attention Residuals (Full AttnRes): The Core Formulation**
To resolve this, the authors complete the transition from recurrence to attention over the depth dimension—mirroring the Transformer's triumph over RNNs in the sequence dimension. **Full AttnRes replaces fixed accumulation with learned softmax attention over previous layers**.

The update rule becomes:
$h^l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot v_i$

Here, the values are the token embedding ($v_0 = h^1$) and previous layer transformations ($v_i = f_i(h^i)$ for $1 \le i \le l-1$). The attention weights $\alpha_{i \to l}$ are computed via softmax:
$\alpha_{i \to l} = \frac{\phi(q^l, k_i)}{\sum_{j=0}^{l-1} \phi(q^l, k_j)}$

Crucially, **the query $q^l = w^l$ is a single layer-specific, learned parameter vector in $\mathbb{R}^d$**. This pseudo-query is entirely decoupled from the current hidden state, allowing parallel computation across layers. The kernel function utilizes an RMSNorm on the keys to prevent layers with massive output magnitudes from dominating the attention distribution: $\phi(q, k) = \exp(q^\top \text{RMSNorm}(k))$. 

*Insight:* While Full AttnRes requires $O(L^2 d)$ arithmetic, $L$ is typically $<1000$, making this trivial. However, because all $L$ layer outputs must be preserved for subsequent layers, **memory and communication overhead grow to $O(Ld)$ under activation recomputation and pipeline parallelism regimes**, creating a bottleneck at massive scale.

**3. Block Attention Residuals (Block AttnRes): Scaling to Production**
To solve the communication bottleneck, **Block AttnRes partitions the $L$ layers into $N$ blocks of $S$ layers each**. Within each block, layer outputs are reduced to a single vector representation via straightforward summation:
$b_n = \sum_{j \in B_n} f_j(h^j)$

For inter-block attention, a layer now attends only to the completed block summaries ($b_0, b_1, \dots, b_{n-1}$) and the evolving partial sum $b_n^{i-1}$ of its current block. **This elegant compression reduces both memory and cross-stage communication overhead from $O(Ld)$ to $O(Nd)$**. Scaling laws reveal that a configuration of just $N \approx 8$ blocks retains nearly all the performance benefits of Full AttnRes.

**4. Infrastructure Optimizations for Training and Inference**
The authors introduce highly specialized engineering to make Block AttnRes a true "drop-in" replacement:

*   **Training (Cross-Stage Caching):** Under pipeline parallelism with $P$ physical stages and $V$ virtual stages, naively sending accumulated blocks incurs an $O(C)$ redundant communication cost. By having each rank cache blocks received in earlier virtual stages, **only the newly incremental blocks are transmitted**. This brings the peak per-transition communication cost down to $O(P)$, enabling full overlap with computation.
*   **Inference (Two-Phase Computation & Online Softmax):** Because the query vectors $w^l$ are static parameters, memory access is amortized. **Phase 1** batches the inter-block attention computation for all $S$ layers in a block simultaneously against cached block representations, reducing HBM reads drastically. **Phase 2** sequentially computes the intra-block attention and merges it with Phase 1 outputs using online softmax. This keeps total inference memory I/O to a minimal $5.5d$ reads per layer (compared to $3d$ for standard residuals and $34d$ for multi-stream approaches like mHC).

**5. Theoretical Insights: Residuals as Structured Matrices**
By framing residual connections as a depth mixing matrix $M \in \mathbb{R}^{L \times L}$, where $M_{i \to l}$ is the weight layer $l$ assigns to layer $i$'s output, the authors unify previous literature:
*   **Standard Residuals:** An all-ones lower-triangular matrix (1-semiseparable rank).
*   **Highway Networks:** Introduce input-dependent scalar gates, but remain 1-semiseparable, acting as a "softmax-free stick-breaking attention".
*   **Multi-stream (mHC):** Functions identically to depth-wise *linear attention* with an expanded matrix-valued state, rendering an $m$-semiseparable rank matrix.
*   **Attention Residuals:** Executes depth-wise *softmax attention*, generating a dense matrix of rank $LM$. The block variant effectively interpolates the rank between standard residuals ($N=1$) and Full AttnRes ($N=L$).

**6. Empirical Findings and Shift in Optimal Architecture**
Pre-trained on 1.4T tokens using the 48B Kimi Linear architecture, AttnRes proves deeply impactful:
*   **Resolves PreNorm Dilution:** Output magnitudes remain strictly bounded in a periodic pattern across block boundaries, and gradient magnitudes are distributed substantially more uniformly across depth due to softmax competition.
*   **Learned Skip Connections:** Attention heatmaps show strong diagonal dominance (layers rely mostly on immediate predecessors), but deeper layers occasionally concentrate attention heavily on specific early layers or the embedding, forming learned, dynamic skip connections.
*   **Favors Deeper, Narrower Models:** An iso-compute/iso-parameter sweep shows that **AttnRes shifts the optimal architecture ratio ($d_{model} / L_b$) from $\approx 60$ in baselines down to $\approx 45$**. This implies that because AttnRes fixes depth-wise information degradation, the model can effectively leverage much deeper networks than standard Transformer heuristics recommend.

