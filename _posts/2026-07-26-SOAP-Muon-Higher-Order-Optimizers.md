---
layout: post
title: "SOAP, Muon, and Beyond: Higher-Order Optimizers at Pretraining Scale"
date: 2026-07-26
categories: [LLM, Optimization]
tags: [Muon, SOAP, Shampoo, AdamW, MoE, Newton-Schulz, Megatron, Distributed Optimizer]
---

Reading notes on:
- [SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales](https://arxiv.org/pdf/2607.20548)

Deep-learning optimization has long lived in tension between the computational simplicity of first-order, element-wise methods (AdamW) and the better convergence of curvature-aware, higher-order ones. AdamW dominates frontier training because it shards trivially across distributed systems — but it is structurally blind to the row/column correlations and operator nature of weight matrices. This NVIDIA work argues the higher-order camp — **SOAP** and **Muon** — is finally production-ready at scale, once you fix the numerical instabilities and systems bottlenecks. It builds directly on [SOAP: Bridging First- and Second-Order Optimization]({% post_url 2026-06-23-SOAP-Shampoo-Adam-Eigenbasis %}).

## 1. AdamW vs. SOAP vs. Muon

**AdamW (baseline).** For a flattened gradient $g_t \in \mathbb{R}^{mn}$ of a tensor $G_t \in \mathbb{R}^{m \times n}$, AdamW keeps independent EMAs of the first ($m_t$) and second ($v_t$) moments and updates element-wise:

$$u_t = m_t \odot \text{diag}\!\left(\frac{1}{\sqrt{v_t} + \epsilon}\right).$$

Each coordinate is rescaled separately — infinitely shardable, but blind to structure.

**Shampoo / SOAP (the middle ground).** Shampoo approximates second-order information via Kronecker-factored covariances for rows and columns:

$$L_t = \beta_2 L_{t-1} + (1-\beta_2)\,G_t G_t^\top, \qquad R_t = \beta_2 R_{t-1} + (1-\beta_2)\,G_t^\top G_t,$$

then preconditions with inverse fractional powers of those factors. **SOAP** runs an Adam-style update *inside* the preconditioner's eigenbasis:

$$u_t = Q_L\,\text{Adam}(Q_L^\top m_t Q_R)\,Q_R^\top.$$

Rotate the gradient into a basis where correlations are diagonal, adapt coordinate-wise, rotate back.

**Muon (the spectral alternative).** Rather than storing massive covariances, Muon EMA-smooths the gradient ($M_t = \beta_1 M_{t-1} + (1-\beta_1)G_t$) and extracts its **polar factor** — the nearest orthogonal matrix — with a matmul-only **Newton-Schulz** iteration. This normalizes the update's singular values without the memory cost of full second moments. Muon is already in production lineages like [DeepSeek-V4]({% post_url 2026-04-26-DeepSeek-V4-Arch-Train %}) and, in per-head form, [Kimi K3]({% post_url 2026-07-18-Kimi-K3 %}).

## 2. One SVD to Unify Them

![SVD unification: SOAP whitening, Shampoo preconditioning, and Muon orthogonalization all reduce to the polar factor UVᵀ](/assets/images/optimizer_svd_unification.svg)

The paper's cleanest insight: under SVD, SOAP, Shampoo, and Muon converge. Let $G_t = U \Sigma V^\top$. For SOAP, the eigenvectors of $G_t G_t^\top$ and $G_t^\top G_t$ are exactly the singular vectors, so $Q_L = U$, $Q_R = V$. Rotating the gradient into that basis:

$$G_t^R = Q_L^\top G_t Q_R = U^\top (U \Sigma V^\top) V = \Sigma.$$

Apply an Adam-like step to the diagonalized gradient (EMA off, Adam step approximated by a sign) → $\text{sign}(\Sigma)$. Rotate back:

$$\Delta W = -\eta\, U\,\text{sign}(\Sigma)\, V^\top = -\eta\, U V^\top.$$

And $U V^\top$ is precisely the **polar factor** of the gradient. So in the simplified limit, SOAP's whitening, Shampoo's inverse-Kronecker preconditioning, and Muon's Newton-Schulz orthogonalization all reduce to extracting the polar factor — damping over-large singular directions while amplifying small ones, respecting the operator structure of linear layers.

## 3. MoEs, Large Batches, and Update-RMS Matching

**Why MoEs benefit specifically.** In an MoE, >90% of parameters live in sparse expert layers. With global batch $B_{\text{Global}}$ and top-$k$ routing over $N$ experts, each expert effectively sees

$$B_{\text{expert\_eff}} = B_{\text{Global}} \times \frac{k}{N}.$$

So as you scale global batch, sparse experts stay in a safe low-batch regime while the **dense/shared parameters take the full large-batch stress**. Muon and SOAP hold that dense path stable where AdamW frays — relevant to any large-scale MoE trainer like [Megatron-Core]({% post_url 2026-03-11-MoE-Megatron %}).

**Fair comparison via square-root scaling.** To compare across batch sizes without huge sweeps, use $\eta' = \eta \sqrt{B'/B}$, keeping the variance of update noise constant. On top of that, **update-RMS matching** aligns learning rates across optimizers: SOAP's orthogonal rotations preserve the Frobenius norm (so it matches AdamW), while Muon gets a Kimi-scaling correction factor $\sqrt{\frac{1-\beta_1}{1+\beta_1}}$.

**Result.** On models up to 72B trained on trillions of tokens, AdamW destabilizes and gives diminishing returns past a critical batch size, while Muon and SOAP keep robust token efficiency at global batches up to **100M tokens**.

## 4. Taming SOAP: Slingshot & KL-Shampoo

Naive SOAP on large models hits a catastrophic **"slingshot" instability**: oscillating gradient norms, then massive loss spikes.

**Stale preconditioner.** Standard SOAP recomputes the expensive eigenbasis infrequently (e.g., every 10 steps) and omits the current step's gradient. Early in large-batch training the landscape shifts fast, and a stale preconditioner throws the trajectory off course. *Fix:* per-step QR orthogonalization with the current gradient folded in real-time.

**KL-Shampoo for conditioning.** Accumulating $G G^\top$ is prone to float noise. A KL-divergence-based covariance estimate improves the preconditioner's condition number dramatically. Standard Shampoo tracks covariance, so eigenvalues $\Lambda \approx \Sigma^2$; KL-Shampoo's coupled update tracks singular values directly, $\Lambda \approx \Sigma$. The payoff:

$$\kappa(S_{\text{KL-Shampoo}}) = \sqrt{\kappa(S_{\text{Shampoo}})}.$$

Operating on the square root of the condition number prevents the gradient explosions that eigen-decomposition would otherwise trigger.

## 5. Systems: The Layer-Wise Distributed Optimizer

Higher-order optimizers often fail in production because modern parallelism (FSDP, ZeRO) shards 2D weight matrices into 1D fragments for memory balance — but Muon and SOAP *need* intact 2D matrices for Newton-Schulz and QR. The authors build a **Layer-Wise Distributed Optimizer** in Megatron-LM:

1. **Load balancing without slicing.** Whole intact parameter matrices are sorted by size and distributed round-robin across GPUs, preserving 2D geometry — a structure-aware sharding philosophy shared with [veScale-FSDP]({% post_url 2026-03-02-vescale-fsdp %}).
2. **Overlapped All-Gather-V.** During the forward pass, asynchronous variable-sized All-Gather-V collectives fetch the next bucket's updated, fully-intact matrices while the current bucket computes — hiding network latency behind compute with no padding waste.

## Final Verdict

**KL-SOAP** holds a slight edge in cross-entropy over Muon across the pretraining horizon — but it demands a large memory footprint for its Kronecker factors. So: if memory is abundant, **KL-SOAP** is the premier choice; if memory is the binding constraint, **Muon** gives nearly identical convergence with a far lighter state burden. Either way, the era of relying solely on element-wise optimizers like AdamW for frontier scaling is ending — and since the pretraining optimizer sets the very loss $L_{pt}$ that governs downstream RL, this connects directly to the [joint pretraining-to-RL scaling law]({% post_url 2026-07-25-Pretraining-RL-Scaling-Law %}).
