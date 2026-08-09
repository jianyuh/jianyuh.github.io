---
layout: post
title: "Kimi K3 Architecture, Derived: Why SiTU-GLU, One RMS Norm, and NoPE Work"
date: 2026-08-04
categories: [LLM, Architecture]
tags: [Kimi K3, MoE, LatentMoE, SiTU-GLU, MLA, KDA, NoPE, DeltaNet, Quantile Balancing, Attention]
---

Reading notes on:
- Jianlin Su (2026), [简单谈谈K3的MoE和Attention](https://kexue.fm/archives/11848)

This is a companion to the [Kimi K3 systems overview]({% post_url 2026-07-18-Kimi-K3 %}). That note surveyed the full stack — attention, MoE, MoonEP, post-training, serving. This one zooms in on the *why*: Jianlin Su's first-principles reading of three design choices that look arbitrary until you do the math. The unifying theme of K3 is compact:

$$\text{K3} = \text{KDA} + \text{MLA} + \text{Stable LatentMoE} + \text{AttnRes},$$

trained with a Moonlight-style **Muon** whose attention weights are optimized **per head**, decoupling head-to-head interference (the per-head Muon connects to the optimizer story in [SOAP, Muon, and Beyond]({% post_url 2026-07-26-SOAP-Muon-Higher-Order-Optimizers %})).

---

## 1. SiTU-GLU: Killing the $\mathcal{O}(\|x\|^4)$ Outlier

Standard FFN/MoE experts use **SwiGLU**:

$$\text{SwiGLU}(x) = W_3\big(\text{SiLU}(W_1 x) \odot W_2 x\big), \qquad \text{SiLU}(x) = x\,\sigma(x).$$

**The blow-up mechanism.** SiLU is the nonlinearity, and it is unbounded on the positive side. When $\|x\|$ is large and a row $w_1$ of $W_1$ nearly aligns with $x$, the inner product $w_1\cdot x$ becomes huge; since $\sigma(w_1\cdot x)\to 1$, the gate output $\text{SiLU}(w_1\cdot x)$ grows *linearly* in $\|x\|$, i.e. $\mathcal{O}(\|x\|)$. The danger is coincidence: if the corresponding row of $W_2$ *also* aligns with $x$ at the **same channel**, then

- $\text{SiLU}(W_1 x)$ on that channel is $\mathcal{O}(\|x\|)$,
- $W_2 x$ on that channel is $\mathcal{O}(\|x\|)$,
- their Hadamard product is $\mathcal{O}(\|x\|^2)$,

and once projected by $W_3$ and cascaded through subsequent nonlinear layers, the effect amplifies to $\mathcal{O}(\|x\|^4)$-level extremes — numerical collapse and training instability.

**The fix — SiTU-GLU.** K3 first designs a **Sigmoid Tanh Unit** gate that softcaps the magnitude:

$$\text{SiTU}(x;\beta) = \text{softcap}(x;\beta)\cdot\sigma(x), \qquad \text{softcap}(x;\beta) = \beta\tanh\!\left(\tfrac{x}{\beta}\right),$$

bounding the gate to $(-\beta,\beta)$. To fully eliminate inflation it softcaps the *other* GLU branch too:

$$\text{SiTU-GLU}(x) = W_3\big(\text{SiTU}(W_1 x;\beta_1)\odot\text{softcap}(W_2 x;\beta_2)\big),$$

with $\beta_1 = 4$, $\beta_2 = 25$ — so the product is bounded by $\beta_1\beta_2 = 100$. Unlike a **hard clip** (GPT-OSS, DSV4), the $\tanh$ softcap is continuous and everywhere differentiable, giving cleaner gradient flow and better convergence at the same numerical ceiling.

---

## 2. Stable LatentMoE: Why a *Single* RMS Norm Is Magic

**The structure.** [LatentMoE]({% post_url 2026-01-31-LatentMoE %}) adds dimensionality reduction around the expert bank so it can double the expert count at fixed FLOPs:

- **Traditional MoE:** $d \xrightarrow{\text{route/gate}} D \to d$ (pick $k$ of $n$ experts).
- **LatentMoE:** $d \xrightarrow{\downarrow} d/2 \xrightarrow{\text{route/gate}} D \to d/2 \xrightarrow{\uparrow} d$ (pick $2k$ of $2n$).

The projection to $d/2$ frees channel budget, so the expert count doubles ($2n$-choose-$2k$) at essentially unchanged compute.

**The instability.** Naive LatentMoE makes down- and up-projection *pure* linear maps, so the forward path stacks **four matmuls in series** ($W_{\text{down}}\to W_{\text{gate/up}}\to W_{\text{down\_exp}}\to W_{\text{up}}$) with MoE routing sandwiched in the middle — a recipe for exploding/vanishing gradients.

**The fix — one RMS Norm before up-projection.** K3 ablated norms at two sites (after down-projection, before up-projection) and found the **pre-up-projection norm is the essential one**. By the minimal-change principle, K3 keeps only that single norm. Why does one weak norm do so much? Jianlin Su lists six reinforcing effects:

1. **Auto-balances shared vs. routed experts.** No hand-tuned scaling factor needed; the mixing ratio self-balances in feature space.
2. **Adds effective depth.** Even a weak nonlinearity breaks the pure-linear series, raising the local effective nonlinear depth of the block.
3. **Removes geometric distortion.** Normalizing latents onto the unit hypersphere makes Top-$K$ routing depend on *angle/direction*, not magnitude — no large-norm token hijacking the routing space.
4. **Mitigates expert skew.** Constraining the RMS scale into each expert suppresses winner-take-all over-activation and representation collapse.
5. **Promotes subspace orthogonalization.** It pushes experts to fit near-orthogonal, non-overlapping basis directions on the sphere — a near-orthogonal union of subspaces — cutting collinear redundancy across up-projection matrices.
6. **Smooths discrete-routing gradients.** The normalized gradient is approximately $\frac{\partial\mathcal L}{\partial W} = g\big(h/\|h\|_{\text{RMS}}\big)^\top$, decoupling step size from $\|h\|$ and damping the oscillation from hard Top-$K$ switching.

---

## 3. Quantile Balancing: A Histogram Trick for 896 Experts

With LatentMoE, K3 routing goes from **448-choose-8** to **896-choose-16** — same sparsity, but double the experts, which sharply worsens multi-node load imbalance. K3 keeps K2's **loss-free** balancing but replaces the SignSGD-style bias update (unstable at 896 experts) with **Quantile Balancing (QB)**: a more principled rule with **zero extra hyperparameters**.

The catch: QB needs a *global* quantile of activation scores, and quantiles are nonlinear and non-additive — an exact global sort would cost enormous cross-node communication. The fix is a **histogram estimator**:

1. Map scores into $[0,1]$;
2. Keep a local $B$-bin frequency histogram;
3. Aggregate across nodes — histograms *are* additive, so a single all-reduce of a length-$B$ vector reconstructs the global distribution and hence the quantile;
4. **1000 bins** give the same load-balancing benefit as 10000 in practice — so 1000 is the production default.

---

## 4. Attention: The Grail, and Why MLA Still Wins

Absent MTP, "under equal training and inference cost, **MLA is likely the best full-attention variant**." But **Multi-Token Prediction (MTP)** trades compute for decoding speed, and MLA's decoding behaves like $\text{head\_dims}=512{+}$ MQA — it front-loads most of the compute, so an "MLA + MTP" combo can be compute-heavy and lose out. Across the main alternatives, though, MLA still wins on balance:

| Variant | Config | KV cache | Train / prefill cost | Trade-off |
| :--- | :--- | :--- | :--- | :--- |
| **MLA** | 192+128 (MHA) | **very low** | **low** | heavier decode compute; less MTP-friendly |
| **GQA8** | 128+128 | **>3×** MLA | low | loss hard to beat MLA |
| **MFA** | 256+256 (MQA) | ≈ MLA | **very high** | prefill compute explodes (bad for agent/coding) |

A "grail" attention that truly beats MLA must simultaneously: (1) match MLA quality; (2) not exceed MLA's train/prefill cost; (3) have *smaller* KV cache; (4) have *smaller* decode compute (MTP-friendly). No single architecture clears all four today — but because **KDA** (linear attention) already carries the bulk of the compute in K3's hybrid, MLA's decode-compute weakness is largely neutralized, so K3 keeps MLA.

![K3's hybrid neutralizes MLA's weakness: KDA carries the compute, MLA handles global content NoPE-style, DeltaNet-in-Q/K supplies implicit position](/assets/images/kimi_k3_attention_grail.svg)

**DSV4 is MLA pushed to an extreme.** DeepSeek V4 nominally drops MLA for $\text{head\_dims}=512$, $K=V$ MQA with QKVO-RoPE — but that is essentially **MLA's decoding form**. Pure MQA blows up train/prefill compute, so DSV4 must bolt on **sparsity** and **KV compression** to amortize it. DSV4 doesn't overturn MLA; it pushes it to the other extreme via heroic infra (see [DeepSeek-V4]({% post_url 2026-04-26-DeepSeek-V4-Arch-Train %}) and its [TensorRT-LLM Blackwell serving]({% post_url 2026-07-20-DeepSeek-V4-TensorRT-LLM-Blackwell %})).

---

## 5. NoPE, Derived: KDA Supplies a Complete Generalized RoPE

K3 keeps standard MLA but **removes RoPE entirely — pure NoPE**. In an all-MLA model (like K2) dropping RoPE wrecks long-context and position sensitivity. K3 can drop it because of the **KDA + MLA** hybrid, and the reason is a clean equivalence.

By the completeness analysis of rotary encodings, *any power of an orthogonal matrix* can serve as a generalized RoPE. Under an orthogonal constraint, the Householder-based **PaTH** architecture rewrites exactly as a softmax attention on transformed queries/keys:

$$\text{SoftmaxAttention}(\tilde Q, \tilde K, V), \quad \tilde Q = Q - \text{DeltaNet}(Q, W, W), \quad \tilde K = K - \text{DeltaNet}(K, W, W).$$

So applying a **DeltaNet** transform to $Q$ and $K$ is *mathematically equivalent* to injecting a generalized RoPE / PaTH position encoding. Since [KDA]({% post_url 2025-12-13-KDA %}) is strictly more general and expressive than DeltaNet, the KDA + MLA hybrid **already provides implicit, complete generalized position encoding at the system level**. Explicit RoPE in the MLA layers becomes redundant — NoPE keeps the same quality while simplifying the MLA path.

**Then why still concatenate an extra 64 dims?** If RoPE is gone, why does K3's MLA still splice a 64-dim latent?

1. **Infra compatibility** — reuse the existing high-performance MLA CUDA/Triton kernels; no need to rebuild the operator base.
2. **Compute economy** — projecting a full 576-dim latent and then separate $192{+}128$ $K$/$V$ looks cleaner in code but wastes projection FLOPs for zero accuracy gain.
3. **Variable control** — K3 already introduces KDA, AttnRes, and more; keeping the mature MLA layer minimally changed maximizes training robustness.

---

## 6. The Design Philosophy: Elegant Trade-offs Under Hard Constraints

The throughline of K3 — and of Jianlin Su's reading of it — is that architecture progress is rarely a single theoretical revolution. It is **systematic trade-off under extreme engineering constraint**: the target may be an idealized perfect architecture, but each production generation must find the most stable balance among quality, training cost, inference throughput, and algorithmic stability. SiTU-GLU bounds an $\mathcal{O}(\|x\|^4)$ outlier with one softcap; a *single* RMS Norm stabilizes a four-matmul chain and buys six free benefits; NoPE falls out of a DeltaNet↔RoPE equivalence rather than a hack. Same lesson as the [systems overview]({% post_url 2026-07-18-Kimi-K3 %})'s takeaway — *bounding the unbounded* — seen this time from the derivation side.
