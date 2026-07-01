---
layout: post
title: "DFlash & DSpark: Block Diffusion and Semi-Autoregressive Drafting for Flash Speculative Decoding"
date: 2026-06-29
categories: [LLM, Inference, Speculative Decoding]
tags: [Speculative Decoding, Block Diffusion, DFlash, DSpark, EAGLE-3, Semi-Autoregressive, Confidence Scheduling, LLM Serving]
---

Reading notes on two closely related works:
- **DFlash** — [*DFlash: Block Diffusion for Flash Speculative Decoding*](https://arxiv.org/abs/2602.06036) (z-lab) · [code](https://github.com/z-lab/dflash)
- **DSpark** — [*DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*](https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf) (DeepSeek-AI & Peking University)

[Speculative decoding]({% post_url 2024-12-15-speculative-decoding %}) has become a cornerstone of Large Language Model (LLM) serving, accelerating inference by decoupling token *proposal* from token *verification*. But production deployments expose a fundamental tension: the trade-off between the **quality** of draft tokens and the **system-level cost** of verifying them. (For the underlying acceptance/maximal-coupling math, see the [LLM efficiency notes]({% post_url 2026-06-26-Efficiency-in-LLMs-Fast-Inference-Memory-Bandwidth %}).)

These two papers attack the problem from complementary angles. **DFlash** ([z-lab](https://github.com/z-lab/dflash)) reimagines the *drafter* itself, replacing serial autoregressive drafting with a lightweight block-diffusion model that generates an entire block in a single parallel forward pass. **DSpark** ([DeepSeek-AI's DeepSpec](https://github.com/deepseek-ai/DeepSpec), which powers DeepSeek-V4 serving) then builds directly on DFlash, adding **Semi-Autoregressive Generation** to recover inter-token dependencies and **Confidence-Scheduled Verification** to make speculation pay off under high concurrency.

This post collects the complete technical notes, mathematical derivations, and system-level insights from both.

---

## Part I — DFlash: Block Diffusion as a Drafter

### 1. The Bottleneck in LLM Inference and Speculative Decoding

LLMs are notoriously memory-bound and bottlenecked by sequential, token-by-token autoregressive generation, which fails to fully utilize modern GPUs. Speculative decoding addresses this by using a small **draft model** to generate candidate tokens for parallel verification by a larger **target model**. However, state-of-the-art methods like **EAGLE-3** still rely on *autoregressive* drafting. This serial drafting process is inherently inefficient, accumulates errors, and effectively caps theoretical speedups at approximately **2–3×**.

On the other end of the spectrum, **Diffusion LLMs (dLLMs)** enable fast parallel text generation but generally underperform autoregressive models in generation quality and require a high number of denoising steps.

**DFlash** bridges this gap: it introduces a speculative decoding framework that uses a lightweight **block diffusion** model for parallel drafting, combining the high-speed parallel generation of diffusion with the lossless accuracy of autoregressive target verification.

### 2. The Mathematics of Speculative Speedup: Autoregressive vs. Diffusion

The efficiency of speculative decoding is governed by the average per-token latency $L$:

$$L = \frac{T_{draft} + T_{verify}}{\tau}$$

where $T_{draft}$ is the drafting time, $T_{verify}$ is the verification cost, and $\tau$ is the expected number of accepted tokens per cycle (inclusive of the target model's bonus token).

*   **Autoregressive Drafting Cost:** $T_{draft} = \gamma \cdot t_{step}$. Here, $\gamma$ is the number of proposed tokens and $t_{step}$ is the latency of a single forward pass. Because drafting cost scales **linearly** with $\gamma$, autoregressive drafters must be extremely shallow (e.g., a single transformer layer in EAGLE-3) to keep latency manageable, severely limiting their representational capacity and acceptance length $\tau$.
*   **Diffusion Drafting Cost:** $T_{draft} = t_{parallel}$. A block diffusion model generates all $\gamma$ tokens in parallel in a single forward pass.

**Insight:** Because $t_{parallel} \ll \gamma \cdot t_{step}$ on modern GPUs, diffusion drafters **decouple drafting cost from the speculation budget**. This allows DFlash to use deeper, more expressive architectures (e.g., a 5-layer model) without sacrificing latency, pushing the Pareto frontier of draft quality and latency far beyond autoregressive limits.

### 3. Architectural Innovation: Target Context Feature & KV Injection

If a small diffusion model speculates from scratch, it achieves poor acceptance lengths because it lacks contextual guidance. DFlash solves this by treating the diffusion draft model as an **adapter** that fuses with the target LLM's deep contextual representations.

During the initial prefill pass, DFlash extracts hidden states from a fixed set of target model layers (uniformly sampled from shallow to deep). Instead of merely fusing these features at the input layer (which dilutes information in deeper draft layers), **DFlash injects these features directly into the Key (K) and Value (V) projections of every single draft layer**.

The mathematical derivation for this KV injection is as follows. First, the extracted target hidden states $H^{(l)}$ are concatenated and fused via a shared projection matrix $W_c$ to create the target context feature $H_t$:

$$H_t = \text{RMSNorm}\left(W_c[H^{(l_1)}; \dots; H^{(l_5)}]\right)$$

For a draft layer $i$, the draft tokens ($H_d$) produce the queries, while the target features ($H_t$) are concatenated with the draft tokens to form the Keys and Values:

$$Q_i = W^Q_i H_d$$
$$K_i = [W^K_i H_t; W^K_i H_d]_{\text{seq}}$$
$$V_i = [W^V_i H_t; W^V_i H_d]_{\text{seq}}$$

This provides strong, persistent conditioning throughout the draft model, fundamentally enabling the acceptance length to scale effectively with the depth of the draft model. The memory overhead for this caching mechanism is **negligible (e.g., ~42 MB for a 35B parameter model)**.

### 4. Training Nuances for Speculative Alignment

Standard block diffusion training divides responses into uniform blocks. DFlash instead heavily tailors its training to mirror inference-time speculative decoding:

*   **Random Anchor Sampling:** DFlash randomly samples "anchor tokens" from the clean response to serve as the first token of a block, masking the remaining tokens for parallel prediction. This perfectly mimics inference, where the draft model is always conditioned on a clean "bonus token" produced by the target model.
*   **Sparse Attention Masking:** Multiple draft blocks are concatenated and processed jointly using **Flex Attention**. Tokens attend bidirectionally within their block and to injected target features, but cross-block attention is strictly masked.
*   **Position-Dependent Loss Decay:** In speculative decoding, an early token error invalidates all subsequent tokens in the draft block. To penalize early errors more heavily, DFlash applies an exponentially decaying weight to the cross-entropy loss:

$$w_k = \exp\left(-\frac{k - 1}{\gamma}\right)$$

Here, $k$ is the token position and $\gamma$ is a scaling factor. This mathematically forces the model to prioritize the accuracy of early positions, resulting in faster convergence and higher overall acceptance lengths.

### 5. Performance and Empirical Results

The framework was evaluated extensively on **LLaMA-3.1** and **Qwen3** models across Math, Code, and Chat tasks.

*   **Speedup:** DFlash achieves over a **6× lossless acceleration** (up to **6.1× on Qwen3-8B Math500**) against standard autoregressive baselines.
*   **SOTA Comparison:** It delivers nearly **2.5× higher speedup than EAGLE-3** while requiring substantially lower verification overhead.
*   **Context Length Adaptation:** While the base draft model is trained on 4K contexts, it adapts remarkably well to long-context scenarios (up to **32K on LongBench**) with just lightweight fine-tuning (3 epochs on 1.6K samples), proving that target features remain highly representative at long contexts.
*   **Serving Viability:** DFlash demonstrates resilient throughput improvements in practical serving frameworks like **SGLang** and **vLLM** (see [vLLM V1 internals]({% post_url 2025-11-30-vLLM %})) across concurrency levels from 1 to 32.

### 6. Expert Insights & Paradigm Shift

Perhaps the most profound insight is a philosophical reframing of Diffusion LLMs. Historically, the AI community has tried to force diffusion models to compete end-to-end with autoregressive models in generation quality—often resulting in slow, cumbersome models.

DFlash demonstrates that **diffusion models are arguably best utilized as specialized, lightweight "drafters."** By confining diffusion models strictly to the drafting stage, we can aggressively minimize denoising steps to maximize parallel hardware utilization. The heavy lifting of quality assurance and reasoning is offloaded to the autoregressive target model during the verification step. This structural synergy might serve as the blueprint for the next generation of high-performance LLM deployment architectures.

---

## Part II — DSpark: Speculative Decoding for High-Concurrency Serving

DFlash showed that a deep parallel drafter can win on latency. DSpark asks the next question: parallel drafters are fast, but can they be made *coherent* enough to keep acceptance high deep into a block—and can we schedule verification so the gains survive a busy serving engine?

### 1. The Core Equation and the Drafter Dilemma

DSpark starts from the same per-token latency equation:

$$L = \frac{T_{draft} + T_{verify}}{\tau}$$

where $\tau$ is the number of accepted tokens, $T_{draft}$ is drafting latency, and $T_{verify}$ is verification latency.

To optimize $L$, you can lower $T_{draft}$, raise $\tau$, or effectively reduce $T_{verify}$. Existing drafters struggle to balance these:

*   **Autoregressive drafters (e.g., EAGLE-3):** Generate high-quality sequences (high $\tau$) because they condition on previous tokens, but $T_{draft}$ grows linearly $O(\gamma)$ with the block size $\gamma$. To keep latency low, they are constrained to very shallow architectures (e.g., 1 layer).
*   **Parallel drafters (e.g., DFlash):** Generate all $\gamma$ tokens in a single $O(1)$ forward pass. This structural advantage allows for deeper architectures (e.g., 5 layers).

**Insight: The Capacity vs. Dependency Trade-off.** Empirical position-wise conditional acceptance reveals a fascinating dynamic. At the very first position, the deeper parallel drafter vastly outperforms the shallow autoregressive one (e.g., **0.88 vs 0.81 on Math**) because it has more parameter capacity. However, because parallel drafters predict tokens *independently*, they cannot model inter-token dependencies. This leads to **multi-modal collisions** (e.g., confusing "of course" and "no problem" into "of problem"), causing rapid acceptance decay deeper into the block.

### 2. Solution Part A: Semi-Autoregressive Generation

To get the best of both worlds—high initial capacity *and* coherent suffix dependencies—DSpark uses a semi-autoregressive split:

**1. The Parallel Stage (Heavy).** A deep parallel backbone (based on DFlash) runs a single forward pass over the whole block, outputting hidden states $h_1, \dots, h_\gamma$ and base logits $U_1, \dots, U_\gamma$.

**2. The Sequential Stage (Lightweight).** DSpark introduces a sequential loop that adds a **transition bias** $B_k(x_0, x_{<k}, x_k)$ to the base logits. The probability of the block is factored autoregressively:

$$P(X \mid x_0) = \prod_{k=1}^\gamma p_k(x_k \mid x_0, x_{<k})$$
$$p_k(v \mid x_0, x_{<k}) = \frac{\exp\left(U_k(v) + B_k(x_0, x_{<k}, v)\right)}{\sum_{u \in V} \exp\left(U_k(u) + B_k(x_0, x_{<k}, u)\right)}$$

The sequential block must be highly efficient ($T_{sequential} \ll T_{parallel}$). DSpark explores two heads for this:

*   **Markov Head (Default):** Restricts the dependency to a first-order transition (only the immediately preceding token). It uses a low-rank factorization:

    $$B(x_{k-1}, \cdot) = W_1[x_{k-1}]W_2 \in \mathbb{R}^V$$

    where $W_1 \in \mathbb{R}^{V \times r}$ is an embedding lookup and $W_2 \in \mathbb{R}^{r \times V}$ is a projection ($r = 256$).
*   **RNN Head:** Maintains a recurrent state $s_k$ to capture full prefix history. The state updates via:

    $$s_k = \sigma(W_g z_k) \odot s_{k-1} + (1 - \sigma(W_g z_k)) \odot \tanh(W_c z_k)$$
    $$B_k(x_{<k}, \cdot) = W_2^\top \tanh(W_o z_k)$$

    where the input $z_k = [s_{k-1}; W_1[x_{k-1}]; h_k]$.

**Insight:** Injecting a tiny amount of local autoregression is vastly more efficient than scaling parallel parameters. A shallow **2-layer DSpark** model outperforms a deeper **5-layer DFlash** model across all domains. The Markov loop adds a negligible **0.2%–1.3%** latency overhead but boosts accepted length by up to **30%**.

### 3. Solution Part B: Confidence-Scheduled Verification

Generating large, coherent blocks is only half the battle. In a high-concurrency engine, indiscriminately verifying trailing tokens with a high risk of rejection wastes precious batch capacity.

**The Confidence Head.** DSpark estimates the per-position conditional probability that a token will survive verification (assuming all prior tokens survived):

$$c_k = \sigma\left(w^\top[h_k; W_1[x_{k-1}]]\right)$$

It is supervised using the analytical acceptance rate based on the **Total Variation (TV) distance** between the draft ($p^d$) and target ($p^t$) distributions:

$$c^*_k = 1 - \frac{1}{2}\|p^d_k - p^t_k\|_1$$

**Sequential Temperature Scaling (STS).** Neural networks are notoriously overconfident. Because the hardware scheduler needs *absolute* (not just relative) probabilities to estimate exact throughput, DSpark calibrates the **joint cumulative probability** $\prod_{i\le k} c_i$ using STS. A left-to-right 1D grid search finds temperature scalars that minimize the **Expected Calibration Error (ECE)** without disrupting the rank order of tokens.

**The Hardware-Aware Prefix Scheduler.** The scheduler's goal is to maximize the expected system-wide token throughput $\Theta$ dynamically. Given $R$ requests, total batch size $B = \sum_{r=1}^R (1 + \ell_r)$, and expected accepted tokens $\tau$, the scheduler maximizes:

$$\Theta = \tau \cdot SPS(B)$$

where $SPS(B)$ is the engine's pre-profiled **Steps Per Second** at batch size $B$.

#### Derivation Insight: The Danger of Selection Bias (Appendix A)

The DSpark authors provide a crucial theoretical derivation highlighting why the scheduler must use an **early-stopping mechanism**. Standard speculative decoding relies on the **non-anticipating property**: the decision to accept/reject must be independent of the future realization of draft tokens.

If the scheduler does a global search without stopping, it creates a *selection bias*. For instance, assume $x_1$ can be $A$ or $B$. If the model samples $x_1 = A$, it might yield a highly confident $c_2$, prompting the scheduler to verify length $\ell = 2$. If $x_1 = B$, it might yield a low $c_2$, prompting the scheduler to fall back to $\ell = 0$ (meaning $x_1$ isn't even sent for verification). Thus, the probability of token $A$ making it to the output gets artificially inflated, irreparably distorting the target distribution $p^t$.

DSpark enforces causality by **stopping the greedy search the moment throughput $\Theta$ drops**, ensuring the admission decision relies solely on previously processed states.

### 4. Training Objectives

The draft model is trained with a **frozen target model** using a combined, position-weighted loss. The weight $w_k = \exp(-(k-1)/\gamma)$ heavily prioritizes earlier tokens, as a rejection early on invalidates the entire subsequent block.

$$L = \alpha_{ce} L_{ce} + \alpha_{tv} L_{tv} + \alpha_{conf} L_{conf}$$

*   **$L_{ce}$ (Cross Entropy):** Predict the ground truth token: $-\sum w_k \log p^d_k(x^*_k)$.
*   **$L_{tv}$ (Distribution Matching):** Directly proxy the expected acceptance rate by minimizing TV distance: $\sum w_k \|p^d_k - p^t_k\|_1$.
*   **$L_{conf}$ (Confidence Loss):** Binary cross-entropy matching $c_k$ to the soft label $c^*_k$.

### 5. Production Engineering and System-Level Insights

Deploying DSpark into the live **DeepSeek-V4** system (see the companion notes on [DeepSeek-V4 Infra]({% post_url 2026-04-25-DeepSeek-V4-Infra %}) and [DeepSeek-V4 Architecture & Training]({% post_url 2026-04-26-DeepSeek-V4-Arch-Train %})) required navigating messy hardware realities.

*   **Jagged Capacity Curves & Asynchronous Scheduling:** Physical hardware throughput $SPS(B)$ is not a smooth curve; it has jagged, step-wise cliffs. To integrate with **Zero-Overhead Scheduling (ZOS)** and not stall the CUDA pipeline, DSpark runs its scheduler *asynchronously*. It approximates the batch capacity $K$ using confidence scores from *two steps prior*. This creates a **"causal barrier"** that naturally prevents the selection bias discussed above, allowing the scheduler to bypass early-stopping and search globally across jagged hardware profiles without breaking target distribution math.
*   **Variable-Length Kernel Routing:** To dynamically route variable draft lengths in the same batch, logical sequences must be decoupled from physical execution. DSpark **flattens all tokens into independent elements** inside compute kernels, managing dependencies via marker tensors in sparse attention. This prevents severe GPU under-utilization caused by standard padding.

**The Bottom Line:** DSpark fundamentally shifts the Pareto frontier of LLM serving. Under strict Service Level Agreements (SLAs)—where baseline systems suffer severe performance cliffs—DSpark aggressively restricts unpromising verifications to conserve batch capacity, unlocking interactivity tiers previously thought unattainable (**accelerating per-user speeds by 60%–85% at matched system capacities**).

---

## Putting It Together

DFlash and DSpark form a clean progression in the speculative-decoding stack:

1.  **DFlash** establishes that block diffusion is the right *drafting primitive*: it decouples drafting cost from the speculation budget ($T_{draft} = t_{parallel}$ instead of $\gamma \cdot t_{step}$), enabling deep, expressive drafters and KV-injected target conditioning for **6× lossless** speedups.
2.  **DSpark** patches the one structural weakness of pure parallel drafting—**independence between tokens**—with a lightweight **semi-autoregressive** Markov/RNN head, then makes the speedup *robust under load* with **confidence-scheduled verification** that allocates scarce batch capacity only to tokens likely to be accepted.

The shared thread across both: keep the *target* model as the lossless arbiter of quality, and aggressively optimize the *drafter* and the *verification schedule* around hardware reality. Together they suggest a blueprint for the next generation of high-throughput, low-latency LLM serving.
