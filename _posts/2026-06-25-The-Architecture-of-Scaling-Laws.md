---
layout: post
title: "The Architecture of Scaling Laws in Deep Learning"
date: 2026-06-25
categories: [Deep Learning, Theory]
tags: [Scaling Laws, LLM, Chinchilla, Kaplan, Optimization]
---

Here is a comprehensive, deeply technical reading note based on Lilian Weng's [*Scaling Laws, Carefully*](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/). This revisits and deepens an [earlier note on scaling laws]({% post_url 2024-12-23-scaling-law %}).

***

# Reading Notes: The Architecture of Scaling Laws in Deep Learning

**Overview:** 
Scaling laws mathematically formalize how deep learning models improve predictably as we scale up model size ($N$), dataset size ($D$), and compute budget ($C$). At its core, the study of scaling laws is a resource allocation problem: given a finite amount of compute, what is the mathematically optimal way to distribute it between parameters and data? **These laws reveal that loss follows a predictable power-law curve, manifesting as a straight line on a log-log plot**. 

This reading note deconstructs the history, mathematical formulations, and recent insights in both infinite-data and data-constrained regimes.

---

### 1. Early Foundations: Predicting the Irreducible
The predictability of generalization error ($\epsilon$) was explored long before modern LLMs. Amari et al. (1992) used Bayesian approaches to derive learning curves, finding that error decays according to power laws based on data noise and algorithmic stochasticity (e.g., $\epsilon \sim c \cdot D^{-1} + E$). 

Later empirical studies by Hestness et al. (2017) formalized that learning curves universally exhibit three phases: a small-data phase (near random guessing), a power-law phase, and an irreducible-error phase. **Crucially, architecture changes merely shift the offset ($E$) of the power-law fit; the exponent ($\alpha$) is a property of the problem domain**. 

Rosenfeld et al. (2020) advanced this by creating a joint function for error across both data and model size:
$\hat{L}(D, N) \approx AN^\alpha + BD^\beta + E$
Here, $A, B, \alpha, \beta \ge 0$ are constants and $E$ is the irreducible error.

---

### 2. The Infinite-Data Regime: Kaplan vs. Chinchilla

The central friction in scaling law literature lies between the influential findings of [Kaplan et al. (2020)]({% post_url 2024-12-23-scaling-law %}) and the corrective framework of Hoffmann et al. ([Chinchilla, 2022]({% post_url 2024-12-23-scaling-law %})). (For practical calculations of how these scaling laws translate to cluster hardware requirements, see [Infra Math for LLM Training]({% post_url 2025-11-28-LLM-Train-GPU %}).)

#### Kaplan et al.: "Scale the Model Faster"
Kaplan et al. analyzed Transformer language models (up to 1.5B parameters) and found that loss $\hat{L}$ scales as a power law with $N$, $D$, and $C$ independently. 
They introduced a critical heuristic for estimating training compute in FLOPs. Assuming context length is relatively small, forward-pass FLOPs per token ($C_{fwd}$) are roughly $2N$. Since backpropagation requires roughly twice the compute of the forward pass, total training FLOPs ($C$) over $D$ tokens is:
**$C \approx 6ND$**.

Kaplan’s joint dependence equation took the form:
$\hat{L}(N, D) = [(aN)^{\frac{\alpha}{\beta}} + bD]^{-\beta}$.
**Kaplan’s conclusion:** $N_{opt} \propto C^{0.73}$, suggesting that for a 10x increase in compute, model size should scale by 5.5x but tokens by only 1.8x. They argued it is more efficient to train a massive model and stop before convergence.

#### Chinchilla (Hoffmann et al.): "Equal Scaling"
The Chinchilla paper fundamentally overturned Kaplan's conclusion, proving that Kaplan's recommendations left massive models severely undertrained. By sweeping over 400 models with fixed compute budgets, they sought to solve the optimization problem:
$N_{opt}(C), D_{opt}(C) = \arg \min \hat{L}(N, D) \text{ subject to } C \approx 6ND$.

They used three methods to fit their curves (varying token budgets, IsoFLOP profiles, and a parametric fit). The parametric fit (Method 3) provides a beautiful closed-form derivation. Starting from Rosenfeld's joint form, they substituted $D = C / 6N$:
$\hat{L}(N) = AN^{-\alpha} + B(\frac{C}{6})^{-\beta}N^\beta + E$.

Taking the derivative with respect to $N$ and setting it to zero yields:
$\alpha A N^{-\alpha - 1} = \beta B (\frac{C}{6})^{-\beta} N^{\beta - 1}$.
Solving for optimal $N$ and $D$:
**$N_{opt} = (\frac{\alpha A}{\beta B})^{\frac{1}{\alpha + \beta}} (\frac{C}{6})^{\frac{\beta}{\alpha + \beta}}$**.
**$D_{opt} = (\frac{\beta B}{\alpha A})^{\frac{1}{\alpha + \beta}} (\frac{C}{6})^{\frac{\alpha}{\alpha + \beta}}$**.

Empirically, Chinchilla found that $\alpha \approx \beta$. **Therefore, $N_{opt} \propto C^{0.5}$ and $D_{opt} \propto C^{0.5}$, meaning model size and training data should be scaled in equal proportions**. 

#### Reconciling the Two Paradigms
Why the massive discrepancy? Pearce & Song (2024) reconciled the two by revealing two core issues in Kaplan's methodology:
1. **Extrapolation Risk:** Kaplan experimented on small models, and tiny fitting differences compound radically in log-log extrapolations.
2. **Embedding Parameters:** Kaplan excluded embedding parameters ($N_{\setminus E}$), which represent a massive fraction of small models but a negligible fraction of large ones. 

By defining total parameters $N = N_{\setminus E} + \omega N_{\setminus E}^{1/3}$, and plugging this into the loss derivative, the relationship between compute and non-embedding parameters ($C_{\setminus E}$ and $N_{\setminus E}$) ceases to be a clean power law. The local exponent $g = \frac{d \log C_{\setminus E}}{d \log N_{\setminus E}}$ starts near Kaplan's 0.73 for small models (768M to 1.5B) but organically converges to Chinchilla's 0.50 as scale increases to infinity.

---

### 3. Why Power Laws?
Why do deep learning losses decay strictly as power laws? Current theoretical hypotheses include:
*   **Data Manifold Dimension:** Language modeling is akin to regression on a low-dimensional data manifold. An effective model size $N$ partitions a $d$-dimensional space into regions, yielding a linear resolution that scales as $\sim N^{-1/d}$, matching the power-law form.
*   **Quantized Skills:** Knowledge is acquired in discrete "chunks." Because the frequency distribution of skills in data naturally follows a power law (learning common skills fast and rare skills slowly), the resulting loss curve mirrors this decay.

---

### 4. The Data-Limited Regime: Hitting the Wall
Classic scaling laws assume infinite unique, high-quality data. As we exhaust the internet's supply of unique tokens, researchers must turn to multi-epoch training, which historically introduces overfitting or "double descent" (where test loss worsens, then improves).

To model data-constrained optimal scaling, Muennighoff et al. (2023) decomposed token count $D$ into unique tokens ($U_D$) and repeated tokens ($R_D$), modeling token value as exponentially decaying with repetition. Their findings suggest that **excess parameters decay in value faster than repeated data ($r_N < r_D$), implying compute is better spent on more epochs rather than larger models**.

However, Lovelace et al. (2026) updated this by introducing an explicit overfitting penalty term based on the *capacity ratio* (parameters relative to unique tokens, $N/U_D$):
$\hat{L}(N, U_D, R_D) = E + AN^\alpha + B(U_D(1+R_D))^\beta + \color{red}{P \cdot R_D^\delta \cdot (\frac{N}{U_D})^\kappa}$.
This models the insight that larger models are vastly more sensitive to data repetition damage. Additionally, they proved that aggressive weight decay can mitigate this specific overfitting penalty.

---

### 5. The Fragility of Fitting Scaling Laws
Despite their clean mathematical forms, empirically fitting these curves is astonishingly fragile. **Because we fit curves on cheap, small models and extrapolate them orders of magnitude, tiny procedural choices completely warp the predictions**.

Besiroglu et al. (2024) replicated Chinchilla's Method 3 and found that DeepMind's original fit was numerically flawed. An L-BFGS minimizer prematurely terminated because they *averaged* rather than *summed* Huber-loss values across examples, and intermediate parameter rounding compounded the errors. A toy simulation reveals that perturbing loss values by mere milli-loss units (0.001) or artificially restricting the fit to "small models only" can completely alter the apparent exponents of the scaling law. 

**Expert Insight:** When building empirical scaling laws for your own architectures, standardizing the optimization setup (batch ramp, schedules, optimizer states) is just as critical as the loss fitting itself. Ensure you fit across at least three orders of magnitude of scale to insulate against local exponent artifacts. For the practical training-recipe side of these decisions, see the [Smol Training Playbook note]({% post_url 2025-11-29-Smol-Train %}).
