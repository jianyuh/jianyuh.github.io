---
layout: post
title: "Why Cross-Entropy Wins: Proper Scoring Rules, Brier Loss, and Sparse Alternatives"
date: 2026-08-25
categories: [LLM, Optimization, Theory]
tags: [Cross-Entropy, Proper-Scoring-Rules, Softmax, Brier, Sparsemax, Entmax]
---

Reading notes on:
- [除了交叉熵，LM Loss还有什么选择？](https://kexue.fm/archives/11854)

Language modeling is usually introduced as next-token classification. That framing is convenient and slightly misleading. The real object is a distribution $p$ over valid continuations, not a single correct label. Once you take that view seriously, the interesting question is no longer "why do we use cross-entropy?" but rather "which losses can even be estimated from samples, and what optimization geometry do they induce?"

This note is about that boundary. It starts from proper scoring rules, explains why some seemingly natural distances are unusable for large-scale LM training, and then shows why the **cross-entropy + softmax** pairing remains the canonical choice even though there is a legitimate family of alternatives.

---

## 1. A Language Model Learns a Distribution, Not a Label

For a loss to be usable in stochastic language-model training, it has to admit Monte Carlo estimation from samples:

$$
L(p, q)
=
\sum_i p_i S(q, i)
=
\mathbb{E}_{i \sim p}[S(q, i)],
\qquad
p = \arg\min_q L(p, q).
$$

That second condition is what makes the scoring rule **proper**: the true distribution minimizes expected loss.

This immediately rules out some popular divergences as practical LM objectives. Take **total variation**:

$$
TV(p, q)
=
\frac{1}{2}\sum_i |p_i - q_i|
=
\mathbb{E}_{i \sim p}
\left[
\frac{1}{2}\left|1 - \frac{q_i}{p_i}\right|
\right].
$$

In language modeling, the dataset gives you a sample from $p$, not the value of $p_i$ itself. So the ratio $q_i / p_i$ is not available. The objective cannot be estimated from token samples alone.

That is why proper scoring rules are the relevant class. They are the losses that can be both **truthful** and **sample-compatible**.

---

## 2. The Geometry of Proper Scoring Rules

Define the minimum achievable score for distribution $p$ as

$$
H(p) = L(p, p).
$$

For any proper scoring rule,

$$
L(p, q) \ge H(p).
$$

Because $L(p, q)$ is linear in $p$ for fixed $q$, it is a tangent plane over the simplex. So the entropy surface $H(p)$ must be **concave**: it always lies below its supporting hyperplanes.

That gives the general construction. For any concave entropy $H(q)$ on the simplex,

$$
S(q, i) = H(q) + (e_i - q)\cdot \nabla_q H(q).
$$

This is the master formula behind the standard catalog:

| Rule | Entropy $H(p)$ | Score $S(q, i)$ |
| :--- | :--- | :--- |
| Logarithmic | $-\sum_i p_i \log p_i$ | $-\log q_i$ |
| Brier | $1 - \sum_i p_i^2$ | $\sum_j q_j^2 - 2q_i + 1$ |
| Tsallis | $\frac{1 - \sum_i p_i^\alpha}{\alpha - 1}$ | $\frac{1}{\alpha - 1}\left(\sum_j q_j^\alpha - \alpha q_i^{\alpha - 1} + 1\right)$ |
| Spherical | $1 - \lVert p \rVert_\alpha$ | $\frac{1}{\alpha - 1}\left(1 - \frac{q_i^{\alpha - 1}}{\lVert q \rVert_\alpha^{\alpha - 1}}\right)$ |

So cross-entropy is not unique in principle. It is just one member of a large family.

---

## 3. Why Cross-Entropy Is Special Anyway

Cross-entropy earns its status from a very specific uniqueness property: it is the only proper scoring rule whose score depends **only on the predicted probability of the correct class**.

Assume

$$
S(q, i) = S(q_i).
$$

Then, under the simplex constraint, the scoring-rule construction implies

$$
q_i S'(q_i) = -c,
$$

whose solution is

$$
S(q_i) = -c \log q_i.
$$

That is the logarithmic score.

This matters because it removes irrelevant dependence on how the model distributes mass among *wrong* classes. The loss only asks one question: how much probability did you give the observed token?

That decoupling is exactly what makes cross-entropy so easy to optimize at scale.

---

## 4. Gradient Dynamics: Cross-Entropy vs. Brier

The loss function cannot be separated from the output activation. With softmax logits $q = \sigma(z)$, the gradients look very different.

For the logarithmic score,

$$
\nabla_z S(q, i) = q - e_i.
$$

This is the famous clean gradient. The softmax Jacobian and the log-loss derivative cancel in a way that preserves a strong learning signal even when the model is confidently wrong.

For the Brier score,

$$
\nabla_z S(q, i)
=
2(\text{diag}(q) - q q^T)(q - e_i).
$$

Now the gradient is modulated by the softmax Jacobian itself. If the model collapses onto the wrong class with $q_j \approx 1$, then $\text{diag}(q) - qq^T \to 0$ and the gradient vanishes. This is the **confident-wrong saturation trap**.

That single fact explains the usual trade:

- **Cross-entropy** converges fast and cleanly.
- **Brier** is more robust to label noise because it suppresses large mistaken gradients late in training.

There is a second geometric difference. Cross-entropy is convex in the logits of the final layer, while Brier under softmax is not. So Brier buys robustness at the price of a more awkward optimization landscape.

---

## 5. If You Change the Entropy, You Should Change the Activation Too

Once you leave the logarithmic entropy, softmax is no longer the canonical inverse map. The right framework is the **Fenchel-Young** construction:

$$
\Phi(z) = \max_{p \in \Delta}(p \cdot z + H(p)),
\qquad
q = \nabla_z \Phi(z).
$$

For entropy families of the form

$$
H(q) = g\!\left(\sum_i q_i^\alpha\right),
$$

the optimality condition gives

$$
q_i
=
\left[
\frac{\lambda - z_i}{\alpha g'(t)}
\right]_+^{\frac{1}{\alpha - 1}}.
$$

That one expression recovers several important activations:

- $\alpha \to 1$ gives **softmax**;
- $\alpha = 2$ gives **sparsemax**;
- $\alpha = 1.5$ gives **entmax-1.5**.

This is the cleanest way to think about sparse output distributions. You do not "bolt sparsity onto softmax." You change the entropy and let the corresponding Fenchel-Young map give you the right activation.

That is also why these ideas keep resurfacing outside pretraining, for example in calibration-aware RL setups such as [Inkling]({% post_url 2026-07-17-Inkling-975B-MoE %}), where proper scoring rules matter because the model is being trained to express uncertainty, not only to imitate labels.

---

## Takeaways

1. **The feasible objective class is smaller than it looks.** If a loss cannot be estimated from token samples, it is not a serious LM training objective no matter how elegant it looks on paper.
2. **Proper scoring rules are the right abstraction.** They connect statistical truthfulness, entropy geometry, and stochastic optimization in one framework.
3. **Cross-entropy wins for structural reasons.** It is the only proper rule depending only on $q_i$, and under softmax it yields the unusually clean gradient $q - e_i$.
4. **Alternatives still matter.** Brier, Tsallis, sparsemax, and entmax are useful when label noise, abstention, or sparse support matter more than raw optimization speed.
5. **If you change the loss, change the link.** Fenchel-Young makes the pairing explicit: alternative entropies want alternative output maps.
