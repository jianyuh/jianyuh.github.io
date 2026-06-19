---
layout: post
title: "The Distributional Lens of Post-Training: SFT, RL, and On-Policy Distillation"
date: 2026-06-13
categories: [LLM, Post-training, RL]
tags: [SFT, RLHF, RLVR, Distillation, On-Policy-Distillation, KL-Divergence, Catastrophic-Forgetting, GRPO]
---

Reading notes based on:
- [SFT, RL, and On-Policy Distillation Through a Distributional Lens](https://nrehiew.github.io/blog/sft_rl_opd/)

A language model is, at its core, a **distribution over sequences**. Every post-training method: SFT, RL, distillation, is just a different way of reshaping that distribution. The thing that actually distinguishes these methods is two-fold: *how they define the target distribution*, and *the mechanism by which they move the model toward it*.

These notes look at post-training through that single distributional lens, and arrive at one load-bearing conclusion: **on-policy data is the ingredient that makes the difference.**

---

## 1. Supervised Fine-Tuning: Forward KL and Catastrophic Forgetting

In SFT, the target distribution is **fixed and externally defined** by an annotated dataset. SFT uses cross-entropy training to pull the model's starting distribution toward that external data.

Up to a constant, SFT minimizes the **forward KL divergence** between the data distribution $p$ and the model distribution $q_\theta$:

$$
D_{KL}(p \,\|\, q_\theta) = \sum_x p(x) \log \frac{p(x)}{q_\theta(x)} = \underbrace{\sum_x p(x) \log p(x)}_{-H(p)} - \underbrace{\sum_x p(x) \log q_\theta(x)}_{-H(p,\,q_\theta)} = -H(p) + H(p, q_\theta).
$$

Since $H(p)$ is constant with respect to $\theta$, minimizing forward KL is exactly minimizing the cross-entropy $H(p, q_\theta)$ — i.e., negative log-likelihood on the demonstrations.

**Characteristics and failure modes:**

- **[Catastrophic forgetting]({% post_url 2026-06-12-LLM-Evaluation-Architecture %}#catastrophic-forgetting).** Because the objective is pure negative log-likelihood, the model's *starting* distribution barely matters — it is pulled straight toward the dataset with no built-in reason to prefer nearby solutions. Forward KL is **mode-covering**, so the model spreads mass to cover the data even at the cost of pre-existing capabilities.
- **Dense, uniform gradients.** SFT exerts gradient pressure uniformly across the whole distribution. It pushes up the probability of *every* demonstrated token — whether it's a task-critical math operator or a throwaway style token like "therefore." Confident, low-entropy predictions get forced to fit divergent labels, producing dense, redundant updates that overwrite prior knowledge.
- **Where it shines.** That same direct pull makes SFT excellent for **cold-start** tasks, where the output format has to be drastically altered.

---

## 2. Reinforcement Learning: Reverse KL and the Nearest Task-Solving Policy

RL has **no arbitrary external target**. Instead the model generates on-policy rollouts, scores them with a reward function, and updates via policy gradient to increase expected reward.

**Characteristics of RL optimization:**

- **Reverse KL minimization.** RL aligns more closely with *reverse* KL minimization, which empirically forgets less than forward KL.
- **Data-dependent regularization.** RL inherently scales updates by certainty. High-diversity, high-variance groups receive smaller updates; consistent high-reward samples trigger aggressive ones.
- **Sparse parameter updates.** RL updates a small subnetwork via sparse but full-rank updates: making each update less redundant and more task-critical than SFT's dense rewrites.

**The on-policy anti-forgetting mechanism.** The most compelling explanation for why RL preserves prior capabilities is its reliance on **on-policy data**. Using a binary reward as an analogy, RL behaves like rejection sampling: filtering out bad generations and rewarding good ones. Because the policy is trained exclusively on states the model *already visits*, the algorithm is heavily constrained to find the **closest reachable optimum** $\pi^*$ among all optimal policies $P^*$. RL reshapes existing high-probability regions rather than dragging the model toward a distant external target.

---

## 3. On-Policy Distillation: Pseudo-RL

*(For a deeper dive into the mechanics and failure modes of OPD, see the previous post: [Revisiting On-Policy Distillation]({% post_url 2026-04-19-On-Policy-Distillation %}).)*

On-Policy Distillation (OPD) is a hybrid: it uses a **teacher signal like SFT**, but gathers its data **directly from the student via on-policy sampling like RL**. Gradients pull the student toward the teacher's distribution via reverse KL divergence.

**On-Policy Self-Distillation (OPSD).** Here the teacher and student are *the exact same model*, except the teacher is handed a privileged reference solution as a prefix and generates target log-probabilities from it.

- **The clipping problem.** Because teacher and student are nearly identical, *style* tokens (e.g., "wait") often show much higher per-token KL than *critical math* tokens. Updating too aggressively on style tokens can collapse the model: which is why per-token clipping becomes necessary.
- **Bias vs. variance.** This dynamic makes OPSD resemble **RLHF** (which needs KL penalties and trust-region clipping because of biased reward models) more than **[RLVR]({% post_url 2025-11-22-RLVR-Limit %})** (where low bias permits loosened constraints like [GRPO]({% post_url 2025-01-20-deepseek-R1-Kimi-k1.5 %})). OPD supplies a unique reward *per token*, but at the cost of more noise and bias per update than the sparse outcome rewards of RLVR.

---

## 4. The Minimal Code-Editing Experiment

To probe generalization and forgetting, consider a **Minimal Code Editing** task: fix bugs without touching the uncorrupted code.

**Surprising results:**

1. **OPD from SFT outperforms SFT.** Students trained via OPD: using *either* an RL teacher or an SFT teacher: ended up nearly identical, slightly beating the RL teacher and substantially beating the SFT teacher.
2. **Inherited anti-forgetting.** Even when distilled from an SFT teacher that had *already* suffered catastrophic forgetting, the OPD student forgot only slightly and largely preserved its general coding ability.

**Why does this happen?**

- **On-policy data is the key.** The *source* of the data matters more than the teacher's distribution. Because OPD trains on the student's own state distribution, it inherits RL's implicit KL regularization and picks the nearest task-solving policy. The practical implication is striking: you can **brute-force a specialized SFT expert and then use OPD to extract the capability safely.**
- **Targeted supervision.** Traditional distillation trains on states the *teacher* visits; OPD trains on prefixes the *student* generates: correcting the student's actual mistakes and preventing compounding errors.
- **Distributional shaping.** KL matching is *not* the same as reward maximization. The teacher conveys information about uncertainty, reasoning structure, and alternative continuations: reshaping the student's distribution and improving sampling behavior without slavishly cloning greedy outputs. Tellingly, OPD training curves show a **sudden, drastic entropy collapse**: the signature of mode-seeking reverse KL: in contrast to RL's gradual reward climb.

---

## 5. The Future of Post-Training: Searching for the "Best" Algorithm

Modern pipelines typically run:

$$
\text{Pretrain} \;\rightarrow\; \text{SFT} \;\rightarrow\; \text{RL} \;\rightarrow\; \text{OPD}
$$

Each stage plays to its strength: **SFT** instills format adherence, **RL** excels in verifiable domains (math, code), and **OPD / self-distillation** shines in noisy-reward domains (creative writing): and is increasingly used to **merge expert capabilities** in models like [DeepSeek-V4]({% post_url 2026-04-26-DeepSeek-V4-Arch-Train %}) and [GLM-5]({% post_url 2026-02-20-GLM5 %}).

A hypothetical *compute-optimal* post-training algorithm has to perfectly balance capability gains against minimal KL movement. The defining insight of this analysis is that **on-policy data is the absolute load-bearing ingredient** for that balance:

- SFT is inefficient because of **distribution mismatch**.
- RL's outcome rewards suffer from **sparse credit assignment**.

So the optimal future algorithm must combine three properties:

1. the **dense signal** of distillation,
2. the **unbiased objective** of RLVR, and
3. the **grounding** of on-policy data.

Get all three at once, and you have a method that gains capability while barely moving the distribution: the closest thing to a free lunch post-training has to offer.
