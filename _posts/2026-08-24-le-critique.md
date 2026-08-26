---
layout: post
title: "Le Critique: Privileged Value Functions and the Return of Credit Assignment in LLM RL"
date: 2026-08-24
categories: [LLM, RL]
tags: [RL, Credit-Assignment, Value-Functions, PPO, GRPO, TETHER]
---

Reading notes on:
- [LE CRITIQUE: Privileged Value Functions for LLM Reinforcement Learning](https://arxiv.org/pdf/2608.16739)

Modern LLM RL mostly walked away from critics. The reason was pragmatic: actor-critic systems are harder to scale, harder to keep numerically aligned, and much easier to destabilize than group-relative baselines. That trade made sense in the throughput-driven era mapped out by [StreamRL]({% post_url 2025-05-04-streamRL %}) and [RL Systems Mind the Gap]({% post_url 2026-06-19-RL-Mind-The-Gap %}).

But the cost of dropping the critic is real. Group baselines give one blunt sequence-level signal to an entire response, which is exactly the wrong shape for long reasoning traces. **Le Critique** asks whether token-level credit assignment can come back without resurrecting the instability that made the field abandon PPO-style critics in the first place. Its answer is elegant: let the critic see **privileged but admissible information**, then mix that value estimate with the reliable group baseline using a learned coefficient.

![Privileged critic branch and TETHER's adaptive mixture of group and value baselines](/assets/images/le_critique_tether.svg)

---

## 1. What the PPO to GRPO Shift Bought, and What It Lost

The generic token-normalized policy-gradient estimator is

$$
\hat{\nabla}_\theta J
=
\frac{1}{\sum_{i=1}^B T_i}
\sum_{i=1}^B \sum_{t=1}^{T_i}
w_{i,t}\,\hat{A}_{i,t}\,\nabla_\theta \log \pi_\theta(y_{i,t} \mid h_{i,t}).
$$

Everything interesting is packed into the advantage estimator $\hat{A}_{i,t}$.

In actor-critic methods, that advantage can vary token by token. In group-relative methods, it is effectively sequence-level. The leave-one-out baseline

$$
b_i^{LOO} = \frac{1}{K - 1}\sum_{j \ne i} R_j
$$

and the GRPO baseline

$$
A_i^{GRPO} = R_i - \bar{R}
$$

are equivalent up to a constant scale:

$$
A_i^{GRPO} = \frac{K - 1}{K} A_i^{LOO}.
$$

That is why critic-free RL scales cleanly: the baseline is training-free and synchronization-light. But it is also why it struggles to tell *which* reasoning step mattered. The whole trajectory shares a single reward contrast.

This explains the historical split:

| Method | Credit assignment | Infra burden | Failure mode |
| :--- | :--- | :--- | :--- |
| PPO / actor-critic | token-level | high | critic lag, instability |
| GRPO / RLOO | sequence-level | low | blunt temporal credit |

The paper's aim is not to revert to old PPO. It is to recover critic granularity without paying the full systems tax.

---

## 2. The Key Theoretical Boundary: Admissible Privilege

The crucial idea is that a critic can condition on more information than the policy, **as long as that extra information does not leak the sampled action itself**.

The sufficient admissibility condition is

$$
\mathbb{E}\!\left[b(h_{i,t}, z_{i,t}) \nabla_\theta \log \pi_\theta(y_{i,t} \mid h_{i,t}) \mid h_{i,t}\right] = 0
\quad \Longleftarrow \quad
z_{i,t} \perp\!\!\!\perp y_{i,t} \mid h_{i,t}.
$$

So the question becomes: what counts as admissible privilege?

**Safe examples:**

- gold answers or solved boards;
- verifier rubrics or proof sketches;
- sibling rollouts and rewards from the same prompt group, excluding the current trajectory.

**Unsafe examples:**

- future tokens of the trajectory being scored;
- the realized terminal reward of that exact trajectory;
- environment feedback that only exists because the current action was taken.

This distinction is load-bearing. It turns privileged critics from a hand-wavy systems trick into a proper control-variate construction.

---

## 3. Privileged Value Functions

The proposed critic is a **Privileged Value Function (PVF)**:

$$
V^\pi(h_{i,t}, z_{i,t}) = \mathbb{E}_{\tau_i \sim \pi}[R_i \mid h_{i,t}, z_{i,t}].
$$

The paper's motivating example is Sudoku. A standard critic observing only a partial board has to implicitly solve a difficult latent inference problem to predict the return. A privileged critic that also sees the completed board turns that into verification rather than search.

The gain is formal:

$$
\mathbb{E}\Big[(R_i - \mathbb{E}[R_i \mid h_{i,t}, z_{i,t}])^2\Big]
\le
\mathbb{E}\Big[(R_i - \mathbb{E}[R_i \mid h_{i,t}])^2\Big].
$$

So, with an optimal predictor, adding admissible privilege can only reduce variance.

That makes PVFs quite different from self-distillation or reward shaping. They are not changing the RL objective. They are estimating a better baseline.

---

## 4. TETHER: Let the Baseline Decide When to Trust the Critic

Even a principled critic can be wrong early in training. The paper's fix is **TETHER**, an adaptive mixture between the stable group baseline and the learned value baseline:

$$
b_{i,t}^{TETHER} = (1 - \rho)\, b_i^{LOO} + \rho\, V_{i,t}.
$$

Define the group advantage and value correction as

$$
A_B = R - B,
\qquad
\Delta = V - B.
$$

Minimizing the least-squares prediction error gives the optimal mixture coefficient

$$
\rho^*
=
\text{clip}_{[0,1]}
\left(
\frac{\mathbb{E}[A_B \Delta]}{\mathbb{E}[\Delta^2]}
\right).
$$

This is a good design for two reasons:

1. When the critic is useless, $\rho$ collapses toward zero and the algorithm falls back to the safe group baseline.
2. When the critic becomes predictive, $\rho$ increases smoothly and token-level credit gradually takes over.

In implementation, $\rho$ is smoothed with an EMA using decay **0.95**, and the paper insists on an **evaluate-before-train** protocol so that the critic is scored on data it has not just memorized.

---

## 5. What the Results Actually Say

The most direct metric is critic explained variance:

| Task | Standard VF EV | PVF EV |
| :--- | :---: | :---: |
| Sudoku ($K=4$) | 0.032 | 0.245 |
| CodeIO ($K=4$) | 0.594 | 0.751 |
| Reasoning Gym ($K=8$) | 0.499 | 0.519 |

That is a big jump, especially on Sudoku where the latent search problem is hardest.

But the headline is more nuanced than "critic wins everywhere."

- On **Sudoku**, TETHER mitigates the damage from a weak standard critic, but the plain group baseline still slightly outperforms it in final reward.
- On **MiniF2F** and **Reasoning Gym**, TETHER wins, which suggests token-level credit matters most when partial prefixes already contain useful proof structure or semantic evidence.

The paper also highlights a long-context tuning detail that is easy to miss and likely more general: for long CoT, $\lambda_{GAE}$ may need to sit extremely close to 1. The reported sweet spot is

$$
\lambda_{GAE} = 0.999888,
$$

chosen so the first token in an **8,192-token** trajectory still retains roughly **40%** of the terminal reward signal. A default value like 0.95 simply washes the credit out.

---

## 6. Why This Matters

The paper restores an old idea in a more modern form:

1. **Critics do not need to be policy clones.** They can be privileged verifiers as long as the privilege is admissible.
2. **Group baselines remain the right bootstrap.** TETHER is persuasive precisely because it does not force the critic to be trusted before it earns that trust.
3. **The real value of critics is temporal resolution.** They help most when long reasoning traces contain locally diagnostic partial structure.
4. **This is a control-variate story, not a reward-model story.** The objective remains the same; only the variance changes.

My read is that this is the right way to bring critics back into large-scale LLM RL. It does not pretend the GRPO era was wrong. It accepts the systems constraints that killed old actor-critic pipelines, then reintroduces value functions only where they can be mathematically defended and operationally hedged.
