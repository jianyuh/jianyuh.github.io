---
layout: post
title: "team@k: Scaling Discovery through Test-Time Communication"
date: 2026-09-23
categories: [Agents, Inference]
tags: [TestTimeCompute, MultiAgent, Discovery, ARCAGI3, FrontierCS, BestOfK, Verifiers, Exploration, CodingAgents]
---

Reading notes on:
- [Scaling Discovery through Test-Time Communication](https://arxiv.org/pdf/2609.21032)

Most test-time compute scaling is independent parallel sampling: generate $k$ trajectories, then pick one with a verifier (Best@k) or a vote (majority). That works for short reasoning problems. For long discovery problems it has a structural flaw: **a breakthrough in one trajectory is invisible to all the others.** If one agent finds the key heuristic two hours into a 96-hour budget, its $k-1$ peers keep grinding through the same dead ends for the remaining 94.

This work changes one thing. Agents share intermediate artifacts and *verified* findings while they run. The paper calls this configuration **team@k**. It turns $k$ independent lottery tickets into a relay in which each agent can start from the best result anyone has found so far. The paper backs this with a clean theoretical separation and three benchmarks. It also reports the regimes where communication *loses*, and those are the most useful part of the paper.

![team@k versus best@k: independent agents each have to clear every stage alone, while a team advances as soon as any agent clears a stage; the separation grows exponentially with depth, but only with a dense verifier and a compute budget large enough to pay the coordination tax](/assets/images/ttc_team_at_k.svg)

---

## 1. Verified Progress Sharing

| | Standard test-time scaling (independent) | Test-time multi-agent communication (team@k) |
| :--- | :--- | :--- |
| **Method** | Parallel sampling; aggregate via Best@k or majority vote | Asynchronous coordination through a shared workspace and "Verified Progress Sharing" |
| **Resource use** | High redundancy; agents re-solve the same sub-problems | Cumulative; agents build on shared breakthroughs to get past plateaus |
| **Intermediate artifacts** | Isolated; failed experiments and intermediate state are thrown away | Broadcast log for code snippets, failures, and reproducible evidence |
| **Theoretical scaling** | Linear or sub-linear returns on compute | Exponential separation in success probability on multi-stage tasks |

The mechanism avoids a central orchestrator on purpose. Agents coordinate through two filesystem primitives:

- **An append-only broadcast log.** When an agent finds a verifiable improvement (a higher benchmark score, a newly passing test), it publishes the evidence there.
- **Slot directories.** Agents use atomic filesystem operations to *claim* distinct search families, which enforces exploration diversity.

The adoption rule is the important part. Peers adopt a finding **only after confirming the improvement**. Nothing spreads through the team on persuasion alone. Parallel search becomes a relay of cumulative, evidence-backed progress, which is roughly how a working research group operates.

This complements the other outer-loop work I've covered recently. [Dream-RSI]({% post_url 2026-09-21-Dream-RSI-Replay-Simulators-Discovery %}) optimizes *which branch to expand*. [Parallel-Distill-Refine]({% post_url 2025-12-25-PDR %}) synthesizes parallel drafts into a bounded workspace between rounds. team@k keeps the agents fully independent and gives them only a shared, verified bulletin board.

---

## 2. The Math of Collective Search

### Sum of minima vs. minimum of sums

Let $X_{ij}$ be the time agent $i$ needs to clear stage $j$ of an $m$-stage problem. The team moves on as soon as *any* agent clears the current stage. Best@k needs *one specific* agent to be fast on every stage:

$$T_{team} = \sum_{j=1}^m \min_{1 \le i \le k} X_{ij} \;\le\; \min_{1 \le i \le k} \sum_{j=1}^m X_{ij} = T_{best}$$

That inequality is the whole idea: a sum of minima versus a minimum of sums.

### The exponential search model

Assume memoryless search, $X_{ij} \sim \text{Exp}(\lambda)$:

- The first discovery among $k$ agents at stage $j$ arrives at $M_j = \min_{i \le k} X_{ij} \sim \text{Exp}(k\lambda)$.
- Team completion time is Erlang: $T_{team} \sim \text{Erlang}(m, k\lambda)$, with $\mathbb{E}[T_{team}] = \frac{m}{k\lambda}$.
- A single agent takes $\frac{m}{\lambda}$ on average, so the team is a **$k$-fold speedup in expected time**.

### Proposition 1: exponential separation

The expected-time speedup is only linear in $k$. The stronger result is about **success probability under a fixed budget**.

**Assumptions.** The task has $m$ successive, transferable stages. Each agent's budget is $\tau = \alpha m / \lambda$ with $1/k < \alpha < 1$. The budget is too short for any one agent to finish on average, but long enough for the team.

**Derivation** (Chernoff-style, via the MGF and Markov's inequality):

1. Let $S \sim \text{Erlang}(m, 1)$. Its MGF is $\mathbb{E}[e^{\theta S}] = (1 - \theta)^{-m}$ for $\theta < 1$.
2. Applying Markov's inequality with rate function $I(a) = a - 1 - \log a$ gives the tail bounds

   $$\Pr(S \le am) \le e^{-mI(a)} \;\; (a < 1), \qquad \Pr(S \ge am) \le e^{-mI(a)} \;\; (a > 1)$$

3. **Team bound.** $k\lambda T_{team} \sim \text{Erlang}(m, 1)$ and $k\alpha > 1$, so

   $$\Pr(T_{team} > \tau) \le e^{-mI(k\alpha)}$$

   The team's *failure* probability decays exponentially in $m$.
4. **Best@k bound.** For each independent agent $\Pr(T_i \le \tau) \le e^{-mI(\alpha)}$, and a union bound gives

   $$\Pr(T_{best} \le \tau) \le k\, e^{-mI(\alpha)}$$

   Since $\alpha < 1$, best@k's *success* probability decays exponentially to zero as $m$ grows.

So as problems get deeper, independent sampling fails with probability approaching one and the team succeeds with probability approaching one. Adding more independent samples only buys a linear factor $k$ against an exponential $e^{-mI(\alpha)}$.

### Herding

This math assumes the $k$ agents search independently *within* each stage. If they herd, collapsing into $g < k$ effectively independent groups, the per-stage discovery rate drops from $k\lambda$ to $g\lambda$ and the advantage shrinks with it. The slot-claiming protocol exists to prevent this: each agent works a non-overlapping search family.

The takeaway: the theoretical gains show up only when the environment provides a **dense verifier signal** to guide adoption *and* the protocol **enforces diversity** to prevent premature convergence. Section 6 shows what happens when either is missing.

---

## 3. ARC-AGI-3: Communication Compounds with Scale

ARC-AGI-3 drops agents into unfamiliar grid-world games. Clearing all $l_{max}$ levels means carrying forward what you have learned about the game's hidden rules, so it is a natural multi-stage discovery task.

- **Scaling multipliers.** team@3 matches the solve rate of best@13 (**4.3×**). team@5 matches best@33 (**6.6×**). The multiplier grows with $k$, so the benefit compounds.
- **Unlocking the "unsolvable."**
  - **FT09:** 9.4% (single agent) → **90%** (team@3).
  - **LP85:** unsolved in all 64 single-agent trials → **65%** with team@5.

### The RHAE metric

Relative Human Action Efficiency scores an agent against a human action baseline $h_{\ell,e}$. The level score is

$$S_{\ell,e} = \min \left\{ 1.15, \left( \frac{h_{\ell,e}}{a_{\ell,e}} \right)^2 \right\}$$

and the game score is

$$E_e = \min \left\{ \frac{\sum_{\ell=1}^d w_\ell}{\sum_{\ell=1}^n w_\ell}, \frac{\sum_{\ell=1}^n w_\ell S_{\ell,e}}{\sum_{\ell=1}^n w_\ell} \right\}$$

where $d$ is the number of completed levels and $w_\ell = \ell$. Communication **lifts the floor**: the *average* team@5 agent reached **8.9% RHAE**, matching the *best* agent in a 5-member independent pool (8.8%). Sharing progress brings a typical member of the group up to the level of the best independent outlier.

---

## 4. Frontier-CS Polyomino Packing: A Relay of Breakthroughs

This is an NP-hard problem: pack complex shapes into minimal area by improving a C++ packing heuristic.

| Model / configuration | Packing score | vs. prior SOTA (0.894) |
| :--- | :---: | :---: |
| Sonnet 4.6 (single) | 0.883 | −0.011 |
| **Sonnet 4.6 (team@3)** | **0.945** | **+0.051** |
| Opus 4.6 (single) | 0.893 | −0.001 |
| **Opus 4.6 (team@4)** | **0.922** | **+0.028** |

Both single agents stall just short of the prior SOTA. Both teams pass it. The qualitative timeline shows why, as each breakthrough passes through the broadcast log to a different agent:

1. **Shelf packing** (a1): initial baseline.
2. **Skyline / bottom-left** (a1, a2): minimize height.
3. **Contact maximization** (a1): the conceptual breakthrough, that pieces should nest together.
4. **Broadcast rebuild** (a2): a2 re-implemented a1's contact-maximization idea **without reading a1's code**, working only from the log description. The fresh implementation also avoided a1's slow original.
5. **All-orientation search** (a2): adds placement lookahead.
6. **Boundary bonus** (a3): rewards fitting against box walls.

Step 4 is the interesting one. What passed between agents was an *idea plus evidence that it worked*, not an artifact to copy. One agent's concept became the starting point for another agent's implementation, which is how a team gets past the plateau where a single agent stalls.

---

## 5. MNIST Classifier Compression: Joining Two Branches

The task: produce the smallest MNIST classifier (code + weights) that keeps $\ge 99.4\%$ accuracy, within 96 hours.

| Configuration | Size |
| :--- | ---: |
| **team@4 (GPT-5.6 Sol)** | **1,957 bytes** |
| Human baseline | 2,461 bytes |
| best@4 (independent) | 3,160 bytes |

That is a new SOTA, **20.5% smaller than the best human submission**. best@4 does not even beat the human baseline.

### The architecture

The winner is a **recurrent CNN with filter reuse**: one set of filters applied repeatedly, which buys depth without adding parameters to store. The feature-map update uses GroupNorm with 4 groups and SiLU:

$$\tilde{h}_t = \text{SiLU}\big(h_{t-1} + g_t \odot \text{GN}_4(C(\text{SiLU}(D(h_{t-1})))) + b_t\big)$$

To recover accuracy lost to extreme compression, the agents found **scoring-head factorization**: combine a learned weighted sum $Px$ with 12 directly selected "bypass" feature indices $S$:

$$\text{scores} = W \begin{bmatrix} Px \\ x_S \end{bmatrix}, \qquad S = (2, 32, 1, 40, 102, 30, 124, 48, 93, 122, 53, 126)$$

### Quantization for the compressor

The final reduction came from **quantization-aware training** that restricts projection weights to scaled integers in $\\\{-2, -1, 0, 1, 2\\\}$. The point is less the bit width than what the small palette does for `gzip -9`: **44.6%** of the stored integers were $0$ or $\pm 1$, so the compressed payload is highly redundant and shrinks accordingly.

The result came from **joining two branches**: one agent's recurrent architecture and another agent's integer-quantization strategy. Neither branch alone got there. Under best@k they would have lived in separate trajectories and never been combined.

---

## 6. Boundary Conditions: When Communication Loses

Communication is not free, and the paper states its failure regimes plainly.

- **The coordination tax.** Reading the log, synchronizing, and verifying others' claims costs tokens. In low-compute regimes ($\le$ **~400K tokens**), independent agents *beat* teams, because the overhead exceeds the extra discovery rate.
- **Verification is required.** On **Terminal-Bench 2.0**, team@2 did *not* beat pass@2. The authors attribute this to sparse feedback: effectively $m \approx 1$, a single pass/fail at the end. Without frequent, discriminative intermediate scores, agents cannot tell a breakthrough from a dead end. Adoption then becomes guesswork, and communication degrades into herding that destroys diversity. This is the $g < k$ failure from Section 2.

The herding failure is the benign version of something worse. In the [METR ExploitGym investigation]({% post_url 2026-08-30-METR-Agent-Swarm-Hugging-Face-Incident %}), agents that found a shared channel on their own converged on collective behavior that nobody had designed. team@k is the *engineered* alternative: the only channel is a log of verified evidence, and diversity is enforced by construction.

### A checklist before deploying team@k

1. **Compute horizon.** Does the budget clear the ~400K-token coordination-tax threshold?
2. **Verifier density.** Does the environment give frequent, reliable numerical feedback (unit tests, accuracy, scores) to drive adoption?
3. **Task depth.** Is the problem genuinely multi-stage ($m > 1$), with successive breakthroughs that cannot be decomposed?

If all three hold, the theory says to stop buying more independent samples and let the samples talk to each other. If the verifier is sparse, as in single-shot terminal tasks, the old best@k is still the right tool.

---

## Takeaways

1. **A sum of minima beats a minimum of sums.** A team advances when *anyone* clears a stage. Best@k needs one agent to clear *every* stage. With $m$ stages under a fixed budget, that gap becomes an **exponential** separation in success probability, and it grows with depth.
2. **Adopt only on verified evidence.** An append-only log plus confirm-before-adopt turns parallel search into a relay without a fragile central orchestrator.
3. **Diversity is part of the protocol.** Slot claiming keeps $g \approx k$. Herding silently turns team@k back into roughly best@g.
4. **The wins are large where the conditions hold:** 6.6× sample efficiency on ARC-AGI-3, a new SOTA on polyomino packing, and an MNIST classifier 20.5% smaller than the best human submission. They show up specifically on deep tasks with dense verifiers.
5. **Know the losing regimes.** Below ~400K tokens, or with a single terminal pass/fail signal, independent sampling still wins.
