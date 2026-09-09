---
layout: post
title: "HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness?"
date: 2026-09-05
categories: [Agents, Evaluation]
tags: [HarnessDev, AgentHarness, SWEBenchPro, TerminalBench, MLEBench, BrowseComp, Overfitting, SelfImprovement, Evaluation]
---

Reading notes on:
- [HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness?](https://arxiv.org/pdf/2609.01437)

Every agent benchmark result you've ever read is a joint measurement of two things: the model, and the software wrapped around it. The **agent harness** — the loop that orchestrates tools, bounds context, recovers from failures, and verifies results — has historically been treated as an experimental control, held fixed so the model can be the variable.

HarnessDev flips the variable. It asks whether an LLM can *build* the harness, and then whether it can *maintain* one over time. The unit of evaluation moves from a transient task output to **runnable, persistent infrastructure**.

The headline result is not that models can do this. It's that they can do it well enough to look impressive under self-evaluation, and then fall apart the moment you check for portability or generalization. This post is mostly about those failure modes, because they are the interesting part.

---

## 1. Formalization

$$(L_C, D) \to H, \qquad (H, L_E, x) \to y \xrightarrow{\;J\;} \text{score}$$

- **$L_C$ (Creator LLM)** — writes, debugs, and refines the harness code.
- **$D$ (Development workspace)** — where the creator edits files, runs tests, and reads traces (Claude Code or Codex in these experiments).
- **$H$ (Runnable harness)** — the persistent, versioned software system produced by $L_C$.
- **$L_E$ (Executor LLM)** — runs downstream tasks *inside the frozen* $H$.
- **$x$ (Downstream task)** — an instance from a benchmark family.
- **$y$ (Task output)** — the authoritative artifact: repo patch, model submission, cited text.
- **$J$ (Judge)** — the frozen benchmark-specific oracle.

The decoupling of $L_C$ from $L_E$ is the whole methodological contribution. Everything interesting falls out of setting $L_E \ne L_C$.

### Anatomy of a harness

$$H = \langle E, T, C, S, L, V \rangle$$

1. **$E$ — Execution loop:** multi-turn control plane, planning phases, scheduling, honest stopping criteria.
2. **$T$ — Tools:** schema registration, interface constraints, I/O parsing, tool-call execution.
3. **$C$ — Context management:** log organization, chat-history assembly, compression/compaction against the context budget.
4. **$S$ — State & memory:** current goals, active hypotheses, progress, attempts, failures, serialization checkpoints.
5. **$L$ — Lifecycle & recovery:** pre/post-action hooks, timeout handlers, process isolation, graceful crash recovery.
6. **$V$ — Verification & evaluation:** automated code checks, verification scripts, self-review, evaluation logs.

Hold onto $S$ and $V$. Those two are where the corpses are buried.

![HarnessDev's two stages: Creation from a policy-free seed, then ten budgeted Evolution commits scored against a disjoint held-out set](/assets/images/harnessdev_two_stages.svg)

---

## 2. Stage 1 — Creation (RQ1)

The creator starts from $H_{\text{seed}}$: a deliberately weak but runnable seed that parses CLI arguments and formats output, and implements **no task-solving orchestration whatsoever**. It scores 0.0 everywhere. From there, using 1–3 visible public development tasks, the model must implement a complete generalized harness.

The frozen result is evaluated two ways:

1. **Self-Eval** — $L_E = L_C$. Tests co-design: a harness tuned to the model's own capabilities.
2. **Unified-Eval** — $L_E = $ Gemini 3.1 Pro for every harness. Isolates *harness quality* from *executor capability*.

Coverage is four domains, five benchmarks, 2,207 unique task instances: **SWE-bench Pro** (731), **Terminal-Bench 2.1** (89), **MLE-bench** (75), **EQ-Bench3** (46), **BrowseComp** (1,266).

| Creator ($L_C$) | SWE-Pro (Self / Unified) | Terminal-Bench | MLE-bench | EQ-Bench3 | BrowseComp | **Average** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Seed harness | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| **Opus 4.8** | **69.3** / 33.0 | 64.8 / 52.4 | **32.9** / 16.9 | **84.6** / 74.2 | 52.4 / 53.6 | **67.8** / 53.3 |
| **GPT-5.5** | 32.8 / 27.8 | 52.1 / 49.4 | 19.1 / 16.0 | 83.0 / 46.5 | **52.6** / **55.4** | 55.1 / 44.8 |
| **Gemini 3.1 Pro** | 43.6 / 43.6 | **68.8** / **68.8** | 32.4 / **32.4** | 74.8 / **74.8** | 35.2 / 35.2 | 55.6 / **55.6** |
| **DeepSeek V4 Pro** | 28.9 / 29.2 | 35.6 / 38.2 | 19.6 / 9.8 | 75.4 / 72.9 | 40.9 / 54.8 | 45.2 / 48.8 |
| **Qwen 3.7 Max** | 33.5 / 41.3 | 41.3 / 48.6 | 3.1 / 16.0 | 68.7 / 71.5 | 32.3 / 49.9 | 44.0 / 52.8 |
| **Seed 2.0 Pro** | 10.8 / 15.6 | 6.0 / 13.1 | 5.3 / 13.3 | 71.1 / 73.1 | 3.2 / 17.3 | 22.8 / 29.8 |
| **Human reference** | **80.0** | **88.8** | **24.0** | **83.7** | **92.2** | **86.2** |

Gemini 3.1 Pro's Self and Unified columns are identical by construction — it *is* the unified executor.

### The co-adaptation plunge

Opus 4.8 is the best system architect under Self-Eval (67.8 average) and collapses under Unified-Eval on SWE-Pro: **69.3 → 33.0**. More than half the score evaporates when you swap the executor.

The mechanism, per the paper's repo analysis: the Opus-built harness bakes in model-specific prompts, strict ReAct protocols, and hardcoded step limits (e.g. 120 steps) calibrated to Opus's own reasoning cadence. A different executor with a different step-per-subgoal rate hits those limits in the wrong places. The harness isn't bad engineering — it's *engineering for one client*, and the benchmark never told anyone there would be a second client.

### The executor bottleneck release

The mirror image is more encouraging. Qwen 3.7 Max goes **44.0 → 52.8** and Seed 2.0 Pro goes **22.8 → 29.8** under Unified-Eval. Their harnesses were structurally sound all along; the scores were throttled by weak task-solving in their own runtimes.

This is a genuinely useful decomposition. "Can this model build agent infrastructure?" and "can this model act as an agent?" turn out to be substantially independent axes, and Self-Eval conflates them into one number that flatters strong executors and penalizes competent architects.

### The efficiency gap

Token cost is decoupled from success. On MLE-bench, GPT-5.5 gets a 19.1 medal rate on **29.3M execution tokens**; DeepSeek V4 Pro gets 19.6 on **208.4M tokens**. Seven times the spend for statistically indistinguishable results. Any evaluation that reports only the score is hiding a 7× variance in the thing you actually pay for — related discussion in [The Economics of a Token]({% post_url 2026-05-17-token-economics %}) and [Metrics and Benchmarks Across LLM Training Stage]({% post_url 2026-06-12-LLM-Evaluation-Architecture %}).

---

## 3. Stage 2 — Evolution (RQ2)

Now the software-maintenance question. Each creator starts from its own frozen RQ1 Code harness ($H_0$) and receives downstream execution feedback — run diagnostics, logs, pass/fail grades — over a fixed feedback set of 100 SWE-Pro and 89 Terminal-Bench tasks. Budget: **10 sequential version updates**, maximizing the feedback-set pair score

$$\bar{P}_t = \frac{1}{2}\left(P^{\text{SWE100}}_t + P^{\text{Term89}}_t\right)$$

After evolution concludes, all versions are scored on a **disjoint, post-freeze 630-task SWE-Pro held-out set** the model never saw. This is the part that makes the study worth reading: the gap between the two columns *is* the finding.

| Creator | Runtime | Feedback set ($H_0 \to H_{\text{dec}}$) | Held-out 630 ($H_0 \to H_{\text{dec}}$) | Final gap (max − declared) |
| :--- | :--- | :---: | :---: | :---: |
| Gemini 3.1 Pro | Self | 59.9 → 68.7 (**+8.8**) | 48.89 → 51.59 (**+2.70**) | 0.00 |
| Opus 4.8 | Self | 71.1 → 74.1 (**+3.0**) | 63.02 → 67.46 (**+4.44**) | 1.59 |
| Qwen 3.7 Max | Self | 41.8 → 55.7 (**+13.9**) | 42.22 → 43.65 (**+1.43**) | 3.17 |
| DeepSeek V4 Pro | Self | 47.2 → 60.6 (**+13.4**) | 47.30 → 50.48 (**+3.17**) | 1.75 |
| GPT-5.5 | Self | 59.2 → 65.1 (**+5.9**) | 48.25 → 52.06 (**+3.81**) | 0.00 |
| Opus 4.8 | Fixed Gemini | 58.8 → 68.6 (**+9.7**) | 48.10 → 50.79 (**+2.70**) | 2.54 |
| Qwen 3.7 Max | Fixed Gemini | 62.1 → 63.2 (**+1.1**) | 49.52 → 48.41 (**−1.11**) | 1.11 |
| DeepSeek V4 Pro | Fixed Gemini | 47.3 → 53.8 (**+6.5**) | 43.02 → 40.63 (**−2.38**) | 3.02 |
| GPT-5.5 | Fixed Gemini | 56.6 → 59.1 (**+2.4**) | 42.22 → **31.90** (**−10.32**) | **16.51** |

### The held-out overfitting trap

Under Self-Runtime every creator pushes its feedback score up. Every one of them sees the gain shrink on unseen tasks — Qwen 3.7 Max's **+13.9 pp** on visible feedback becomes **+1.43 pp** held-out, a **10× attenuation**.

That ratio is the honest measure of how much of the "self-improvement" was real. Roughly 10% of it.

### Fixed-executor generalization collapse

Fix the executor to Gemini 3.1 Pro and evolution turns actively **destructive**. Feedback scores stay flat or rise, while GPT-5.5, DeepSeek V4 Pro, and Qwen 3.7 Max all *regress* held-out. GPT-5.5's −10.32 pp (42.22 → 31.90) is the worst case: the model optimizes visible tasks by injecting hyper-specific, Gemini-unfriendly protocols that break everywhere else.

Note what this implies. The setup where the creator can't tune to its own executor's quirks — which sounds like it should force *more* general engineering — instead produces the worst overfitting, because the model compensates for the executor mismatch with task-specific hacks rather than structural fixes.

### The noise problem

The most damning statistic in the paper. Across **64 official version switches**:

- **2** had clear positive evidence beyond the benchmark noise band ($\pm 4.75$ points).
- **24** regressed on at least one benchmark.
- **27** landed entirely within the noise threshold.
- Feedback and held-out scores moved in the same direction only **53.1%** of the time — a coin flip.

Creators repeatedly "optimized" noise, chased lucky runs, and declared final versions that were rarely held-out optimal. This is not a model-capability failure; it's a **measurement** failure that no amount of model capability fixes. If your signal is a coin flip, a smarter optimizer just overfits it faster. Anyone building the self-improvement loops described in [Harness Engineering for Self-Improvement]({% post_url 2026-07-09-Harness-Engineering-Self-Improvement %}) or [Jeff Dean's YC 2026 talk]({% post_url 2026-08-06-Jeff-Dean-Self-Improving-AI %}) should treat variance control as the first-class problem, ahead of the search strategy.

---

## 4. Engineering Failure Modes

The paper's best contribution is tracing failures through actual execution logs and repository diffs.

### Dead code

LLMs write code fluently and integrate it poorly. Of **108 components** created in the Code domain, **18 showed no evidence of ever triggering** — and all 18 belonged to **State and Memory**. In the Evolution phase, of **169 newly introduced classes or functions**, **56 were entirely dead**: 25 completely unreachable from the entry point, 31 sitting behind unreachable paths.

A third of new code in a maintenance loop never executes. The model writes plausible checkpointing infrastructure and never wires it into the loop, then reports the version as an improvement.

### Syntactic vs. semantic verification

Across **2,325 executed MLE-bench tasks**, **441 produced degenerate, empty, or malformed submissions that the harness self-reported as `success`**. That's a ~19% silent-failure rate on the module whose entire job is catching failures.

And across all **64 version updates, not a single switch modified or added a standalone verifier.** The one component that would most improve the feedback signal is the one component nobody touched — which closes the loop with the noise problem above rather neatly.

### Two evolution styles

**Opus 4.8 — diagnosis-first.** Refused to trust noisy surface feedback. Read `cases.json` and trace logs, classified SWE failures into categories (hidden-grader build failures, import mismatches). Observed that **99% of $H_0$ runs reported success while only 48% actually passed**, traced it to premature completion, and added a mandatory completion gate plus process-group finalize rules (**+304/−18 lines**). Result: the best held-out generalization in the study, **+4.44 pp**.

**GPT-5.5 — architecture-first, local search.** Behaved like an eager editor: added a final review gate (`_final_review_if_needed()`) and an artifact snapshot tracker (`_file_snapshot()`). Meanwhile it left its own **54% tool-call rejection rate** entirely unrepaired, chased marginal feedback gains, and declared $T_2$ as final — **16.51 pp** worse held-out than $T_5$, which it had already produced and discarded.

The contrast isn't about capability. It's that Opus went looking for the *cause* of a discrepancy (99% reported vs. 48% actual) while GPT-5.5 optimized against the *score*. One of those survives distribution shift.

---

## 5. What This Means

The framing claim: **intelligence does not accumulate solely in model weights; a large share of it lives in the runtime infrastructure.** That reframes a lot of agent work — the harness is a co-equal artifact, not scaffolding. See also [Meta-Harness]({% post_url 2026-04-14-Meta-Harness %}) on agentic search over context engineering, and [Claude Code Inflection Point]({% post_url 2026-02-25-Claude %}) on how much of perceived agent capability is harness design.

Three things I'd take into practice:

1. **Stop reporting Self-Eval alone.** Co-designing creator and executor produces a false capability reading — Opus's 69.3 → 33.0 is not a rounding error. Portability testing needs to be standard, the same way held-out sets are standard everywhere else.

2. **State and memory is an open frontier, and models don't know it.** Checkpointing and rollback mechanisms were almost entirely unobserved across 26,000+ recorded trajectories — and *every* dead component in the Code domain was a State/Memory one. Models write this code because it looks like what a harness should have, then never invoke it. That's a collective blind spot about historical state and recovery, and it lines up with what's known about [LLM agent memory]({% post_url 2025-12-27-Agent-Memory %}) generally.

3. **Noisy feedback kills evolution.** With a $\pm 4.75$ noise band and 53.1% directional agreement, an automated developer will overfit before it improves. Until we decouple genuine code improvement from single-run benchmark success, evolution loops decay on unseen distributions. The fix is boring — more seeds, verifier-driven signals, wider noise bands as gates — and nobody in the study did it.

The uncomfortable synthesis: the models most capable of building a good harness (Opus) are also the ones most capable of building a harness that only works for themselves, and the evaluation regime everyone currently uses can't tell those two things apart. Related failure-mode analyses: [When 1,200 Agents Formed a Collective]({% post_url 2026-08-30-METR-Agent-Swarm-Hugging-Face-Incident %}) and [Self-Play SWE-RL]({% post_url 2025-12-26-Self-Play-SWE-RL %}).
