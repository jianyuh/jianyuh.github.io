---
layout: post
title: "SwarmChat: Slack for AI Agent Swarms, and What 8 to 16 Coding Agents Buy You"
date: 2026-11-02
categories: [LLM, Agents]
tags: [SwarmChat, MultiAgent, AgentSwarm, ProgramBench, ParallelScaling, CodingAgents, Coordination, Harness, GPT-6-Luna, ClaudeHaiku]
---

Reading the following blog post:
- [Introducing SwarmChat: Open-Source Slack for Your AI Agent Swarm](https://swarmchat.io/)

Most multi-agent work asks whether a team of agents can do something a single agent cannot. SwarmChat asks a more practical question: if you give a team of coding agents a Slack-like workspace and let them build real software together, **how much faster do they get there**, and what do they do with the channels?

SwarmChat itself is small: a dependency-free, open-source CLI and server that gives agents public and private channels, direct messages, threads, read cursors, and reactions. The interesting part is the study run on top of it. On ProgramBench (build 10 programs from scratch, from `hex` to SQLite), an 8-agent swarm reaches solo-agent quality **2.9× to 7.0× faster** in wall-clock time. At 16 agents the swarm behaves like a single agent with roughly **11× the token budget**, while spending only about **1.5×** the tokens that single agent would. The capability ceiling, though, does not move: a solo agent given enough hours catches up.

Two other findings stand out. Imposed org charts (a lead agent, feature teams, phased gating) do **not** beat a flat, self-organizing team on average. And the two models tested coordinate in opposite styles: **GPT-6-Luna** teams form a private peer-to-peer mesh (81% DMs), while **Claude Haiku 5.5** teams form a public broadcast hub (83% channel posts).

This is the cooperative counterpart to the [METR ExploitGym investigation]({% post_url 2026-08-30-METR-Agent-Swarm-Hugging-Face-Incident %}), where 1,200 agents built their own message board through a side channel. Here the message board is the product, and the question is what it buys.

![Panel A: effective solo token budget N^lambda for GPT-6-Luna swarms of 1, 2, 4, 8 and 16 agents on ProgramBench, against the ideal lambda = 1 line and the prior lambda band of 0.48 to 0.68; the 8- and 16-agent swarms reach about 5.7x and 10.9x. Panel B: GPT-6-Luna versus Claude Haiku 5.5 teams on share of public-channel messages, messages per agent per hour, median message length, and total messages on the melody task, plus their contrasting testing styles](/assets/images/swarmchat_scaling_phenotypes.svg)

---

## 1. Duration Scaling vs. Swarm Scaling

The post frames the work with a scaling taxonomy formalized by Toby Ord, which separates two ways to spend tokens on a task:

1. **Duration scaling.** One agent works for longer, consuming tokens sequentially: $N = 1$, total tokens $= T$.
2. **Swarm scaling.** $N$ agents work in parallel over the same elapsed time, each with a budget $T$, for roughly $N \times T$ total tokens.

To compare them, define a parallel efficiency $\lambda$ through an equivalence: a swarm of $N$ agents with $T$ tokens each performs like a single agent given $N^\lambda T$ tokens.

$$\text{Performance}(N \text{ agents},\ T \text{ tokens/agent}) \equiv \text{Performance}(1 \text{ agent},\ N^\lambda T \text{ tokens})$$

- $\lambda = 1$ is ideal parallelism: $N$ agents are worth one agent running $N$ times as long.
- $\lambda = 0$ means extra agents add nothing; capability is capped by one agent's budget $T$.
- Prior evaluations of GPT-5.6 Sol on three reasoning benchmarks put $\lambda$ between **0.48 and 0.68**.

This is the same question [team@k]({% post_url 2026-09-23-Test-Time-Multi-Agent-Communication-Discovery %}) asks for discovery problems (does communication turn $k$ independent attempts into something better than best-of-$k$?), measured here in wall-clock time on software projects instead of in solve probability.

### Measured exponents

GPT-6-Luna teams on ProgramBench, scored on 1-hour snapshots:

| Swarm size $N$ | Parallel efficiency $\lambda$ | Implied $N^\lambda$ | 1-hour score | Tokens in that hour |
| :--- | :--- | :--- | :--- | :--- |
| 1 | n/a (baseline) | 1× | ~62% | ~40M |
| 2 | ≈ 0.56 | ≈ 1.5× | 73.4% | ~80M |
| 4 | ≈ 0.43 | ≈ 1.8× | not reported | not reported |
| 8 | ≈ 0.84 | ≈ 5.7× | 80.9% | 306M (vs. 225M for the matching ~6-hour solo run) |
| 16 | ≈ 0.86 | ≈ 10.9× | 82.3% | ≈ 1.5× the matching ~11-hour solo run |

The curve is not monotone: 4 agents are *less* efficient than 2, and then the efficiency jumps at 8 and 16 to well above the prior band. At $\lambda \approx 0.86$, a 16-agent swarm matches a single agent with a token multiplier of

$$N^\lambda = 16^{0.86} \approx 10.9 \approx 11\times$$

and the 8-agent swarm's 80.9% at one hour matches a solo agent at about 6 hours (81.2%), consistent with $8^{0.84} \approx 5.7$. The cost side is the striking part: the 8-agent hour burns 306M tokens against 225M for the six solo hours it replaces, and the 16-agent hour costs about 1.5× the 11 solo hours. **A modest token overhead buys an order-of-magnitude cut in wall-clock time**, without coordination collapsing as the team grows.

---

## 2. System Design

SwarmChat avoids the usual multi-agent orchestration costs: a database layer, polling latency, and stuffing one agent's full context into another's prompt.

```
                   SwarmChat server  (~/.local/share/swarmchat/)
   server.json, users/
   state/                         per-agent read cursors, unread pointers
   chats/<chat_id>/log/000001.jsonl   append-only event log
                 |                                      |
      Agent CLI client                         Observability UI
      swarmchat send / wait                    swarmchat view
      token-free blocking wait                 live message trace, DAG map,
      plain text or --json output              read-pointer audit
```

Five design choices:

1. **Zero dependencies.** Client and server ship as one Python package (`pip install swarmchat`), with no external database or message broker.
2. **Append-only JSONL storage.** Messages, edits, reactions, and read receipts are appended to `.jsonl` files under `chats/<chat_id>/log/`. That gives offline auditability, git-friendly reproducibility, and no schema migrations. It is the same "the filesystem is the interface" bet that [Meta-Harness]({% post_url 2026-04-14-Meta-Harness %}) makes for its optimizer's feedback.
3. **Token-free idle waiting (`swarmchat wait`).** An agent that polls `swarmchat read` pays context and API cost on every empty poll. `wait` blocks in a background process without generating any inference turns until an event arrives. Agents in a swarm spend a lot of time waiting on each other, so this matters; [DSec]({% post_url 2026-09-25-DeepSeek-DSec-Sandbox-Infrastructure-Agentic-RL %}) reports agent sandboxes sitting idle about 90% of the time in RL rollouts.
4. **Agent-shaped CLI output.** Compact text summaries or `--json`, and explicit read markers so an agent ingests only the delta since its last read.
5. **Non-intrusive observability (`swarmchat view`).** A browser view of message topology and read state that does not touch read receipts or any agent's context.

In [harness engineering]({% post_url 2026-07-09-Harness-Engineering-Self-Improvement %}) terms, SwarmChat is a coordination layer added to each agent's harness: the agents still run in Codex CLI or Claude Code, and SwarmChat only supplies the shared message substrate.

---

## 3. ProgramBench: Faster, Not Higher

**Task.** Build 10 complete programs from scratch, from command-line utilities (`hex`, `hwatch`) up to SQLite, `datasurgeon`, and `rust-sloth`. Agents get the spec documentation and a black-box reference binary they can execute for differential testing but cannot inspect. Hidden test suites score a snapshot of the workspace every 15 minutes, which is the [execute-it-against-tests]({% post_url 2026-10-25-Codex-Evaluating-LLMs-Trained-on-Code %}) style of evaluation scaled up from functions to whole programs.

**Agents.** GPT-6-Luna in Codex CLI and Claude Haiku 5.5 in Claude Code, both at `xhigh` reasoning effort.

**Speedups for 8-agent swarms:**

- **GPT-6-Luna:** the swarm scored **88.5% in 6 hours**; a single agent needed **17.5 hours** to reach that, a **2.9×** wall-clock speedup.
- **Claude Haiku 5.5:** the swarm reached the solo agent's ceiling (**93.0%**) in **103 minutes**; the solo agent needed **12 hours**, a **7.0×** speedup.

**Saturation.** Given enough time, a single agent closes the gap: GPT-6-Luna solo reaches 89.5% at 24 hours, against 88.5% for the swarm at 6 hours. Swarms compress days of iteration into hours, but the base model's reasoning sets the asymptote. Parallel agents are a latency lever, not a capability lever.

---

## 4. Does Team Structure Matter?

Four topologies, each with 8 GPT-6-Luna agents over 6-hour runs on all 10 programs:

1. **Flat.** Everyone in `#general`; roles, tasks, and file boundaries emerge on their own.
2. **Lead.** Agent 1 assigns modules, reviews milestones, and writes no feature code.
3. **Features.** A fixed functional split into three domain sub-channels.
4. **Phased.** Temporal gating: 40 minutes of collective spec writing and test definition before any feature implementation.

Pass rates (%):

| Program | Flat | Lead | Features | Phased | Solo (6h) | Solo (24h) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DataSurgeon | 87.8 | 86.7 | 87.3 | 86.5 | 55.0 | 87.5 |
| hex | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| hwatch | 95.5 | 93.6 | 96.5 | 94.2 | 95.7 | 97.2 |
| melody | 80.0 | 75.5 | 89.4 | 85.6 | 81.2 | 88.1 |
| pier | 97.0 | 96.7 | 95.5 | 97.0 | 96.7 | 97.0 |
| rnr | 98.7 | 98.7 | 99.7 | 100.0 | 99.7 | 99.6 |
| rumdl | 77.5 | 76.6 | 76.2 | 60.3 | 57.2 | 59.0 |
| rust-sloth | 77.6 | 76.6 | 77.6 | **94.5** | 77.6 | 97.4 |
| srgn | 96.9 | 96.1 | 95.3 | 94.4 | 94.6 | 96.2 |
| SQLite | **73.6** | 51.6 | 40.9 | 65.1 | 54.6 | 73.5 |
| **Mean** | **88.5** | **85.2** | **85.8** | **87.8** | **81.2** | **89.5** |

What the table says:

- **No structure wins globally.** The four means sit within about 3 points (85.2 to 88.5), roughly experimental noise. Per program, though, the swing reaches 33 points (SQLite: 73.6 Flat vs. 40.9 Features).
- **The lead is a bottleneck.** Lead trails Flat by **3.3 points** on average and collapses on the tightly coupled SQLite (51.6 vs. 73.6). The lead agent spent its time chasing blockers and reassigning work, all over DMs, and on `melody` it stopped communicating entirely at minute 160.
- **Phased gating helps where specs are the hard part.** On `rust-sloth`, a modular 3D parsing and rendering codebase, the Phased team reaches **94.5%**, 16.9 points over Flat, by pinning tests and specs before writing code. The same gating costs 17 points on `rumdl` (60.3 vs. 77.5).
- **Feature silos break on cross-cutting code.** Features falls to 40.9% on SQLite, where parser, engine, and storage depend on each other circularly and a fixed three-way split cuts through those dependencies.

The 6-hour swarms all beat the 6-hour solo agent (81.2) and land near the 24-hour solo agent (89.5), which is the duration-vs-swarm trade from Section 1 again.

---

## 5. Two Communication Phenotypes

The communication logs show that how a team coordinates depends far more on the model than on the org chart. Panel B of the figure summarizes it.

| | GPT-6-Luna: peer-to-peer mesh | Claude Haiku 5.5: broadcast hub |
| :--- | :--- | :--- |
| Routing | 81 to 82% DMs; at most 27% public in every run | 83% public; at least 66% public in 18 of 20 runs |
| Rate | median 44 messages/agent/hour | median 4.2 messages/agent/hour |
| Message length | median ~360 characters | median ~820 characters |
| `melody`, 6 hours | 2,482 messages; all 28 possible agent pairs linked | 168 messages (15× fewer) |
| Testing | private per-agent harnesses, differential checks run locally, bug reports DMed to the module owner | a shared `tests/` harness committed to git in the first minutes, `X/Y passing` posted to `#general` |

GPT-6-Luna decomposes coordination into many small point-to-point negotiations. Haiku batches its coordination into fewer, longer, structured status reports on a single shared board, and makes the test suite itself a shared artifact. Both styles reach high scores; they just need different affordances from the workspace.

---

## 6. Coordination at Scale: git and Good Manners

The usual worry about many agents in one repo is contention: merge conflicts, races, and duplicated work. Going from 2 to 16 agents:

| Metric | 2 agents | 16 agents |
| :--- | :--- | :--- |
| Share of git pushes by the most active agent | 74% | 12% (uniform would be 6.25%) |
| Merge conflicts per agent per hour | 1.5 to 2 | 1.5 to 2 |
| Messages per agent per hour | 22 | 39 (peak 45 at 8 agents) |

1. **Communication grows sub-linearly.** An 8× larger team sends less than 2× as many messages per agent.
2. **Per-agent merge contention is flat.** Team-wide conflict and rebase mentions rise from 4 to 27 per hour, but that is linear in team size; each agent still sees 1.5 to 2 per hour.
3. **git activity decentralizes.** One agent dominates pushes in a pair (74%); in a 16-agent team the top pusher is at 12%, close to uniform.
4. **Agents concede.** There were 115 explicit cases of an agent saying it dropped, reverted, or discarded its uncommitted work after noticing that someone else had pushed or claimed an overlapping component. Teams also negotiated file ownership within the first 60 seconds of a run, without being told to.

---

## 7. Takeaways for Building Agent Workspaces

1. **Make waiting free.** Give agents an event-driven `wait` primitive instead of letting them poll; otherwise idle time turns straight into token cost.
2. **Default to flat teams with local module ownership.** A lead agent is a single point of failure and underperforms unless interface boundaries are truly static. Peer negotiation over a shared channel matches or beats it.
3. **Fit the workspace to the model.** Mesh-style models like GPT-6-Luna need cheap pairwise DMs. Broadcast-style models like Claude Haiku 5.5 need a good shared channel and a shared test suite in the repo.
4. **Use a spec-and-test phase for tightly specified systems.** When the hard part is a strict contract or a complex state machine, gate implementation behind an initial phase of differential spec probing and test authoring, as with `rust-sloth`. Do not apply it blindly: it hurt `rumdl`.
5. **Buy latency, not ceiling.** A swarm is the way to finish in hours what one agent finishes in days, at modest token overhead. It will not solve what the base model cannot.
