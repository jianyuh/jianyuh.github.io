---
layout: post
title: "Multi-Harness RL: One Sandbox per Rollout, and Training Inside Claude Code, Codex and OpenCode"
date: 2026-11-07
categories: [LLM, RL]
tags: [AgenticRL, Harness, MultiHarness, OpenEnv, Harbor, TRL, AsyncGRPO, CaptureProxy, Sandbox, ClaudeCode, Codex, OpenCode, MiniSWEAgent, LFM2.5, SFTvsRL, TokenFaithful, TrainingAgents]
---

Reading notes on:
- [One sandbox per rollout, or how labs run RL for agents in 2026](https://huggingface.co/blog/sergiopaniego/rl-environments-2026)
- [The ultimate guide to multi-harness RL](https://huggingface.co/spaces/FineEnvs/multi-harness-rl)

This is step 5, the last step, of the Hugging Face **Training Agents** curriculum (hub: [A Curriculum for Post-Training Agents]({% post_url 2026-11-08-Curriculum-For-Post-Training-Agents %})). In [step 4]({% post_url 2026-11-05-RL-Environments-Gym-To-OpenEnv-CARLA-Watercolour %}) the trainer owned the loop and called `env.step()`. Here **each rollout is a full run of an existing agent harness inside a remote sandbox.** Training through the same harness the agent uses in practice means the model learns in the setting where it will run. The systems half of the step, async GRPO, is covered separately in [Keep the Tokens Flowing]({% post_url 2026-11-06-Async-RL-16-Libraries-Keep-Tokens-Flowing %}).

The first post is a census of what 15 frontier reports say about where agent RL happens. The second is the most substantive piece in the whole curriculum: an open framework (OpenEnv + Harbor + TRL) that trains a 2.6B model inside unmodified Claude Code, Codex, OpenCode and Mini-SWE-Agent, with careful ablations.

---

## Part I: One Sandbox per Rollout

### "Environment" means four different things

The census covers fifteen reports and release posts from thirteen labs, published October 2025 to September 2026: GLM-5 and GLM-5.3, Kimi K3, MiniMax, Nemotron 3 Ultra, OLMo 3, DeepSeek, Qwen3.8, Inkling, LFM2.5, MAI-Thinking-1, GPT-5.6 and GPT-6 Astra, plus Cursor's Composer and Meta's Code World Model. It counts only what a lab says it **trains** in, not what it evaluates on. Zhipu's [GLM-5.3]({% post_url 2026-08-16-GLM-5.3-Post-Training-Scaling-IndexShare-SAO %}) blog states the premise: "As agent capability improves, much of the difficulty in scaling post-training moves from the model to the environment".

The reports use "environment" at four layers: the **task and its verifier**, the **contract** the agent acts through, the **sandboxed machine** underneath, and the **whole rollout system**. Keeping them apart is most of the battle.

### A machine per attempt

Classic RL environments were simulators living in the trainer's memory. `reset` and `step` were nearly free, and thousands of CartPoles fit on one GPU. An agent needs a real computer: a filesystem, a shell, and processes that survive between calls. So the environment became **a machine whose state belongs to one attempt, booted, used once, and destroyed.**

- **LFM2.5** (2.6B): "During training, we sample a task and randomly select a corresponding harness. Each rollout runs in a dedicated sandbox with its own runtime." The RL framework underneath is verl.
- **Cursor Composer**: "running hundreds of thousands of concurrent sandboxed coding environments in the cloud". Cursor rewrote its VM scheduler for the spikes of training runs.
- **MAI-Thinking-1**: the sandbox layer "provisions a fresh container for each agentic task and destroys it upon task completion".
- **[Kimi K3]({% post_url 2026-07-18-Kimi-K3 %})**: "million-token agentic RL with persistent rollout and sandbox states", using "resumable microVM sandboxes to preserve long-lived model and environment state". At a million tokens on one task, the system has to checkpoint and resume worlds, not just models.

So the unit is a machine. It is thrown away after a short attempt, and checkpointed and resumed through a long one. At the high end there are hundreds of thousands at once, and even a 2.6B model gets the same one sandbox per rollout.

### What labs put inside them

There are three recurring kinds, with coding in the lead:

- **Coding.** GLM-5: "over 10k verifiable environments across thousands of repositories spanning 9 programming languages". MiniMax converts "real-world programming scenarios into a diverse corpus of verifiable terminal tasks". Cursor and Meta describe coding and nothing else.
- **Search.** Kimi K3 trains on "multi-step complex information searching", where the model plans, gathers evidence, and produces a verifiable answer.
- **Office work.** MiniMax covers "deep search and open-web research, knowledge-worker office tasks, financial analysis and spreadsheet operations, and slide generation". Kimi ships "realistic mock implementations of widely used applications, such as Gmail, Notion, Slack, and Canvas". Kimi alone adds GPU kernel tasks "ranging from single-operator kernels to fused mega-kernels".

**The harness itself is becoming the environment.** Kimi K3's environment "can instantiate mainstream harnesses such as Kimi Code, Claude Code, Codex, OpenClaw, and Hermes", rebuilt from modules (tool interfaces, system prompts) as a **white-box** environment. Liquid gets there from the other side, with a proxy that "lets us treat agentic harnesses as black boxes with no modification, while transparently capturing the token-level trajectories needed to reconstruct and validate RL training samples". Microsoft's Agent Lightning calls this *harnessed agentic RL*, where "the harness, rather than the training engine, owns the environment interaction loop". NVIDIA's Polar does the same at scale. Thinking Machines' [Inkling]({% post_url 2026-07-17-Inkling-975B-MoE %}) randomized the tool set and schema during training to make it less sensitive to any one harness.

The newest pattern is **environments built by agents**. GLM-5.3 has research agents turn real work into runnable long-horizon environments. A judge agent verifies that each task is solvable, and a verifier ships only after passing "oracle, no-op, and unsolved-state checks". Zhipu admits this "still require[s] a meaningful amount of human-in-the-loop work".

Openness varies a lot. The open-weights labs (Kimi, GLM, MiniMax, NVIDIA) describe environments in detail, and OLMo 3 documents grading down to "17.2 million generated code samples". OpenAI's GPT-5.6 system card names plenty of environments, but every one is an evaluation. DeepSeek documents its sandbox platform ("hundreds of thousands of sandboxes per cluster"; my notes on it are in [DeepSeek DSec]({% post_url 2026-09-25-DeepSeek-DSec-Sandbox-Infrastructure-Agentic-RL %})) without naming what runs inside.

### Every lab built its own stack, and the stacks have the same shape

Labs converge on recipes for distillation and RL algorithms, but each built its own environment infrastructure:

- **GLM-5 / slime:** an asynchronous design that "drastically improves post-training efficiency by decoupling generation from training". With a synchronous loop, the trainer waits for the slowest container in the batch. Decoupling trades that idle time for staleness. GLM-5.3 shows the payoff: environments plug in "as data generation rather than as changes to the training loop", which let Zhipu keep adding environments through GLM-5.2 and 5.3 without rebuilding the stack. Jie Tang frames GLM-5.3 as an ablation: "same base, same architecture, same total and activated parameters as GLM-5.2. One month of scaling long-horizon environments and RL. The gains are not marginal."
- **MiniMax Forge** "seamlessly admits both white-box and black-box (API-only) agents within a unified training loop".
- **Nemotron 3 Ultra:** "multi-environment RLVR".
- **Inkling:** large-scale RL on synthetic and human-made environments, over 30 million rollouts.
- **Meta's Code World Model:** an execution service running "tens of thousands of code snippets per second" in isolated containers.

Under the names, every stack has four pieces: **a task with a verifier, a contract the agent acts through (increasingly the harness itself), a sandbox, and a trainer that does not wait for the slowest rollout.** SemiAnalysis's conclusion matches: "RL training is as much of an infrastructure problem as an algorithm one". Nathan Lambert adds that environments are the easiest part of frontier training to hide, reports "$10M+ for single environments" at top labs, and sums up GLM-5.3 with: "one does not simply 'distill' RL environments, infrastructure to run them at scale, or algorithms to mix them together effectively".

### The open version of each layer

- **Tasks:** Prime Intellect's Environments Hub (on verifiers), **Harbor** (decouples task, harness and sandbox), and Repo2RLEnv.
- **Environments:** **OpenEnv** standardizes the Gymnasium-style contract (reset, step, state) over HTTP. Its technical committee now includes commercial vendors (Prime Intellect, Mercor, Fleet AI) next to Meta-PyTorch, Hugging Face, Nvidia and Microsoft, and **over 4,000 Hub Spaces** carry the openenv tag. SkyRL, BrowserGym and TextArena serve whole task families.
- **Sandboxes:** Modal and E2B sell this layer. Moonshot open-sourced **AgentENV**, the sandbox layer under Kimi K3, with fast snapshot, resume and fork. **Hugging Face Sandboxes** (experimental) packs many CPU sandboxes per host with SandboxPool.
- **Trainers:** TRL's `GRPOTrainer` (white box, one argument to connect an environment) and `AsyncGRPOTrainer` (decoupled, the slime-style design). RadixArk's **Miles** v0.1, a slime fork with OpenEnv, Harbor and verifiers connectors, says it was battle-tested "on frontier open models like Kimi K3, DeepSeek V4, Qwen 3.8, GLM 5.2, Inkling, MiniMax H3".

What labs keep private is the environments, the task data, and the compute for hundreds of thousands of sandboxes.

---

## Part II: Training Inside the Harnesses People Actually Use

### The problem: the harness moves the score more than the model does

A harness is "the software layer between the model and the task that constructs the context the model sees, mediates its tool calls, validates its outputs, and decides when to retry, escalate, or stop". The evidence that it matters:

- On SWE-bench Pro, **GLM-5.2 scored 23% in one harness and 52% in another.** Codex was the 2nd-best of ten harnesses for GLM-5.2 and the 9th for Gemma 4 26B-A4B.
- A controlled study ran GLM-5.1, GPT-5.4 and Kimi K2.6 (within three points of each other on a leaderboard) on the same 100 SWE-bench Verified tasks under three nested harness configurations. **Each model won under a different harness.** Swapping the harness moved GLM-5.1 by 13 points, while swapping the model inside one harness moved the score by only 2.5 to 5.
- Claude Opus 4.5 scores 45.9% on SWE-bench Pro on Scale's SEAL leaderboard and 55.4% inside Claude Code, with the same weights.
- Model cards now attach a harness to each score. Kimi K2 reports Terminal-Bench at 25.0 under Terminus and 30.0 under its own framework, and DeepSeek-V3.2's thinking mode would not run under Terminus at all.

![Dumbbell chart of the same weights under two harnesses: GLM-5.2 on SWE-bench Pro 23% versus 52%; Claude Opus 4.5 on SWE-bench Pro 45.9% on the SEAL leaderboard versus 55.4% in Claude Code; base LFM2.5-2.6B on SmolDataEnvs 33% in Claude Code versus 62% in Mini-SWE-Agent; OpenSWE-32B on SWE-bench Verified 3.6% in Kimi-CLI versus about 62.4% in its home harness OpenHands](/assets/images/harness_sensitivity.svg)

**Leaving the home harness hurts.** In the Orchard paper, OpenSWE-32B lost 7.5 points moving from OpenHands to Mini-SWE-Agent, and under Kimi-CLI it **dropped 58.8 points to 3.6%** on SWE-bench Verified and scored zero on Terminal-Bench 2.0. Scale-SWE stopped producing valid tool calls outside its home harness. Orchard's own model (SFT on two harnesses, RL in one) stayed between 45.0 and 64.3 across three. The paper names two failure modes: a **degraded resolve rate** and a **catastrophic format failure**.

The KwaiKAT team names the cause: with a single fixed harness the model learns "not 'how to solve the task' but 'how to solve the task under that particular harness's interface conventions'". It overfits in three ways: to the action **format**, to the **context structure**, and to the **control flow** (retries, stop conditions). Harnesses differ along exactly these lines. There are four API dialects (OpenAI chat-completions, OpenAI Responses, Anthropic Messages, Gemini), plus differences in tool calling, compaction and control. Aider asks for prose edit blocks instead of tool calls. In practice the failure often shows up as a rejected tool call (wrong tool name or argument names), not a lower score.

**Frontier labs now vary the harness on purpose.** Poolside's Laguna adds 1.3B tokens of OpenHands, OpenCode and Mini-SWE-Agent trajectories to SFT but runs RL only in its production harness. Kimi K3 composes harnesses from modules. Qwen3-Coder-Next generates data in six harnesses, and [MiMo-V2.6]({% post_url 2026-09-24-MiMo-V2.6-Scaling-RL-Self-Improvement %}) trains on a pool of mini-harnesses. LFM2.5 picks a random harness per task. OpenForgeRL found that a three-harness model beat a single-harness model **even on the single harness's home turf (48.5 vs 46.0)**, and nearly doubled its score under Codex. Agent Lightning took Qwen3.5-9B from 41.8% to 56.4% on SWE-bench Verified with about 6,000 examples. The missing piece was an open way to do this with **any** harness. For harness-side optimization without training, see [Meta-Harness]({% post_url 2026-04-14-Meta-Harness %}) and [HarnessDev]({% post_url 2026-09-05-HarnessDev-LLMs-Building-Their-Own-Harness %}).

### White box versus black box, and why rewards are not enough

- **White box:** the trainer owns the loop. It samples, calls `env.step()`, and holds every token it sampled. This is what `GRPOTrainer` with an `environment_factory` expects.
- **Black box:** the harness owns the loop. It runs in a sandbox, calls its own tools, compacts its own context, and stops when it decides to. The trainer sees only a stream of model API calls. The model API is the one interface guaranteed to exist outside every harness.

An on-policy policy gradient needs two things per token: **which token was sampled, and the probability it was sampled with.** A harness that returns text and a score provides neither:

1. **Retokenization drift.** Different token sequences can decode to the same text, so re-tokenizing saved text can produce IDs the model never sampled. "In RL, you optimize on the exact tokens the model produced."
2. **Harness edits.** Harnesses add role markers, change whitespace, or repair malformed JSON. Training on edited text updates the model on responses it never generated.
3. **Recomputed probabilities are wrong under async.** If the policy has moved since sampling, recomputed logprobs are not the behavior logprobs. Using the current value on both sides of the IS ratio makes it 1 even when the policies differ.

So the sampler must return token IDs and per-token logprobs, and the trainer must keep them. **Never re-encode decoded text.** Append template suffixes by ID concatenation, which works because most chat templates extend token for token (18 of the 19 models tested in TRL's write-up). Prime Intellect's renderers library solves this from the trainer side, but it needs the trainer to own the loop. The authors note honestly that OpenForgeRL builds the same proxy from prompt/response pairs without token IDs or logprobs and still reports the strongest multi-harness results. This guide takes the token-faithful side, and the question is open.

### The framework: OpenEnv capture proxy + Harbor + TRL

![Multi-harness RL framework. A task from Harbor (instruction, environment, test script) runs in one E2B or Daytona sandbox per rollout, where an unmodified harness (Claude Code, Codex, OpenCode, Mini-SWE-Agent) runs its own loop. Every model call goes to the OpenEnv capture proxy, keyed by a per-rollout session id; the proxy detects the API dialect, converts it to chat completions, calls vLLM without streaming and with top_p 1.0, and records prompt token ids, sampled token ids and per-token logprobs. Calls are linked into a rollout graph by longest exact token prefix, each root-to-leaf path becomes a training sequence with loss mask 1 on sampled tokens, checked against Harbor's ATIF trajectory. The verifier reward, or None if it never ran, and the TrainingTrace go to TRL AsyncGRPOTrainer on one H100, which syncs weights to vLLM on a second H100](/assets/images/multi_harness_capture.svg)

**OpenEnv** is the shared interface: Gymnasium-style `reset`, `step` and `state` over a client/server transport, typed, packaged as Docker, publishable to the Hub, under BSD-3-Clause and steered by twelve organizations. It trains nothing and defines no rewards. Two pieces matter here: the **capture proxy** and the **Harbor integration**.

**The capture proxy.** A harness finds it like any model provider, through a base URL and an API key. The key is a **capture session ID minted per rollout**, so one proxy on one port serves every concurrent rollout, and unknown keys get a 401. The proxy detects the dialect (path, then headers, then body shape), converts to chat completions with converters vendored from Polar, and replays the answer in the caller's format. **It never streams from the engine.** It waits for the full completion and replays it as a stream if asked, so it never parses partial deltas.

**What gets recorded.** Prompt token IDs, sampled token IDs and a logprob per sampled token, sampled from the **full distribution with no top-p or top-k**. Truncation biases sampling away from the policy, a known route to entropy collapse, and vLLM's processed logprobs are computed after truncation, so they would not match. Moving `top_p` to 1.0 took the importance ratio from **0.985 to 0.993** up to **0.9984 to 0.9999**. On vLLM this needs `--return-tokens-as-token-ids --logprobs-mode processed_logprobs`. The proxy probes each engine and grades it. Hosted APIs (OpenAI, Anthropic, Inference Providers) land below the `tokens` level, so they can evaluate a harness but not train through it.

**From calls to training sequences.** Harnesses retry, spawn subagents and compact context, so a rollout is rarely one clean conversation. Each call becomes a node linked to the earlier call whose prompt+completion is the **longest exact token prefix** of its prompt. A retry is a sibling that never continued. A subagent or a compaction starts a new root. Each root-to-leaf path becomes one sequence, with loss mask 0 on context and 1 on sampled tokens. Turns with missing or misaligned logprobs stay as context and are never targets. Any program that calls one of the four APIs can be captured (`POST /sessions`, point the agent at the proxy, `GET /sessions/{id}/rollout`). The proxy is one process: its health check starved at about 200 concurrent sessions, and it crashed at 320.

**Harbor** keeps task, harness and sandbox independent: 40+ harness adapters, 26 execution backends (Docker, Daytona, Modal, E2B), and about 80 task datasets (Terminal-Bench, SWE-bench). A task is a directory with an instruction, an environment and a test script, and it names no harness, backend or trainer. The task suite here is **SmolDataEnvs**, data-analysis questions from Kaggle notebooks, each with an automatic grader. Ten harnesses passed the capture contract across 16,000 rollouts.

Engineering lessons that generalize:

- **Nothing raises across the trainer boundary.** An earlier in-process version let an exception from someone else's agent crash one rank, and the other ranks hung forever at the NCCL barrier. Now a failed rollout returns `ok=False, reward=None`.
- **Never combine verifier scores silently.** An earlier run had a `+0.2 for submitting anything` term. The policy learned to submit immediately, and **held-out accuracy fell from 0.740 to 0.178 while training reward looked healthy.**
- **A dead sandbox is not a wrong answer.** If the verifier never runs, the reward stays `None` all the way to the trainer.
- **Cross-check the capture independently.** Each rollout is compared call by call with Harbor's own ATIF trajectory (turn count, completion tokens per call, agent steps). This caught a harness that sent an empty tools array, got a 400, and was cut short while leaving a well-formed-looking graph.
- **Raise `max_concurrent_envs`.** The default of four is less than one GRPO group of eight.

On the TRL side, `HarborSessionFactory` produces sessions for `HarnessRolloutWorker` (TRL #6947, `trl.experimental`), which returns a typed `TrainingTrace` (token IDs, behavior logprobs, loss masks) to `AsyncGRPOTrainer`. Every rollout in a GRPO group uses the **same harness**, so the advantage compares actions, not harnesses.

### Experiments: LFM2.5-2.6B across four harnesses

**Setup.** 1,000 SmolDataEnvs training tasks (400 medium, 600 hard) in a fixed order, and 250 held-out test tasks with no overlap. Two runs: **OpenCode-only**, and **multi-harness** (OpenCode, Claude Code, Codex, Mini-SWE-Agent, one harness per GRPO group of eight). Every 100 steps, each test task runs once under each harness: 1,000 cells, solved if the first graded attempt is correct. Async GRPO for 1,000 steps, with one H100 training and one serving vLLM. Rollouts are usually 2 optimizer steps old and never more than 4. One E2B sandbox per rollout (1 CPU, 4 GB). Training took about 32 hours (OpenCode-only) and 46 hours (multi-harness).

**The baseline already shows the problem.** The same LFM weights solve **62% under Mini-SWE-Agent and 33% under Claude Code.**

**Reward.** Correctness (1 or 0) plus a **tool-efficiency bonus of at most 0.1**, paid only on correct answers, so the model cannot earn it by quitting early. It came from the earlier Qwen runs, where correctness alone gave no reason to stop exploring and tool calls crept from 13 to 41. Because GRPO learns from differences within a group, a group where all eight are correct teaches nothing under correctness alone. The bonus is then the only contrast, and it favors the shortest solutions. In **97 of 555 OpenCode-only groups (17.5%)** and **155 of 686 multi-harness groups (22.6%)**, every scorable rollout was correct and the bonus supplied the only signal.

**Results.**

| | OpenCode-only RL | Multi-harness RL |
| --- | --- | --- |
| Overall accuracy at step 1,000 | 52.3% (+10) | 54.2% (+12) |
| Under OpenCode | **58%** (from 34%) | 50% |
| Under Claude Code | 42% | **49%** |
| Under Codex | 43% | **54%** |
| Under Mini-SWE-Agent | tie | tie |
| Fewer tool calls vs base (tasks both solve) | 11% | **31%** |

- **A model gets better where it trains.** OpenCode-only gained mostly in OpenCode (34 to 58%). Multi-harness gained under all four.
- **The overall 1.9-point gap is within noise.** The difference is **where** the gains land.
- **Efficiency transfers unevenly.** The multi-harness model uses about half the calls under Codex. The OpenCode-only model used **more** calls and tokens than the base model under Claude Code, up to 62% more tokens at step 500.
- **Steps are a poor unit of comparison.** Over the same 1,000 steps, multi-harness saw 626 distinct tasks versus 555, 18.7M versus 12.5M supervised tokens, and **451M versus 162M tokens processed**. Claude Code rewrites its own history, so each of its rollouts became about eight training rows. In the earlier Qwen runs it produced 77% of training rows from about a quarter of the rollouts.

### SFT versus RL on the same harnesses

The capture also records whole trajectories, so a larger model's successes can be imitated. A Qwen3.8-27B teacher ran the training tasks in all four harnesses (up to three attempts each, first correct kept), giving 3,189 successful rollouts from 888 tasks. Each assistant turn is one example, with loss on the next response only, tokenized with LFM's template. Both SFT runs are full fine-tunes for two epochs at LR 3e-6, effective batch size 8.

| | OpenCode SFT | Multi-harness SFT | Best multi-harness RL |
| --- | --- | --- | --- |
| Training data | 801 rollouts, 4,825 examples | 3,189 rollouts, 17,929 examples | own rollouts |
| Overall | 47.5% | 43.1% (base 42.2%) | **54.6%** (step 700) |
| OpenCode | 51.6% (from 33.6%) | 38.0% | not reported |
| Claude Code | 33.2% (no change) | 42.4% (from 33.2%) | not reported |
| Codex | not reported | 46.8% (from 40.0%) | not reported |
| Mini-SWE-Agent | 58.8% (from 62.1%) | **45.2%** (from 62.1%) | improved |
| Fewer tool calls (tasks both solve) | 8.5% | 24.2% | 26.7% (step 700), 31.1% (step 1,000) |
| Training time | 1.5 h | 6.5 h | 46 h |

(The guide notes that choosing RL's best checkpoint on the test set flatters RL slightly. Both RL runs are within half a point of their step-1,000 scores.)

![LFM2.5-2.6B results. Panel A, overall accuracy: base 42.2%, multi-harness SFT 43.1%, OpenCode SFT 47.5%, OpenCode-only RL 52.3%, multi-harness RL 54.2% at step 1,000. Panel B, fewer tool calls than the base model on tasks both solved: OpenCode SFT 8.5%, OpenCode-only RL 11%, multi-harness SFT 24.2%, multi-harness RL 31.1%](/assets/images/lfm_sft_vs_rl.svg)

- **RL ends ahead of both SFT runs**, by 11.5 points over SFT on the same four harnesses.
- **OpenCode SFT looks like a smaller OpenCode RL.** Nearly all its gain is in OpenCode.
- **Multi-harness SFT spreads gains and then loses them in Mini-SWE-Agent** (62.1 to 45.2%), so overall it ends within noise of the base. After epoch 1 it was **below** base (38.3%, with OpenCode at 22.4%). The cause is not yet understood.
- **SFT cuts tool calls without being asked**, 24.2% fewer for multi-harness SFT under every harness.
- **The caveat:** the SFT runs differ in supervised tokens (about 7×) and steps (about 4×), and SFT and RL differ in objective, data and compute, each run once. These are observations, not a ranking. The runs they want next are SFT then RL versus RL from base, and token-balanced harness mixes. This lines up with [When High SFT Scores Mislead]({% post_url 2026-09-16-SFT-Scores-Mislead-Predicting-Post-RL %}) and [TailSFT]({% post_url 2026-09-28-TailSFT-Filtered-Fine-Tuning-Coverage-for-RL %}).

### The Qwen runs: exact tokens are only the start

The first runs used Qwen3.5-2B with a correctness-only reward on an easier pool. All three improved at first, which proved that an unmodified harness can be trained through once you have exact tokens. Multi-harness went from **14.6% to 37.0%** at step 500, and OpenCode-only peaked at **39.5%** at step 700. Then both **fell back to about 26%** by step 1,000:

- **Multi-harness ran out of output budget.** Training allowed 16,384 output tokens but evaluation cut at 4,096. Truncated cells rose from **9 to 556** of 1,000, and under OpenCode calls per task collapsed from 16 to 3.
- **OpenCode-only kept working without finishing.** Calls per task rose from 17 to 21, and the share that submitted fell from **69% to 41%**.
- **Many steps taught nothing.** Between **35% and 58%** of optimizer steps had no reward contrast in any group.
- A resume bug replayed old tasks after step 684, but the decline started at 500.

The LFM setup changed three things: the efficiency bonus (signal in all-correct groups, a cost for endless loops), medium and hard tasks only, and **the same 4,096-token cap in training and evaluation**. The LFM runs did not decline, though the model and task pool also changed.

---

## Takeaways

1. **The unit of agent RL is a disposable (or resumable) machine per rollout.** Every lab's stack has four layers: task+verifier, contract (increasingly the harness), sandbox, and a non-blocking trainer.
2. **The harness moves scores more than the model does.** GLM-5.2 scores 23% versus 52% across harnesses, and a model trained in one harness can fail catastrophically in another.
3. **Training across harnesses spreads gains, while single-harness training concentrates them.** For LFM2.5-2.6B, overall accuracy was similar, but Claude Code and Codex improved only with multi-harness training.
4. **Capture tokens at the model endpoint.** Keep sampled IDs and behavior logprobs, sample untruncated, and never re-tokenize text. (Whether this is strictly necessary is still debated.)
5. **Reward design decides when the agent stops.** A capped efficiency bonus gives signal in all-correct groups. A `+0.2 for submitting` term destroyed accuracy while the reward looked healthy.
6. **Match output limits between training and evaluation**, and compare runs by tokens processed, not steps.
