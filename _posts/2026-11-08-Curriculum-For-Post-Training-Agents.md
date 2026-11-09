---
layout: post
title: "A Curriculum for Post-Training Agents: SFT, Distillation, RL, Environments, and Harnesses"
date: 2026-11-08
categories: [LLM, Post-training]
tags: [AgenticRL, PostTraining, SFT, Distillation, GRPO, RLEnvironments, OpenEnv, TRL, Harness, Scaffold, AgentGlossary, HuggingFace, TrainingAgents]
---

Reading notes on:
- [A curriculum for post-training agents](https://huggingface.co/blog/huggingface/train-agents)
- [Harness, Scaffold, and the AI Agent Terms Worth Getting Right](https://huggingface.co/blog/agent-glossary)

Hugging Face published a one-page curriculum for **agentic post-training**: the stages after pretraining (supervised fine-tuning, distillation, reinforcement learning) applied to a model that works over many turns, calls tools, reads results and decides what to do next. Training has to account for the whole sequence of turns, not a single answer.

The page follows **Training Agents**, a livestream series on the Hugging Face YouTube channel. Each session covers one post-training method, with training a coding agent as the running example. Each step comes with a diagram, a livestream or slides, a companion blog post and the TRL docs. The page itself is mostly a map. The substance is in the posts it links, so I read all of them and wrote this up as a series:

| Step | Post in this series |
| --- | --- |
| 0. Vocabulary | this post, §2 |
| 1. SFT | this post, §3 |
| 2. Distillation | [Distillation for Agents: From Model Compression to MOPD]({% post_url 2026-11-03-Distillation-For-Agents-Compression-To-MOPD %}) |
| 3. RL | [Training on Outcomes: Verifiable Rewards, the GRPO Family, and a 100-Step Recipe]({% post_url 2026-11-04-Training-On-Outcomes-GRPO-Family-IFStruct %}) |
| 4. RL environments | [RL Environments: Why They Work Now, Plus CARLA and RL over Taste]({% post_url 2026-11-05-RL-Environments-Gym-To-OpenEnv-CARLA-Watercolour %}) |
| 5. Harnesses and scale | [Multi-Harness RL: One Sandbox per Rollout]({% post_url 2026-11-07-Multi-Harness-RL-Sandbox-Per-Rollout %}) and [Keep the Tokens Flowing: 16 Async RL Libraries]({% post_url 2026-11-06-Async-RL-16-Libraries-Keep-Tokens-Flowing %}) |

---

## 1. The Five Steps at a Glance

The curriculum's summary table, lightly expanded:

| Step | The model learns from | TRL trainer | What you need |
| --- | --- | --- | --- |
| 1. SFT | agent traces from a stronger model | `SFTTrainer` | a dataset of agent traces |
| 2. Distillation | a teacher's tokens or probabilities | `DistillationTrainer` | prompts and a teacher model |
| 3. RL | a score for each answer | `GRPOTrainer` | prompts and a reward function |
| 4. RL environments | the outcome of acting in an environment | `GRPOTrainer` | tasks and an environment (OpenEnv) |
| 5. Harnesses and scale | full runs of a real agent | `AsyncGRPOTrainer` | tasks, an environment and an agent harness |

![The five-step curriculum for post-training agents as a ladder. Each step lists what the model learns from, the TRL trainer, and the limitation that motivates the next step. SFT copies a stronger model's traces but never sees its own mistakes. Distillation gives a dense per-token signal on the student's own tokens but is capped by the teacher. RL needs only a check but gets one number per answer. RL environments add multi-turn state and tools, which makes rollout generation the bottleneck. Harnesses and scale train through real agent harnesses in remote sandboxes, with generation and training on separate GPUs](/assets/images/agent_post_training_curriculum.svg)

Reading the table top to bottom, **each step removes a dependency of the step before it and pays for that with a weaker or more expensive signal.**

- **SFT** needs correct traces. The model only ever sees success, so it never learns to recover from its own mistakes.
- **Distillation** trains on the model's own tokens with a dense teacher signal, one value per token. It needs a teacher, and the student is capped by what the teacher knows.
- **RL** replaces the teacher with a check. The signal drops to one number per answer, so the model needs many attempts.
- **RL environments** let the model act several times and see each result. Episodes get long, and **generating rollouts becomes the slowest part of training.**
- **Harnesses and scale** run each rollout as a full agent run through a real harness in a remote sandbox. The model learns in the setting where it will run, and generation and training move to separate GPUs so long episodes do not leave the trainer idle.

Frontier pipelines do not pick one of these. They chain them. The reports surveyed in this series typically run **SFT, then RL experts per domain, then on-policy distillation to merge them, then agentic RL in environments**, and the last stage is where most of the 2026 infrastructure effort goes. (For how these stages interact, see [Interplay of Training Stages]({% post_url 2025-12-14-Pre-Mid-Post-Train %}) and [The Distributional Lens of Post-Training]({% post_url 2026-06-13-Distributional-Lens-Post-Training %}).)

---

## 2. Step 0: Getting the Vocabulary Right

The curriculum opens with a glossary, because the words have drifted. After ICLR 2026, Aritra Roy Gosthipaty asked: "What do you mean by the terms 'harness' and 'scaffold' in the context of agents? I have heard a lot of explanations while I was at ICLR, but I could not understand why they did not converge to a single explanation." The glossary does not claim to settle definitions. It gives a practical mental model, and it is worth adopting because the later posts depend on these distinctions.

![Two panels. A: anatomy of an agent. An agent box (agent = model + harness) contains the scaffold (system prompt, tool descriptions, output parsing, memory across steps), the model (text in, text out, no memory, no loop) and the harness (calls the model, executes tool calls, manages context, decides when to stop); below it, a sub-agent with its own model, scaffold and harness. B: every RL training system for LLMs as a loop: the policy acts in an RL environment, which produces a rollout, which a reward scores, which the trainer uses to update the policy's weights](/assets/images/agent_anatomy_rl_pipeline.svg)

### The deployment terms

| Term | Working definition |
| --- | --- |
| **Model** | The LLM: text in, text out. No memory between calls and no loop. It can express the intent to call a tool, but needs a harness to execute it. |
| **Scaffolding** | The behavior-defining layer around the model: system prompt, tool descriptions, how responses are parsed, what is remembered across steps. It is what the model **works from**. |
| **Harness** | The execution layer: calls the model, handles its tool calls, decides when to stop. It is what makes the agent **run**. |
| **Agent** | From RL, a function from observation to action in a loop with an environment. For LLMs: a model plus everything around it that lets it act, not just respond. |
| **Context engineering** | Designing what goes into the context window at each step, including short-term memory (this run) and long-term memory (stored, retrieved, re-injected). |
| **Policy** | The behavior: the probability of each action in each situation. Part of it lives in the weights, and part depends on scaffolding and harness. A policy is not an agent. |
| **Tool use** | A structured call the harness receives and routes to a function. The result goes back into context. |
| **Skills** | Reusable packages of knowledge for multi-step goals ("investigate this bug, form a hypothesis, write a fix"), loaded on demand. |
| **Sub-agent** | An agent called by another agent, with its own model and scaffold, that can itself reason, use tools and call further sub-agents. The caller is sometimes called an orchestrator. |

The community shorthand is **Agent = Model + Harness**: "if you're not the model, you're the harness". Products like Claude Code use "harness" in this broad sense (its docs say "Claude Code serves as the agentic harness around Claude"). The narrower scaffold/harness split matters when you have to reason about the two separately, and **a training pipeline is exactly that case.** At training time you control the weights and maybe the scaffold, but the loop may belong to someone else's program. That distinction becomes the white-box versus black-box environment split in [step 5]({% post_url 2026-11-07-Multi-Harness-RL-Sandbox-Per-Rollout %}).

Two more points from the glossary:

- **Model, harness, and product are three different things.** Two products on the same model can feel completely different because their harnesses make different choices, and swapping a better model into the same harness also changes the experience. The multi-harness work later measures this: the same weights score 23% in one harness and 52% in another on SWE-bench Pro.
- **Context engineering costs differ by phase.** At inference a bad context is just a prompt change and a redeploy. At training, what the model sees shapes what it learns, and getting it wrong means retraining. (See [Meta-Harness]({% post_url 2026-04-14-Meta-Harness %}) and [Harness Engineering for Self-Improvement]({% post_url 2026-07-09-Harness-Engineering-Self-Improvement %}) for optimizing the inference side automatically.)

An **eval harness** is the same pattern at evaluation time: it runs fixed scenarios at a checkpoint and records metrics instead of updating weights. lm-evaluation-harness is the canonical one.

### The training terms

Every RL system for LLMs is built around the same pipeline: an environment produces rollouts, a reward scores them, and a trainer updates the policy.

| Term | Working definition |
| --- | --- |
| **RL environment** | A stateful object that takes an action (usually a tool call), updates its state, and returns an observation. A filesystem is the simplest example: `touch foo.txt` changes state, and the observation is the new listing. |
| **Trainer** | Runs many episodes, scores them, updates the weights. TRL's `GRPOTrainer` handles generation, scoring and updates in one class. |
| **Rollout** | One full agent run: what it saw, what it did, what reward it got. Also called a trajectory or trace. The raw data RL learns from. |
| **Reward** | Verifiable (tests, exact match) or learned (preferences, LLM-as-judge), sparse (end of episode) or dense (per step). |
| **Rubric** | A reward broken into weighted, explicit dimensions instead of one number. OpenEnv and Verifiers implement rubrics as composable objects (`WeightedSum`, `Sequential`, `Gate`). |

Rubrics show up concretely later. CARLA's port adds `CarlaTrolleyRubric` and `CarlaNavigationRubric`, and the watercolour project's reward is effectively a gated weighted sum of four terms ([step 4]({% post_url 2026-11-05-RL-Environments-Gym-To-OpenEnv-CARLA-Watercolour %})).

---

## 3. Step 1: Supervised Fine-Tuning on Agent Traces

Step 1 has no long companion blog in the curriculum (it links a livestream, a code repo, a thread and the `SFTTrainer` docs), so I summarize it here.

**The model learns to copy agent traces**: recordings of an agent solving a task, with every tool call and its result, usually produced by a stronger model. It is cheap and stable. Its structural weakness is exposure. **The model only sees correct traces, never its own attempts.** When it makes a mistake mid-task it has never seen a recovery from that state, because that state never appears in the data. On-policy distillation and RL fix this by training on the model's own outputs.

Two results from the series sharpen the picture:

- In the multi-harness experiments, SFT on a 27B teacher's successful rollouts (3,189 rollouts from four harnesses) moved a 2.6B model's overall score from 42.2% to only 43.1%. The aggregate hid gains of 4 to 9 points in three harnesses and a 17-point drop in the fourth. RL in the same four harnesses reached 54.6%. Imitation transfers the teacher's harness-specific habits along with its skill.
- SFT on traces still cut tool calls by 24% without any efficiency objective, so imitation does transfer **how** a stronger model works, not only whether it succeeds.

The practical lesson matches [When High SFT Scores Mislead]({% post_url 2026-09-16-SFT-Scores-Mislead-Predicting-Post-RL %}) and [TailSFT]({% post_url 2026-09-28-TailSFT-Filtered-Fine-Tuning-Coverage-for-RL %}): judge SFT by how good a starting point it is for RL, not by its own score.

---

## 4. What the Whole Curriculum Teaches

Reading all eleven linked pieces together, a few themes recur across steps:

1. **Signal density versus dependency.** Per-token teacher signal (distillation) is about 10× cheaper than RL in Qwen3's accounting, but it needs a teacher. Outcome reward needs only a check. Frontier labs use both in sequence: RL to discover a behavior in specialists, distillation to merge it into one model.
2. **Groups with no contrast teach nothing.** Magistral drops zero-advantage groups. The watercolour run turned off per-group reward scaling because one gated failure squashed every other advantage. The multi-harness Qwen runs had 35% to 58% of steps with no reward contrast, and a 0.1 tool-efficiency bonus was the only signal in about a fifth of LFM's groups.
3. **Rewards are exploited exactly as written.** A `+0.2 for submitting anything` term drove held-out accuracy from 0.740 to 0.178 while training reward looked healthy. A broken reward in the RL class made the dashboard look great while coding got worse. The watercolour model ignored a prompt instruction that the reward never paid for.
4. **Infrastructure failure must not look like model failure.** A timed-out render or a dead sandbox returns `None`, not zero. Otherwise the model trains on noise.
5. **The environment is the new dataset.** Environments, task suites and the sandboxes to run them are where frontier labs now spend (reportedly $10M+ per environment at top labs) and what they keep private. OpenEnv, Harbor and TRL are the open stack for each layer.
6. **The harness is part of the policy.** The same weights move by nearly 30 points across harnesses. Training inside several unmodified harnesses, with tokens captured at the model endpoint, spreads gains across all of them.

---

## Takeaways

1. **Five steps, each trading a dependency for a weaker or costlier signal:** traces, then a teacher, then a check, then an environment, then a real harness at scale.
2. **Use the vocabulary precisely.** Scaffold is what the model works from, harness is what runs it, and agent = model + harness. In training, who owns the loop decides what you can record.
3. **SFT is a starting point, not a destination.** It never shows the model its own mistakes, and it carries a teacher's harness habits along with its skill.
4. **The series that follows** goes step by step: [distillation]({% post_url 2026-11-03-Distillation-For-Agents-Compression-To-MOPD %}), [RL]({% post_url 2026-11-04-Training-On-Outcomes-GRPO-Family-IFStruct %}), [environments]({% post_url 2026-11-05-RL-Environments-Gym-To-OpenEnv-CARLA-Watercolour %}), [async RL systems]({% post_url 2026-11-06-Async-RL-16-Libraries-Keep-Tokens-Flowing %}), and [multi-harness RL]({% post_url 2026-11-07-Multi-Harness-RL-Sandbox-Per-Rollout %}).
