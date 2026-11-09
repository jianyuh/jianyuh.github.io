---
layout: post
title: "Distillation for Agents: From Model Compression to Multi-Teacher On-Policy Distillation"
date: 2026-11-03
categories: [LLM, Post-training, Distillation]
tags: [Distillation, OnPolicyDistillation, MOPD, SelfDistillation, KnowledgeDistillation, TRL, GKD, DeepSeekV4, MiMo, GLM5, Nemotron, Qwen3, Composer, TrainingAgents]
---

Reading notes on:
- [A brief history of distillation in AI](https://huggingface.co/blog/sergiopaniego/brief-history-of-distillation-in-ai)
- [Distillation in 2026 (so far): which frontier models use it and how](https://huggingface.co/blog/sergiopaniego/distillation-2026)

This is step 2 of the Hugging Face **Training Agents** curriculum, which I cover as a series. The hub post is [A Curriculum for Post-Training Agents]({% post_url 2026-11-08-Curriculum-For-Post-Training-Agents %}). Step 1, SFT, trains the model on traces from a stronger model. The student only ever sees correct traces, so when it makes a mistake of its own it has never learned how to recover. Distillation is the first step that can fix this. A teacher supervises the student, either through text the teacher writes or through the teacher's probabilities on the **student's own tokens**.

The curriculum's one-paragraph summary of the trade-off:

- **Upside:** the signal is dense, one value per token, so training needs far fewer samples than RL.
- **Downside:** the student is capped by what the teacher knows, and on-policy distillation has to run the teacher at every training step.
- **What comes next:** RL replaces the teacher with a reward that only checks the result ([step 3]({% post_url 2026-11-04-Training-On-Outcomes-GRPO-Family-IFStruct %})).

The two companion posts tell this story twice. Once as history (compression to transfer to frontier post-training), and once as a census of how 2026 frontier reports actually use distillation.

---

## 1. A Short History: Same Mechanic, Bigger Goals

![Three eras of distillation. Compression: 2006 Model Compression, 2015 Hinton, Vinyals and Dean's soft targets (dark knowledge), 2019 DistilBERT at 40% less size. Transfer: 2023 Alpaca and Self-Instruct, 2025 DeepSeek-R1-Distill. Frontier post-training: 2023-24 on-policy distillation (MiniLLM, GKD), now used by GLM-5.2 and Qwen3; 2026 MOPD merging same-size RL experts; 2026 self-distillation with hint-conditioned or earlier-checkpoint teachers](/assets/images/distillation_history.svg)

### Compression (2006 to about 2022)

- **Model Compression** (Bucilă, Caruana, Niculescu-Mizil, 2006): train a small model to reproduce a big, slow model's predictions. The behavior stays the same at a fraction of the cost. It adds no new capability. (Schmidhuber, predictably, claims he did it first in 1991.)
- **Distilling the Knowledge in a Neural Network** (Hinton, Vinyals, Dean, 2015) made the idea click with the teacher-student setup. The student learns the teacher's full output distribution softened by a temperature, not just the single hard label. The extra signal, which other options were plausible and how close they were, is the "dark knowledge". It is why a student can learn more from a teacher than from the raw data.
- It then spread through vision (FitNets, attention transfer) and NLP. **DistilBERT** (2019) kept most of BERT's quality at 40% less size.

For almost a decade, distillation meant one thing: make an existing model smaller and faster.

### Transfer (2023 to 2025)

Around 2023 the teacher became an LLM, and the goal changed from compressing a model you already had to **copying behavior the student did not have**. Alpaca and Self-Instruct pulled capability out of a bigger model's text for a few hundred dollars. The most visible case is **DeepSeek-R1-Distill** (2025). Reasoning traces from R1 were distilled into compact Qwen and Llama students and shipped openly. A small model plus a good teacher, and suddenly it can reason.

### Frontier post-training (2025 onward)

At the frontier the current form is **on-policy distillation**. The student generates, and a stronger teacher scores the student's own tokens, so the student learns to fix the mistakes it actually makes. GLM-5.2 and Qwen3 both lean on it. TRL implements the on-policy form as GKD (Agarwal et al., 2024), with a reverse-KL variant in MiniLLM (Gu et al., 2023). I went through the failure modes of this objective in [Revisiting On-Policy Distillation]({% post_url 2026-04-19-On-Policy-Distillation %}), and how to do it when teacher and student use different tokenizers in [Cross-Tokenizer On-Policy Distillation]({% post_url 2026-08-26-cross-tokenizer-on-policy-distillation %}).

The whole arc is one shift in ambition: compress a model, speed it up, copy another model's behavior, and now transfer its reasoning.

### The controversy

Distillation is also at the center of a dispute. Anthropic has accused several labs (DeepSeek, Moonshot, MiniMax, and more recently Alibaba) of distilling Claude's outputs at scale through large numbers of accounts, which it says breaks its terms of service. OpenAI has said similar things. Model outputs are not copyrightable in the US, so the argument is about terms of service and access, not copyright. The post's takeaway is that learning from a stronger model's outputs is both how open models get built and what these fights are about. Permission and access are what separate the tool from the attack.

---

## 2. Three Ways Frontier Labs Distill in 2026

The class taught three stages: off-policy, on-policy, and self-distillation. The census maps each one onto real reports.

![Three uses of distillation in 2026 frontier post-training. Panel 1, big teacher to small student: off-policy, either soft labels (white-box, match the teacher's next-token distribution) or hard labels (black-box, SFT on teacher text), e.g. Gemma 3, DeepSeek-R1-Distill. Panel 2, merge RL experts: same-size domain experts trained with SFT then GRPO teach one student on its own rollouts with reverse KL (MOPD), e.g. DeepSeek-V4, MiMo-V2-Flash, Nemotron 3 Ultra, GLM-5. Panel 3, the teacher is you: the same model with a hint or an earlier checkpoint is the teacher, e.g. Cursor Composer 2.5, Thinking Machines continual learning](/assets/images/distillation_three_uses.svg)

### 2.1 A large teacher and a smaller student

The original use is still everywhere. Gemma 3's report says its post-training "relies on an improved version of knowledge distillation from a large IT teacher". The [Gemma 4]({% post_url 2026-04-05-Gemma4 %}) report describes a similar recipe, so some distillation is likely involved there too. DeepSeek-R1-Distill is the sequence-level flavor, plain SFT on the teacher's text.

These are the two flavors of the off-policy stage:

| Flavor | Signal | Access needed | Example |
| --- | --- | --- | --- |
| Soft labels | match the teacher's next-token distribution | white-box (logits) | Gemma 3 |
| Hard labels | train on the teacher's generated text | black-box (text only) | DeepSeek-R1-Distill |

### 2.2 Merge RL experts into one model

This is the newer use, and the one most frontier labs converged on this year. Getting one model good at everything through RL turns out to be hard, because skills gained in one training stage tend to degrade during the next. The workaround is to train a **separate RL expert per domain** (math, code, agentic tasks) and then distill all of them into one student **while the student generates its own rollouts**. The student writes, and the teachers grade every token.

The detail the author highlights: **the teachers are usually not bigger models.** They are checkpoints of the same base, the same size as the student, each pushed further in a single domain with RL. Specialization makes them good teachers, not scale.

- **[DeepSeek-V4]({% post_url 2026-04-26-DeepSeek-V4-Arch-Train %})** gives the cleanest description. Each domain gets its own expert (SFT, then GRPO). Afterwards "a single unified model is trained through on-policy distillation", with the student optimizing a **reverse KL** loss against the specialist teachers.
- **MiMo-V2-Flash** gave the multi-teacher form its name: **MOPD**, Multi-Teacher On-Policy Distillation, later studied in its own paper. Domain teachers provide a dense, token-level signal on whatever the student generates. (Its successor's RL recipe is in [MiMo-V2.6]({% post_url 2026-09-24-MiMo-V2.6-Scaling-RL-Self-Improvement %}).)
- **[GLM-5]({% post_url 2026-02-20-GLM5 %})** applies it across training stages instead of across domains. After the sequential RL phases, a final distillation pass recovers capability that degraded along the way, and the teacher is an earlier checkpoint of the same lineage. That is one step away from the model teaching itself.
- **[Nemotron 3 Ultra]({% post_url 2026-06-06-Nemotron-3-Ultra %})** adopts the multi-teacher form at scale: **more than ten** specialized teachers, each with its own domain pipeline, give the student dense token-level guidance on its own rollouts.
- **Qwen3** uses the same mechanic in the classic direction, a big teacher and small students that generate and align their logits to it. The report puts the cost at roughly **1/10 the GPU hours of RL**, with better results.

Every lab justifies this with the same argument. A teacher gives feedback on every token, while an RL reward is one number for the whole attempt. So distillation converges much faster on the exact behavior the student needs to fix. Thinking Machines' on-policy distillation write-up makes the same case with numbers, matching their RL baseline at a fraction of the compute.

*My gloss, not from the source posts:* written as an objective, reverse KL on the student's own samples is

$$
\mathcal{L}(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \sum_t \left[ \log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_{\text{teacher}}(y_t \mid x, y_{<t}) \right],
$$

which reads as a per-token reward of $-(\log \pi_\theta - \log \pi_{\text{teacher}})$ on every sampled token. It is mode-seeking: the student is pushed off tokens the teacher considers unlikely, rather than being asked to cover the teacher's whole distribution.

### 2.3 When the teacher is you

The third use drops the separate teacher.

- **Cursor's Composer 2.5** trains with self-distillation. A hint describing the desired behavior is injected into the context, and **the model with the hint becomes the teacher for the same model without it**. A per-token KL pulls the unhinted policy toward its hint-conditioned self, so the model ends up producing the behavior without needing the hint at inference. The class called this the *privileged teacher*.
- **Thinking Machines** shows another self-teacher using the same on-policy recipe with only the teacher swapped. After fine-tuning on new domain data, they distill from the **pre-fine-tune checkpoint** to restore behavior that fine-tuning erased, while keeping the new knowledge. This is their pitch for continual learning. It is the same problem GLM-5 solves across RL stages, at a personal scale.

The teacher does not need to be bigger, only better in context. Sometimes that is the model itself.

---

## 3. How This Sets Up RL

The next post in the series ([Training on Outcomes]({% post_url 2026-11-04-Training-On-Outcomes-GRPO-Family-IFStruct %})) shows the other half of the pipeline the reports describe. **RL finds the behavior using outcome rewards, then distillation transfers it into the final model with a dense signal.** DeepSeek-V4, MiMo-V2-Flash and LFM2.5-2.6B all run this two-step. Distillation and RL are not alternatives here. They are consecutive stages, which matches the view in [The Distributional Lens of Post-Training]({% post_url 2026-06-13-Distributional-Lens-Post-Training %}) that SFT, RL and on-policy distillation are points on one spectrum.

One systems note carries forward to the async RL post. Structurally, on-policy distillation **is** GRPO with the reward function swapped for a teacher forward pass. Every async-RL design question (rollout buffers, weight sync, staleness) applies to it unchanged. Self-distillation adds one more requirement: periodically hot-swapping the teacher server to a fresh snapshot of the student ([Keep the Tokens Flowing]({% post_url 2026-11-06-Async-RL-16-Libraries-Keep-Tokens-Flowing %})).

---

## Takeaways

1. **Distillation's purpose has moved from compression to transfer to capability integration.** The mechanic barely changed.
2. **Off-policy has two flavors.** Soft labels need logits (white-box). Hard labels need only text (black-box), which is also what makes distillation of closed models possible and contested.
3. **MOPD is the 2026 default for merging RL specialists.** The teachers are usually same-size, same-base checkpoints specialized by RL, not bigger models.
4. **Dense per-token signal is why it is cheap.** Qwen3 reports about 1/10 the GPU hours of RL for its students.
5. **Self-distillation (hint-conditioned or earlier-checkpoint teachers) is the frontier's tool for internalizing behavior and for fighting forgetting.**
