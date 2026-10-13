---
layout: post
title: "Reflection Beam: A 501B Open-Weight MoE Built on 100M RL Rollouts"
date: 2026-10-06
categories: [LLM, MoE]
tags: [Beam, Reflection, MoE, AsyncRL, Pretraining, MOPD, LoadBalancing, GB300, ReasoningEffort, OpenWeights]
---

Reading notes on:
- [Introducing Beam: Reflection's 501B open-weight model](https://reflection.ai/blog/introducing-beam)

Beam is Reflection's first open-weight foundation model, aimed squarely at software engineering, multi-step reasoning, and agentic work. It is a sparse MoE with **501B total and 23B active parameters**, pretrained on 23.8T tokens and then pushed hard with asynchronous RL: over **100 million rollouts** on 10,500 GB300 GPUs. The release is mostly a systems story. The architecture choices are about keeping numerics healthy under long RL, the data pipeline is about recovering technical text that generic filters throw away, and the RL platform is about tolerating a day of policy staleness without falling over.

These notes go through the architecture, pretraining, the RL infrastructure, inference efficiency and benchmarks, and the multi-teacher alignment recipe.

![Beam's training pipeline: pretraining on 23.8T tokens, midtraining to 1M context, a capability teacher scaled with 100M+ asynchronous RL rollouts and a separate safety teacher, merged by multi-teacher on-policy distillation into the released 501B/23B model](/assets/images/reflection_beam.svg)

---

## 1. Architecture: Sparse MoE Built for Numerical Health

```
Total Parameters:    501 Billion
Active Parameters:   23 Billion per token
Total Layers:        52
Attention Topology:  Interleaved Local and Global Attention with Elementwise Gating
```

Beam's 52 layers interleave local and global attention, so most layers handle nearby context cheaply and a subset tracks long-range dependencies. Each attention block has **elementwise gating** that scales the signal before it is projected into the residual stream.

### Auxiliary-loss-free load balancing, with a decay

Routing builds on the auxiliary-loss-free scheme from DeepSeek (covered in the [DeepSeek-V3 notes]({% post_url 2024-12-26-deepseek-v3 %})): a per-expert bias steers top-k selection without adding a balance term to the loss. Reflection adds one change: **cosine decay of the expert-bias updates**. As the model converges, the bias stops moving, so routing is not perturbed late in pretraining or during RL, when a routing shift can silently change which experts a token touches. (Routing drift between training and inference is a recurring theme; see [Training-Inference Parity in MoE Models]({% post_url 2026-04-08-MoE-Numeric-Parity %}) and the frozen-router choice in [MiMo-V2.6]({% post_url 2026-09-24-MiMo-V2.6-Scaling-RL-Self-Improvement %}).)

Sequence-level balancing is also enforced so experts stay evenly used on out-of-distribution data. At the end of pretraining, layer-averaged expert load is nearly flat:

$$\text{Peak Expert Load} = 1.04 \times \text{Uniform Load}$$

That matters both for capacity (all 501B parameters actually learn) and for systems: in expert parallelism, the hottest expert sets the step time. For how far rack-scale balancing can be pushed from the systems side, see [UltraEP]({% post_url 2026-08-12-UltraEP-Exact-Load-Balancing-Rack-Scale-MoE %}).

### Keeping the residual stream bounded

Deep MoEs tend to grow activation norms over long training. Beam uses four containment measures:

1. **Depth-based scaling** of sublayer outputs, so contributions do not add up linearly with depth.
2. **SandwichNorm**: normalization both before and after each sublayer.
3. **Elementwise attention gating** on what leaves the attention block.
4. **FP32 residual accumulation**, so small updates added to a large residual are not lost to rounding.

The reported residual-stream RMS stays bounded across all 52 layers through pretraining, midtraining, and RL, with smooth trajectories and no spikes. The FP32 residual choice echoes the argument in [High Precision Used for Reasoning Recipes]({% post_url 2025-06-23-FP32-Reasoning %}): precision is cheap where it matters most.

---

## 2. Pretraining: Data, Midtraining, and the Cluster

| Data dimension | Value |
| :--- | :--- |
| **Total pretraining tokens** | 23.8 trillion |
| **Raw web filtering rate** | 95% eliminated via deduplication, parsing, and quality classification |
| **Retained custom web-code tokens** | 1.8 trillion tokens (~87% retained via in-house classifiers) |
| **Pretraining compute** | 6,144 NVIDIA GB300 NVL72 GPUs |
| **Pretraining time** | Under 4 weeks |
| **Pretraining goodput** | 92.3% |

### Curating for technical text

The central claim is that generic web filters discard valuable technical content because code and math look like noise to heuristics built for prose. Reflection trained its own quality classifiers for web, code, and STEM:

- **Web.** 95% of raw web tokens were discarded, but the classifiers kept **1.8T high-quality tokens**, including about 87% of curated web-code tokens that standard open-source filters drop.
- **Code.** Filtering is per language, removing autogenerated boilerplate, corrupted syntax, and unlearnable scripts. Overrepresented languages are downsampled for broader coverage.
- **PDFs and STEM.** A distributed vision-language OCR pipeline across thousands of GPUs ingests scientific papers and manuals, with artifact detectors to catch bad reconstructions.

High-quality code and documentation were **repeated** over the training horizon, with fuzzy deduplication and optimized packing to avoid memorization.

### Midtraining

A dedicated midtraining stage builds the prior for RL:

- **Context extension to 1M tokens** using code repositories, long-horizon tasks, and long-form technical synthesis.
- **Reasoning-rich synthetic data**: long logic chains and multi-step transformations to prime long rollouts.

### Running 6,144 GPUs for four weeks

- **Topology-aware scheduling**: a Kubernetes-integrated scheduler maps all-reduce patterns onto the physical rack and switch topology.
- **Silent data corruption (SDC) detection**: automated monitoring for non-deterministic gradient-norm spikes and hardware math anomalies. The run needed **9 semi-automatic rewinds** for gradient spikes or suspected SDC.
- **92.3% goodput**, from fast checkpointing and rapid isolation of bad nodes.

---

## 3. RL at 100 Million Rollouts

RL is Beam's main scaling axis.

```
RL GPU Allocation:          10,500 NVIDIA GB300 GPUs
Campaign Duration:          4 Weeks
Total Generated Rollouts:   > 100 Million (> 80M for Reasoning Expert)
Max RL Context Length:      256,000 (256K) Tokens
Total Sandbox Requests:     ~ 1.3 Billion
Peak Concurrent Sandboxes:  170,000 across 20 clusters, 2 clouds, 4 regions
Environment Setup Latency:  90% ready in < 10 seconds
```

For scale: [Inkling]({% post_url 2026-07-17-Inkling-975B-MoE %}) reported 30M+ rollouts and MiMo 753K. Rollouts draw on roughly **1 million environments** across SWE, terminal, competitive coding, STEM, browsing, and tool use, continuously filtered for underspecified, guessable, or broken tasks. The sandbox layer (1.3B requests, 170K concurrent) is in the same regime as [DeepSeek DSec]({% post_url 2026-09-25-DeepSeek-DSec-Sandbox-Infrastructure-Agentic-RL %}).

### Staleness: up to one day, 107 updates

The trainer is fully asynchronous: generation fleets run decoupled from optimization, so a single rollout can contain tokens sampled from checkpoints several versions old. Reflection reports stable learning on rollouts with **up to 1 day of staleness, i.e. 107 policy updates behind** the trainer. That is far beyond the few-step lag most async systems assume. How they correct for it is not described; [Score Centering]({% post_url 2026-09-17-Score-Centering-Off-Policy-RL-Drift %}) covers what goes wrong under staleness and how importance sampling plus centering handle it, and [GLM-5.3's single-rollout async RL]({% post_url 2026-08-16-GLM-5.3-Post-Training-Scaling-IndexShare-SAO %}) is a point of comparison.

### Fleet mechanics

- **Inference-to-training ratio** shifted dynamically between **3.9:1 and 5.4:1**. Generation dominates, as the throughput analysis in [RL Systems Mind the Gap]({% post_url 2026-06-19-RL-Mind-The-Gap %}) predicts.
- **Hierarchical weight distribution**: new weights reach the rollout fleet in a **median of 12 seconds**, sent across racks over RoCE and fanned out locally over NVLink. That cut cross-rack traffic by **75%** and sped up weight sync **2.2×**.
- **Dynamic batch packing** kept batch occupancy at **99.99%** and per-GPU trainer throughput within **1.5%** of peak, even as mean rollout length grew by nearly 70%.
- **Fault tolerance**: **71 inference-side incidents** were absorbed without stopping training. Median capacity recovery was 8 minutes, and lost work was **0.02%** of GPU-minutes.

---

## 4. Inference Compute, Length Control, and Benchmarks

### Counting generation FLOPs

Reflection compares efficiency with the standard forward-pass estimate:

$$\text{FLOPs} \approx 2 \times N_{\text{active}} \times M_{\text{tokens}}$$

where $N_{\text{active}}$ is active parameters per token ($23 \times 10^9$ for Beam), $M_{\text{tokens}}$ counts all generated tokens (reasoning plus answer), and each multiply-add is 2 FLOPs. Prefill, context-dependent attention, and serving overhead are excluded. This makes the metric a clean comparison of *generation* cost, but it favors sparse models with short outputs and ignores the attention cost of long contexts. For the full serving picture, see [The Economics of a Token]({% post_url 2026-05-17-token-economics %}).

### A length penalty with two phases

RL used a controllable length penalty that initially penalized token inflation. Training then went through two phases:

1. **Efficiency contraction**: the policy learned to solve tasks with fewer, denser tokens, so length fell while accuracy rose.
2. **Capability expansion**: higher reasoning efforts let completions grow again to unlock harder agentic tasks.

At inference this is exposed as a **reasoning-effort parameter** (mechanics in [The Mechanics of Reasoning Effort and Inference Scaling]({% post_url 2026-07-19-Reasoning-Effort-Inference-Scaling %})). On hard benchmarks, Beam matches GLM-5.2 accuracy with **3 to 4× less inference compute**. Active parameters alone (23B vs. [GLM-5.2]({% post_url 2026-06-21-GLM-5.2 %})'s 40B) explain less than 2× of that, so most of the gap has to come from shorter outputs. The contraction phase is the same "dense reasoning" behavior the [hidden-CoT extraction study]({% post_url 2026-10-05-Hidden-Chain-of-Thought-Forced-Reasoning %}) found in GPT-6 Astra, here trained in deliberately.

### Benchmarks

| Domain | Benchmark | Beam (501B/23B) | Inkling | Nemotron 3 Ultra | GLM 5.2 | GLM 5.3 | Kimi K3 | Qwen 3.8 Max | DeepSeek V4.1 Flash |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Agentic / Coding** | DeepSWE v1.1 | **44.4** | NR | NR | 44.0 | 61.0 | 68.0 | 51.0 | 74.2 |
| | SWE Bench Pro v2-Hard | **77.2** | 56.9 | NR | NR | 84.3 | 88.2 | NR | NR |
| | SWE Bench Pro v1 | **65.5** | 54.3 | 46.4 | 62.1 | NR | NR | 67.7 | NR |
| | Terminal Bench v2.1 | **80.1** | 63.8 | 56.4 | 81.0 | 88.2 | 88.3 | 86.6 | 90.6 |
| | SWE Atlas Codebase QnA | **34.6** | NR | NR | NR | 61.0 | 68.0 | NR | NR |
| | SWEBench Multilingual | **78.0** | NR | 67.7 | NR | NR | NR | NR | NR |
| | SWEBench Verified | **80.9** | 77.6 | 70.7 | NR | NR | NR | NR | NR |
| **Reasoning** | AIME 2026 | **97.8** | 97.1 | NR | 99.2 | NR | NR | NR | NR |
| | HLE (no tools) | **36.2** | 29.7 | 26.7 | 40.5 | 42.3 | 46.9 | 43.6 | 39.1 |
| | SciCode | **49.7** | 46.1 | 44.6 | NR | 59.0 | 58.7 | 52.1 | 52.0 |
| | CriPT AA | **16.3** | 5.4 | 3.1 | 20.9 | 19.1 | 23.4 | 20.0 | 14.3 |
| | GPQA Diamond | **90.5** | 87.2 | 87.0 | 91.2 | 91.7 | 93.5 | 92.6 | 90.9 |
| **Tool Calling / Search** | AutomationBench public | **37.0** | NR | NR | 26.2 | 48.2 | 46.7 | 39.8 | 54.8 |
| | MCP Atlas | **78.7** | 76.0 | 63.1 | 77.8 | 84.2 | 82.3 | 84.5 | NR |
| | tau3 banking | **38.0** | 25.0 | 22.6 | 37.1 | NR | 37.1 | 55.2 | NR |
| | BrowseComp (w/ context) | **77.4** | 77.1 | 44.4 | NR | NR | 91.2 | NR | NR |
| | DeepSearchQA (w/ context) | **80.1** | NR | NR | NR | NR | 95.0 | NR | NR |
| **General** | AA-LCR | **79.3** | 77.3 | 79.3 | 78.3 | 79.7 | 88.7 | 80.3 | 84.0 |
| | LongBench v2 | **65.5** | NR | 61.9 | 64.0 | NR | NR | 66.3 | NR |
| | IFBench | **79.7** | 79.8 | 81.7 | 73.3 | NR | NR | 82.8 | NR |
| | AA Omniscience | **13.0** | 14.2 | 8.6 | NR | 22.6 | NR | NR | 6.6 |

*NR: not reported in the official disclosures.*

Read honestly, Beam clearly beats [Inkling]({% post_url 2026-07-17-Inkling-975B-MoE %}) and [Nemotron 3 Ultra]({% post_url 2026-06-06-Nemotron-3-Ultra %}) on most rows. It is roughly level with GLM-5.2, and behind the newer, larger [GLM-5.3]({% post_url 2026-08-16-GLM-5.3-Post-Training-Scaling-IndexShare-SAO %}), [Kimi K3]({% post_url 2026-07-18-Kimi-K3 %}), and [DeepSeek-V4.1-Flash]({% post_url 2026-09-14-DeepSeek-V4.1-Flash-KV-Cache-Compression %}) on most agentic and reasoning rows (HLE 36.2 vs. 46.9 for Kimi K3; Terminal Bench 80.1 vs. 90.6 for DeepSeek). The pitch is the compute ratio, not the leaderboard.

### Out-of-distribution case studies

- **Unprompted web search and API use**: during reasoning RL, Beam learned to issue web queries, look up model documentation, call OCR APIs, and invoke other LLMs without browsing supervision.
- **Land or water**: on 16,200 grid points ($180 \times 90$ over latitudes $-89^\circ$ to $89^\circ$ and longitudes $-179^\circ$ to $179^\circ$), Beam scored **95.5%**, between Claude Opus 5 (92.5%) and Fable 5 (97.8%).
- **NYC subway live map**: from public MTA data, Beam read the docs, checked authentication, parsed route geometry, and built frontend and backend code that renders trains in real time.
- **p5.js astronaut game**: a 3D asteroid-dodging game from a text prompt, showing spatial reasoning without multimodal input.
- **Gemma-4 fine-tuning**: inside OpenCode, Beam read Unsloth docs and Gemma-4 model cards and built a Text2SQL fine-tuning pipeline that improved Gemma's test accuracy by **66.5%**.

---

## 5. Alignment: Two Teachers and On-Policy Distillation

### Multi-Teacher On-Policy Distillation (MOPD)

Tuning one policy for capability and then safety tends to cause forgetting or over-refusal. Reflection instead branches **two teachers** from the same pretrained base:

1. **Capability teacher**: scaled with the high-compute RL above for SWE, STEM, and reasoning.
2. **Safety and alignment teacher**: trained with targeted SFT and RL for policy adherence and helpfulness.

Both are then distilled **on-policy** into the final Beam weights. Because the student learns from its own samples against both teachers at once, neither objective has to be learned on top of the other. The approach and its failure modes (one teacher dominating) are covered in [Open-MOPD]({% post_url 2026-08-22-open-mopd %}). The general case for on-policy distillation over SFT is in [The Distributional Lens of Post-Training]({% post_url 2026-06-13-Distributional-Lens-Post-Training %}).

### A three-tier behavior spec

- **Tier 1, hard rules**: non-negotiable safety policies and preservation of AI identity.
- **Tier 2, core qualities**: context grounding, factual precision, transparent reasoning, explicit uncertainty.
- **Tier 3, default style**: direct, concise, efficient, proactive.

### Generative reward models and deliberative alignment

For non-verifiable domains Reflection uses generative reward models, and reports that the reward model's offline signal predicts actual RL gains well:

$$\text{Correlation with RL Gains: } r = 0.79 \quad \text{(vs. Best-of-N Ceiling } r = 0.46\text{)}$$

That is a useful property: it lets you choose a reward model before paying for the RL run, the same goal as [predicting post-RL performance from SFT checkpoints]({% post_url 2026-09-16-SFT-Scores-Mislead-Predicting-Post-RL %}). Safety training adds **deliberative alignment** (Guan et al., 2024), writing policy principles into the CoT itself, plus iterative red-teaming whose jailbreaks and over-refusals are folded back into the SFT and RL mixtures.

---

## 6. Release

Weights ship under **Apache 2.0** later this month, along with the model card, a full technical report and safety evaluation suites, fine-tuning harnesses, developer scripts, and partner integrations.

---

## Takeaways

1. **Numerics are a design goal.** Bias-update decay, sequence-level balance, SandwichNorm, gating, and FP32 residuals are all aimed at one thing: a model that stays stable through a long RL campaign without routing or activation drift.
2. **Staleness tolerance is the RL headline.** Learning from rollouts 107 updates old is what lets a 10,500-GPU fleet run at 99.99% batch occupancy through 71 incidents. The algorithm behind it is the most interesting thing the post does not explain.
3. **Efficiency, not peak score.** Beam does not top the open leaderboard, but matching GLM-5.2 at 3 to 4× less generation compute, with a trained-in effort knob, is a credible position for an Apache-licensed 23B-active model.
