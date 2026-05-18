---
layout: post
title: "Native Interaction Models: Baking Real-Time Collaboration into the Weights"
date: 2026-05-16
categories: [TML]
tags: [TML, MoE, Multimodal, Inference, Streaming, Audio, Video]
---

Reading notes based on:
- [Interaction Models](https://thinkingmachines.ai/blog/interaction-models/) (Thinking Machines Lab)

This continues the [earlier TML reading notes]({% post_url 2025-11-21-TML %}) on LoRA, Modular Manifolds, and On-Policy Distillation. Where those posts focused on *training* dynamics, this one is about *serving*: how to architect a model that can hold a real-time, multimodal conversation without bolting a pipeline of external components onto a turn-based LLM.

---

## 1. The Collaboration Bottleneck

Today's frontier LLMs are optimized for *autonomous* operation: a user sends a turn, the model thinks, the model replies, repeat. Voice stacks built on top of that paradigm rely on a brittle chain of external components — Voice Activity Detection (VAD), turn-boundary classifiers, ASR encoders like Whisper, and TTS decoders — each one a hand-crafted heuristic that the rest of the system inherits the limitations of.

The thesis of TML's *Interaction Models* post is the application of Sutton's Bitter Lesson to interaction itself: **hand-crafted interaction rules will be surpassed by general models that learn interaction natively**. If interactivity is baked into the weights, then *scaling the model makes it simultaneously smarter and a better collaborator*, rather than forcing a trade-off between latency and intelligence.

---

## 2. Dual-Model System Architecture

The system splits the workload across two concurrent models to break the latency-vs-reasoning tradeoff:

- **Interaction Model** (TML-Interaction-Small): a **276B-parameter MoE with 12B active parameters**, running in a constant bidirectional loop across audio, video, and text. It maintains sub-second response presence and handles dialogue state intrinsically — no external manager.
- **Background Model**: runs asynchronously for deep reasoning, tool use, and long-horizon tasks. Its context and partial results stream back to the interaction model and are woven into the live conversation organically.

The split mirrors a System-1 / System-2 decomposition: the fast interaction model is always present, while the slow background model is invoked when depth is needed — without ever stalling the user-facing channel.

---

## 3. Time-Aligned Micro-Turns

The most consequential architectural choice is **discretizing time** instead of analyzing complete conversational turns.

Both input and output are treated as **streams**, divided into discrete **200ms chunks**. The transformer sees a flattened, interleaved sequence:

```
input_0, output_0, input_1, output_1, input_2, output_2, ...
```

Each `input_i` and `output_i` represents a 200ms slice of multimodal activity. Working at this granularity gives near real-time concurrency: the model can listen, see, and generate output **simultaneously** within the same forward pass cadence. Interruptions, backchannels, and overlapping speech become native capabilities rather than edge cases that need a state machine.

---

## 4. Encoder-Free Early Fusion

Rather than wiring large standalone encoders (Whisper-style ASR) and decoders (autoregressive TTS) around the LLM, all modalities are projected into a shared embedding space and **co-trained from scratch with the transformer backbone**.

| Modality | Input Path | Output Path |
| :--- | :--- | :--- |
| Text | Direct embedding | Standard unembedding |
| Audio | **dMel** (mel-spectrogram) → lightweight embedding | **Flow head** (flow-matching decoder, [Lipman et al. 2022](https://arxiv.org/abs/2210.02747)) |
| Video | **40×40 patches** → **hMLP** (hierarchical MLP) | — |

Inputs may be any subset of {text, audio, video} arriving on independent streams; outputs are produced as text + audio concurrently. This early-fusion design eliminates an entire class of latency, alignment, and error-cascade problems that plague pipeline architectures.

---

## 5. Inference Optimizations

A 12B-active MoE doing **200ms prefills forever** is not what off-the-shelf serving stacks are tuned for. Several optimizations make this tractable:

### 5.1 Streaming Sessions
The client sends each 200ms chunk as a separate request. The server appends chunks into a **persistent sequence in GPU memory**, eliminating frequent KV reallocations and metadata recomputation overhead. The implementation is upstreamed to SGLang.

### 5.2 MoE Decode Kernels: `gather + gemv` over Grouped GEMM
Standard grouped GEMM is the right call for prefill-heavy MoE workloads, but for the constant-cadence decode shapes here, the kernel of choice is a **gather + gemv** strategy explicitly tuned for low-latency bidirectional serving shapes.

### 5.3 Low-Latency Collectives
All-reduce and reduce-scatter use the **NVLS** (NVLink SHARP) path for low-latency, deterministic communication across tensor and sequence parallel dimensions. Determinism here is non-negotiable: any reduction-order variance breaks the bitwise-equivalence story that 5.4 and 5.5 depend on.

### 5.4 Split-KV Attention with Consistent Chunking
Split-KV attention is the standard way to keep attention efficient when the K/V cache grows long, but it introduces a subtle numerical headache: the **accumulation order differs between prefill and decode**, breaking bitwise equivalence. The fix is consistent chunking — splitting SMs to process **exactly 4096 tokens at a time, left-aligned** — so that prefill and decode use the same accumulation order and remain mathematically consistent.

### 5.5 Trainer-Sampler Alignment
**Batch-invariant kernels** (TML's [*Defeating Nondeterminism in LLM Inference*](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)) are used to guarantee **bitwise reproducibility between training and inference**, at a reported **<5% end-to-end overhead**. The trick: write attention and matmul kernels whose output is identical regardless of how the batch is split across SMs — so prefill and decode (which have different shapes and parallelization) produce the same numerics down to the last bit.

Why does this matter beyond engineering tidiness? The [Stabilize-RL notes]({% post_url 2025-12-02-Stabilize-RL %}) make the formal argument: any nondeterminism between the rollout engine (`μ_θ_old`) and the training engine (`π_θ_old`) corrupts the importance-sampling weight denominator and breaks the **first-order approximation** that justifies token-level policy gradient on a sequence-level reward. Batch-invariant kernels are the inference-side technique that makes that denominator exactly 1, restoring the approximation's validity. The same hygiene shows up here at serving time for prefill-decode parity in a streaming regime.

---

## 6. Benchmarks: New Capabilities, Not Just Better Numbers

Because the system unlocks **continuous multimodal perception**, TML introduces benchmarks that previous frontier models essentially cannot attempt. The interesting story is not just the absolute score; it is the gap between this architecture and turn-based baselines, which mostly score near zero.

| Benchmark | TML | GPT-realtime-2.0 | Gemini-3.1-flash-live | What it measures |
| :--- | :--- | :--- | :--- | :--- |
| Audio MultiChallenge (APR) | 43.4% | 48.5% (xhigh) | 36.1% (high) | Raw intelligence |
| FD-bench v1.5 | **77.8** | 46.8 (min) | — | Interaction quality |
| FD-bench v3 (Pass@1, w/ tools) | **82.8 / 68.0** | 81.0 / 58.0 (xhigh) | — | Agentic tool use |
| Turn-taking latency | **0.40 s** | 1.18 s (min) | 0.57 s (min) | Responsiveness |
| TimeSpeak | **64.7%** | 4.3% (min) | — | Time-aware proactive speech (e.g. "remind me to breathe every 4s") |
| CueSpeak | **81.7%** | 2.9% (min) | — | Simultaneous speech / correction on language switch |
| RepCount-A | **35.4%** | 1.3% (min) | — | Continuous visual counting from streamed video |
| ProactiveVideoQA (PAUC@0.5) | **33.5** | 25.0 (min ≈ random) | — | Speak only when visual evidence appears |
| Charades (mIoU) | **32.4** | 0% (min) | — | Temporal action localization via speech |

Two patterns to read off:

- **On raw intelligence (Audio MultiChallenge, FD-bench v3) TML is competitive, not dominant.** It loses to GPT-realtime-2.0 at xhigh reasoning effort by ~5 points on APR, but wins on FD-bench v3 (Pass@1 with tools) — meaning the agentic loop benefits even when the base reasoning is comparable.
- **On interactivity (TimeSpeak / CueSpeak / RepCount-A / Charades) the gap is categorical, not incremental.** Turn-based competitors score near zero because the tasks require the model to **perceive while staying silent**, then **act spontaneously on a time or visual cue**. A VAD-gated architecture has no mechanism to enter the speech state without an external trigger.

The latency picture is also worth noting: Gemini-3.1-flash-live at 0.57 s shows that low-latency turn-taking is achievable in turn-based stacks too, but the interactivity benchmarks then expose that low latency alone doesn't get you the streaming-perception capability.

### Safety

The streaming/audio setting makes refusal training non-trivial — refusals must sound natural in spoken form and behave consistently across speech and text channels. TML's pipeline uses TTS-generated refusal phrasings plus an automated multi-turn red-teaming harness. Result: **99.0% refusal on Harmbench**, on par with text-only frontier models.

---

## 7. Acknowledged Limitations

- **Long-session context accumulation** — a streaming model that runs forever accumulates context forever; no clean answer yet.
- **Connectivity dependence** — streaming audio/video assumes a reliable bidirectional channel; degraded networks expose the architecture in ways turn-based stacks don't.
- **No "Large" sibling yet** — only TML-Interaction-Small (276 B / 12 B) exists publicly, so the scaling story for this architecture is still extrapolation.
- **Background-agent integration is early** — the slow-model handoff works but is described as a work in progress.

Prior real-time efforts in the same space (Moshi, PersonaPlex, Nemotron VoiceChat, Qwen-Omni) are acknowledged; the distinguishing claim here is the *integrated* end-to-end multimodal training rather than external scaffolding around a turn-based LLM.

---

## 8. Why This Matters

Two threads tie this work back to the broader trajectory:

1. **The Bitter Lesson applied to the harness.** Just as RL/post-training is moving from hand-crafted reward shaping toward dense token-level signals (cf. [OPD]({% post_url 2025-11-21-TML %})), the serving stack is moving from hand-crafted dialogue management toward end-to-end learned interaction. The pattern is the same: replace external scaffolding with capability that scales with parameters and data.

2. **Numerics and the streaming regime.** The 4096-token Split-KV chunking and batch-invariant kernels are not flashy, but they are the kind of mathematical hygiene that becomes load-bearing once a system has to run forever in a streaming regime. The same concerns that show up in trainer-sampler parity for RL show up here for prefill-decode parity in streaming inference.

The clean separation between a **fast interaction model** and a **slow background model**, joined by streaming context exchange, is close to a clean implementation of System-1 / System-2 in production-grade infra. It is also a useful reminder that *latency is an architectural property*, not just a kernel-tuning one — the 200ms micro-turn structure is what makes everything else downstream possible.
