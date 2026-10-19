---
layout: post
title: "StreamingLLM: Attention Sinks and Why Window Attention Collapses"
date: 2026-10-17
categories: [LLM, Inference]
tags: [StreamingLLM, AttentionSink, KVCache, WindowAttention, LongContext, Softmax, RoPE, ALiBi, Streaming]
---

Reading notes on:
- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/pdf/2309.17453)

Serving an LLM as a long-lived assistant runs into two walls at once. The **KV cache grows linearly** with the conversation, so memory and per-token latency keep climbing until the server runs out of memory. And past its **pretraining window $L$**, a model's attention and position encodings (RoPE, ALiBi) stop generalizing, so perplexity degrades even if memory were free.

The obvious fix is **window attention**: keep only the last $W$ tokens' KV states. Memory and compute become constant. It also fails catastrophically: perplexity explodes the moment the *first* tokens of the stream are evicted, even though those tokens are usually semantically irrelevant to what is being generated.

StreamingLLM explains why, and the fix is almost embarrassingly small. Autoregressive LLMs dump a large share of attention onto the first few tokens regardless of content, using them as **attention sinks**. Keep the KV states of **4 initial tokens** next to a rolling window of recent tokens, and pretrained models (Llama-2, Falcon, Pythia, MPT) stream stably past **4 million tokens** with **constant memory**, no fine-tuning, and up to **22.2× speedup** over the sliding-window-with-recomputation baseline.

![Panel A: the KV cache contents of four policies while decoding a token far past the window; dense keeps everything, window attention evicts the first tokens and perplexity jumps to 5158, sliding window with recomputation is stable but 22.2x slower, StreamingLLM keeps 4 sink tokens plus a rolling window at PPL 5.40. Panel B: softmax must sum to 1 so leftover attention parks on the first tokens; positions are re-indexed by slot inside the cache](/assets/images/streamingllm_sinks.svg)

---

## 1. Four Ways to Manage the Cache

Decoding at position $T$ with $T$ much larger than $L$:

| Strategy | Time (T tokens) | Memory | Perplexity past $L$ | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Dense attention** | $O(T^2)$ | $O(T)$, unbounded | Fails beyond the pretraining window | Not viable for long streams |
| **Window attention** | $O(TW)$ | $O(W)$ | Collapses (PPL > 5000) | Unusable |
| **Sliding window + recomputation** | $O(TL^2)$ | $O(L)$ | Stable (oracle, PPL ≈ 5.4) | Correct but up to 22.2× slower |
| **StreamingLLM** | $O(TW)$ | $O(W)$ | Stable, matches the oracle (PPL ≈ 5.4) | Built for continuous streaming |

Sliding window with recomputation is the quality reference: at every step it rebuilds the KV states of the last $L$ tokens from scratch, so the model always sees a "fresh" sequence that starts at position 0. That is exactly why it is stable, and exactly why it is slow. StreamingLLM gets the same perplexity at the cost of window attention.

---

## 2. Why Window Attention Fails

### The attention sink phenomenon

Look at attention maps layer by layer and a pattern appears. The first two layers attend locally, as expected. From there on, almost every head sends a large fraction of its attention to the **first token**. In Llama-2-7B, the last query token puts **over 50% of its softmax weight** on token 1 in most deeper heads, regardless of sequence length.

### It is a softmax artifact

With logits $x_i = \mathbf{q}^\top \mathbf{k}_i / \sqrt{d}$,

$$\text{SoftMax}(\mathbf{x})_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}} = \frac{e^{x_i}}{e^{x_1} + \sum_{j=2}^{N} e^{x_j}}$$

The weights must sum to exactly 1. When a head has nothing useful to attend to (the query already carries what it needs, or nothing in context is relevant), the probability mass still has to go *somewhere*. The model learns to park it on a token that is always available.

### Why the first tokens

Causal masking makes the choice for it. The initial tokens $t_1, t_2, \dots$ are visible to **every** later token in every training sequence; a token at position $k$ is only visible to positions $j \ge k$. So the first tokens are the only ones that can serve as a universal dumping ground, and training shapes their keys to absorb the excess attention.

Window attention then evicts exactly those tokens. Removing $e^{x_1}$, the dominant term, from the denominator rescales every remaining attention weight, the distribution moves far from anything seen in training, and perplexity spikes.

### Position, not meaning

To check that the sinks matter because of *where* they are, not *what* they say, replace the first four tokens with newlines. Llama-2-13B on PG-19:

| Cache (initial + recent) | PPL |
| :--- | :--- |
| Window attention, 0 + 1024 | 5158.07 |
| Attention sink, 4 initial + 1020 recent | 5.40 |
| Dummy sink, 4 × `\n` + 1020 recent | 5.60 |

Meaningless newlines recover almost all of the quality. The model needs *something* in the first positions to absorb attention; the content barely matters.

This "softmax needs a place to put nothing" story is why recent architectures ship an explicit sink by default: learned sink logits in gpt-oss ([LLM Architecture Evolution]({% post_url 2025-12-05-LLM-Arch %})) and a learnable attention sink in [DeepSeek-V4]({% post_url 2026-04-26-DeepSeek-V4-Arch-Train %}).

---

## 3. The StreamingLLM Cache

```
Logical (stream) positions:  [ 0  1  2  3 ] ... [ 100  101  102  103  104 ]
Positions used inside cache: [ 0  1  2  3 ] ... [   4    5    6    7    8 ]
                              attention sinks       rolling KV window
```

### Two blocks

1. **Attention sinks**, $S$ tokens (typically $S = 4$): the KV states of the first 4 tokens, kept permanently to anchor the softmax denominator.
2. **Rolling window**, $W - S$ tokens: a circular buffer of the most recent tokens, which carries the actual local language-modeling context.

### Cache-centric position re-indexing

The subtle part: if cached keys keep their *original* stream positions, the relative distance between the current query and the sinks grows without bound and blows past anything seen in training. StreamingLLM assigns positions by **slot within the cache**, not by position in the stream.

**RoPE** (Llama-2, Falcon, Pythia):

1. Cache the **unrotated** keys $\mathbf{k}_i$.
2. At each decoding step, assign the cached keys positions $0, 1, \dots, \text{CacheSize} - 1$.
3. Apply the rotation at attention time:

$$\mathbf{q}_m^{\text{rot}} = R_{\Theta, m}\,\mathbf{q}_m, \qquad \mathbf{k}_p^{\text{rot}} = R_{\Theta, p}\,\mathbf{k}_p, \qquad p \in [0, \text{CacheSize} - 1]$$

**ALiBi** (MPT): apply the linear distance bias using contiguous in-cache distances, rather than stream distances that would jump discontinuously between the sinks and the window.

---

## 4. Pretraining With a Dedicated Sink

Vanilla models spread sink behavior across the first 3 to 4 tokens because pretraining chunks don't start with a consistent token. The paper asks whether a sink can be designed in at pretraining time, and compares three options on 160M-parameter models:

- **Vanilla softmax.**
- **$\text{SoftMax}_1$ ("zero sink")**, which drops the sum-to-one constraint:

$$\text{SoftMax}_1(\mathbf{x})_i = \frac{e^{x_i}}{1 + \sum_{j=1}^{N} e^{x_j}}$$

  This is equivalent to prepending a virtual token whose key and value are both zero.
- **A learnable `[Sink]` token** prepended at position 0 of every training sample.

| Variant | 0 + 1024 cache PPL | 1 + 1023 cache PPL | Sinks needed when streaming | Zero-shot benchmarks |
| :--- | :--- | :--- | :--- | :--- |
| Vanilla | 27.87 | 18.49 (unstable) | ≥ 4 initial tokens | Baseline |
| Zero sink ($\text{SoftMax}_1$) | 29214 | 19.90 (unstable) | More than 1 initial token | Slight degradation |
| **Learnable sink token** | 1235 | **18.01 (stable)** | **Exactly 1** | No penalty, small gains |

The learnable sink concentrates all of the sink behavior into one slot, so streaming needs only $1 + (W - 1)$ cache entries, and it costs nothing on downstream tasks. Notably, $\text{SoftMax}_1$ alone is *not* enough: giving the head a zero-valued escape hatch doesn't stop it from also leaning on the initial tokens.

---

## 5. Results

### Streaming language modeling

On concatenated PG-19 (100 books, more than 4M tokens), StreamingLLM keeps perplexity flat across families and sizes: **Llama-2** 7B/13B/70B, **Falcon** 7B/40B, **Pythia** 2.8B/6.9B/12B, and **MPT** 7B/30B. Dense attention diverges past $L$ (or runs out of memory), and window attention spikes as soon as the first tokens leave the cache.

### Latency and memory

On a single A6000 with Llama-2-7B and 13B, per-token decoding is up to **22.2× faster** than sliding window with recomputation, because each step costs $O(W)$ instead of $O(L^2)$. Memory stays flat at the configured cache size.

### Multi-round streaming QA

On ARC-Easy, ARC-Challenge, and StreamEval run as one continuous stream: dense attention runs out of memory, window attention drops to near-zero accuracy once question-answer pairs exceed the cache, and StreamingLLM matches the one-sample-at-a-time baseline (**about 71.3%** on ARC-E with Llama-2-7B).

### What it does not do

StreamingLLM stabilizes decoding; it does **not** extend memory. A token that leaves the rolling window is gone, so tasks that need long-range retrieval (questions about the middle of a long document, whole-book summarization) are no better off. It is complementary to context-extension methods (LongChat-32k, RoPE scaling), which make the window itself larger. The [LongPPL post]({% post_url 2026-08-19-longppl-key-tokens-long-context %}) is a good reminder of this: flat perplexity on a stream says little about whether the model can *use* long context.

---

## 6. Where This Shows Up Today

Sliding-window layers are now standard in hybrid attention stacks: MAI-Base-1 ([MAI-Thinking-1]({% post_url 2026-06-07-MAI-Thinking-1 %})) and [MAI-Base-2]({% post_url 2026-10-18-MAI-Base-2-Frontier-Pretraining %}) interleave local and global layers, as do [Gemma 4]({% post_url 2026-04-05-Gemma4 %}) and gpt-oss. Every local layer is a window-attention layer, which is why explicit sink mechanisms such as learned sink logits often travel with them. StreamingLLM's diagnosis is the reason those designs carry a sink at all.

---

## Takeaways

- **Window attention fails because of softmax, not missing context.** Evicting the first tokens removes the denominator's largest term and pushes attention out of distribution.
- **The first tokens are sinks by position, not meaning.** Four newlines work nearly as well as the real tokens (PPL 5.60 vs 5.40).
- **Serving existing models:** keep **4 sink tokens + a rolling window** ($4 + (W - 4)$), and always **re-index positions within the cache**; keeping stream positions breaks once tokens are evicted.
- **Pretraining new models:** prepend a **learnable sink token** to every sample. One sink slot is then enough ($1 + (W - 1)$), at no cost to quality.
- **Know the limit:** StreamingLLM gives stable, constant-memory decoding, not longer memory. Evicted tokens cannot be recalled.
