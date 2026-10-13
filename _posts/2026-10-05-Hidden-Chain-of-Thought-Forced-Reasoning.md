---
layout: post
title: "Capable yet Parsimonious: What Hidden Chain-of-Thought Looks Like When You Force It Out"
date: 2026-10-05
categories: [LLM, Reasoning]
tags: [ChainOfThought, Reasoning, ToolCalling, Interpretability, Distillation, Compression, ReasoningTrees, APIDesign]
---

Reading notes on:
- [Capable yet Parsimonious: Extracting and Characterizing Hidden Chain-of-Thought in Frontier Models](https://arxiv.org/pdf/2609.26637)

Frontier models lean on chain-of-thought (CoT) for math, science, and code, but closed providers hide the native trace. GPT-6 Astra, GPT-5.6 Sol, Claude Opus 4.8, and Claude Sonnet 5 return a final answer, sometimes a summary, and a reasoning-effort knob. That leaves a **measurement gap**: benchmark accuracy tells you *what* a model solves, not *how*, so you cannot tell a sound inference path from a post-hoc rationalization.

This paper closes the gap with a surprisingly small trick. The **FORCED-REASONING** protocol uses nothing but standard tool calling to make a closed model write its intermediate reasoning into a scratchpad tool argument. With those traces in hand across MATH, Humanity's Last Exam (HLE), and LiveCodeBench (LCB), the authors measure length, compressibility, local granularity, global tree shape, and whether one model's trace can be reused by another. The headline is in the title: the strongest model, GPT-6 Astra, reasons *parsimoniously*. Its traces are short, dense, and nearly linear, and that density is exactly what makes them hard for small models to use.

It is a natural companion to [Cracking the Vault]({% post_url 2026-08-15-Cracking-The-Vault-Encrypted-Reasoning-Extraction %}), which pulled reasoning out of encrypted envelopes via model siblings. FORCED-REASONING needs no siblings and no ciphertext: it just asks through the tool interface.

![FORCED-REASONING: force a scratchpad tool call with native reasoning off, record the argument, acknowledge with Received, and loop on auto; the extracted traces show GPT-6 Astra writing the shortest and least compressible reasoning, and its traces transfer to strong recipients but not weak ones](/assets/images/forced_reasoning.svg)

---

## 1. The FORCED-REASONING Protocol

The protocol needs no weights, no shadow-model demonstrations, and no fine-tuning. It registers one free-form tool, turns native reasoning down to `none` (or `low`), and forces the first call to that tool:

```python
# Initialize message payload and register free-form scratchpad tool
messages = [system_prompt, user_prompt]
reasoning_config = {"effort": "none", "exclude": False}
tools = [{
    "name": "forced-reasoning",
    "description": "Scratchpad for working through the problem.",
    "parameters": {
        "type": "object",
        "properties": {"reasoning": {"type": "string"}},
        "required": ["reasoning"]
    }
}]

# Force initial tool choice
response = chat(messages, reasoning_config, tools, tool_choice="forced-reasoning")

while response.has_tool_calls():
    messages.append(response.assistant_message)
    for call in response.tool_calls:
        # Extract and archive intermediate reasoning text
        extracted_trace = parse_json(call.arguments)["reasoning"]
        record_trace(extracted_trace)
        # Return fixed acknowledgment
        messages.append(tool_result(call.id, content="Received"))
    # Restore standard automatic selection for subsequent steps
    response = chat(messages, reasoning_config, tools, tool_choice="auto")

return response.visible_text
```

The tool does no computation. Its only job is to move reasoning out of the hidden channel and into a visible payload, while keeping it in the conversation history so the model can build on it. After the first forced call, `tool_choice` goes back to `auto` and the model decides when it is done.

Why does this work? With native reasoning switched off, the model still wants to think, and a tool whose schema says "scratchpad" is the only place left to do it. This lines up with the thinking-mode-fusion picture in [The Mechanics of Reasoning Effort and Inference Scaling]({% post_url 2026-07-19-Reasoning-Effort-Inference-Scaling %}): the "off" switch removes a channel, not the capability.

---

## 2. Are the Traces Real Reasoning?

The obvious objection is that a tool-argument trace might be a rationalization written after the fact. The authors validate on two open models, **DeepSeek-V4-Flash** and **GLM-5.2**, where native CoT is visible and can be compared directly.

**Performance recovery.** On MATH, forced reasoning recovers near-native accuracy and far exceeds the no-reasoning baseline:

| Model | No reasoning | Forced | Native (high) |
| :--- | :---: | :---: | :---: |
| DeepSeek-V4-Flash | 30.5% | 73.3% | 70.0% |
| GLM-5.2 | 21.4% | 84.3% | 89.9% |

**Lexical overlap.** Native-to-forced pairs (N–F) are almost as similar to each other as two native samples (N–N) are:

| Model | Pair | ROUGE-1 F1 | ROUGE-L F1 | 5-gram Jaccard | Median length ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DeepSeek-V4-Flash | N–N | 0.613 | 0.198 | 0.017 | $1.42\times$ |
| | N–F | 0.538 | 0.188 | 0.017 | |
| GLM-5.2 | N–N | 0.588 | 0.196 | 0.019 | $1.39\times$ |
| | N–F | 0.465 | 0.171 | 0.008 | |

(A single median length ratio is reported per model, so the N–F rows leave that column blank.)

**Functional alignment.** Segmenting traces with **Schoenfeld Episode Theory** gives matching distributions of functional episodes in native and forced modes.

On the closed models the protocol reproduces native accuracy to within a few points almost everywhere:

| Benchmark | Method | Claude Opus 4.8 | Claude Sonnet 5 | GPT-5.6 Sol | GPT-6 Astra |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MATH** | None | 72.5% | 42.5% | 25.0% | not reported |
| | Native | 86.3% | 77.5% | 97.5% | 97.5% |
| | **Forced** | **85.0%** | **81.3%** | **91.3%** | **93.8%** |
| **HLE** | None | 24.0% | 17.0% | 13.0% | not reported |
| | Native | 33.0% | 21.0% | 25.0% | 35.0% |
| | **Forced** | **27.0%** | **22.0%** | **23.0%** | **32.0%** |
| **LCB** | None | 58.0% | 52.0% | 47.0% | not reported |
| | Native | 83.0% | 77.0% | 90.0% | 92.0% |
| | **Forced** | **82.0%** | **77.0%** | **90.0%** | **89.0%** |

The largest gap is Opus 4.8 on HLE (33 to 27). Sonnet 5 on MATH actually does *better* forced than native.

---

## 3. Four Dimensions of Hidden Reasoning

### 3.1 Length and information density

Redundancy is measured with a lossless compression ratio against mean forced output length $O_F$:

$$\text{zlib ratio} = \frac{\text{Compressed Bytes (zlib Level 9)}}{\text{Original UTF-8 Uncompressed Bytes}}$$

A higher ratio means the text is hard to compress: less repetition, fewer predictable patterns.

| Model | zlib ratio | Forced output tokens $O_F$ |
| :--- | :---: | :---: |
| GPT-6 Astra | 0.56 to 0.62 | 650 to 1,752 |
| GPT-5.6 Sol | 0.45 to 0.47 | 2,214 to 3,925 |
| Claude Opus 4.8 | 0.41 to 0.45 | 3,627 to 16,283 |
| Claude Sonnet 5 | 0.37 to 0.39 | 10,705 to 27,851 |

(Ranges span the three benchmarks.) Astra sits alone in the corner of short-and-dense on every benchmark. Sonnet 5 emits up to about 28k tokens of highly redundant text. The two axes move together: the longer a model's trace, the more compressible it is.

### 3.2 Local granularity: what gets verbalized

Side-by-side steps show two compression habits in Astra:

1. **Collapsed arithmetic and algebra.** Routine derivations are resolved internally instead of written out.
2. **Implicit retrieval.** Retrieved facts are used without being restated.

The valence-electron count on HLE-075 makes it concrete:

- **Claude Opus 4.8:** "Valence electrons: C=4, H=1, O=6. 12\*4 + 18\*1 + 6\*6 = 48+18+36 = 102."
- **GPT-5.6 Sol:** "Valence electrons: C12\*4=48, H18=18, O6\*6=36 total 102. Exactly."
- **GPT-6 Astra:** "valence48+18+36=102."

Astra follows a **sparse externalization** strategy: elementary operations stay in the activations, and only the pivot decisions reach the text. This is the same intuition behind latent-depth architectures such as [looped transformers]({% post_url 2026-07-02-Looped-Transformers-Computers-and-Length-Generalization %}): computation that does not need to be a token need not be one.

### 3.3 Global structure: reasoning trees

Traces are converted to trees with an adaptation of **LCoT2Tree**:

- **Width $p$:** maximum lateral expansion at any sketch step (branching and exploration).
- **Depth $q$:** index of the furthest occupied sketch step (progress toward a solution).
- **Nodes $N$:** total non-root reasoning nodes.

On MATH (medians with interquartile ranges):

| Model | Median width $p$ | Median depth $q$ | Median nodes $N$ | Mean nodes $N$ |
| :--- | :---: | :---: | :---: | :---: |
| **GPT-6 Astra** | **5.0** [3.0, 9.0] | **11.0** [8.8, 15.0] | **27.0** [17.8, 44.0] | **41.0** |
| GPT-5.6 Sol | 12.0 [5.0, 19.3] | 11.0 [10.0, 13.0] | 60.0 [29.8, 90.3] | 65.9 |
| Claude Opus 4.8 | 22.0 [5.0, 44.0] | 12.0 [9.0, 14.0] | 96.0 [28.5, 220.3] | 144.2 |
| Claude Sonnet 5 | 28.5 [9.0, 49.3] | 13.0 [10.0, 15.0] | 136.0 [44.8, 234.8] | 171.4 |

Depth barely moves (11 to 13). Width and node count move by 5× or more. Every model takes roughly the same number of steps to reach an answer; what differs is how many side branches it opens along the way. Astra commits to a path early and rarely backtracks, while Sonnet 5 builds a wide trial-and-error tree ($p=28.5$, $N=136.0$).

### 3.4 Cognitive repertoire: nothing is skipped

Using seven Schoenfeld categories, the episode mix (% of units) is:

| Model | READ | ANALYZE | PLAN | IMPLEMENT | EXPLORE | VERIFY | MONITOR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Astra | 6.4 | 28.5 | 13.0 | 25.2 | 12.4 | 10.5 | 3.0 |
| Sol | 8.3 | 29.4 | 17.0 | 21.6 | 9.7 | 10.5 | 2.8 |
| Opus 4.8 | 10.8 | 29.0 | 15.6 | 21.2 | 7.5 | 7.8 | 5.3 |
| Sonnet 5 | 7.2 | 31.9 | 13.9 | 22.5 | 6.2 | 10.8 | 4.7 |

Astra's brevity is not from skipping steps. Its relative distribution is close to the others, and its VERIFY share matches Sol's. Over normalized time, every model puts READ in the first 0 to 10% of the trace and VERIFY in the last 80 to 100%. The workflow is the same; Astra just spends fewer characters per episode. (If anything it allocates *more* to EXPLORE, which suggests its exploration is cheap per unit rather than absent.)

---

## 4. Trace Transplants and the Distillation Capacity Gap

Can a dense trace be reused? In **donor-recipient** experiments, a donor's trace (with the final answer removed) is placed in a recipient's context.

- **Strong recipients:** Kimi-K3, GLM-5.2, and GPT-5.6 Luna read Astra's compressed traces almost losslessly, recovering about 90 to 95% accuracy.
- **Weak recipients:** Claude Haiku 4.5 falls from Astra's 93.8% donor baseline to 76.5%. GPT-5.4 Nano falls to 80.0%.

### The exposed-answer paradox

In **83.33%** of Haiku 4.5's errors with Astra traces, the trace *explicitly contained the correct answer*, and Haiku still got it wrong. On HMMT P20 (cylindrical path counting) the Astra trace says "...total per start 1023=2^10-1 times 20 = 20460...". GPT-5.4 Nano answers `10220` (it dropped a term), and Haiku 4.5 answers `5242880`.

The answer was on the page. The recipient could not follow the compressed reasoning that justified it, so it did not trust it and recomputed, badly.

### Why this matters for distillation

The paper offers this as a mechanism for the well-known **teacher-student capacity gap**:

1. Strong teachers like Astra write dense, implicit traces that skip routine steps.
2. Small students lack the latent capacity to reconstruct those skipped steps.
3. So distilling a small model directly on raw frontier CoT can underperform distilling on more explicit traces from a moderately sized teacher.

That is a token-level version of the support-mismatch story in [Revisiting On-Policy Distillation]({% post_url 2026-04-19-On-Policy-Distillation %}) and [The Distributional Lens of Post-Training]({% post_url 2026-06-13-Distributional-Lens-Post-Training %}): if the teacher's distribution puts mass on steps the student cannot generate, imitation breaks. Step-level schemes like [SOPD]({% post_url 2026-08-20-sopd %}) are one way to bridge it. An "expand the teacher trace first" pass is another, cheaper one this paper implies.

---

## 5. Prompt Sensitivity and Provider Boundaries

**Wording matters.** The amount of externalized reasoning depends heavily on the tool description:

- **GPT-5.6 Sol** responds to a *"Maximal Deliberation"* description, going from 1,940 visible tokens (81.3% accuracy) to 3,481 tokens (91.3%).
- **GPT-6 Astra** prefers *"Think-Here"*. Pushing it with "maximal deliberation" makes it keep reasoning in the hidden native channel instead of writing it out.

**Some models refuse.** Claude Opus 5, Fable 5, and Fable 5.1 had a **0% extraction success rate**: they refused the forced tool selection when native thinking was disabled. That looks like a deliberate provider-side boundary, and it is the cleanest mitigation in the paper: if thinking is off, do not let a tool become the thinking channel.

---

## Takeaways

1. **Reasoning controls leak.** Plain tool-choice controls can route around hidden-reasoning restrictions by turning a tool argument into a scratchpad. Providers that care need the kind of refusal the newest Claude models show.
2. **Frontier capability looks parsimonious.** GPT-6 Astra's traces are short, incompressible, and structurally direct: narrow trees, same depth, same episode mix, far fewer words. Capability shows up as *less* visible search, not more. Labs are now training for this directly: [Reflection's Beam]({% post_url 2026-10-06-Reflection-Beam-501B-MoE %}) uses a length penalty whose first phase contracts completions while accuracy rises.
3. **Dense traces are poor supervision for small models.** The exposed-answer paradox shows a trace can contain the answer and still fail to teach it. Expand compressed teacher traces, or pick a more explicit teacher, before distilling into a small student.
