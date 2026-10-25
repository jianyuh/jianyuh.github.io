---
layout: post
title: "Codex and HumanEval: pass@k, Functional Correctness, and the Origins of Code LLMs"
date: 2026-10-25
categories: [LLM, Evaluation]
tags: [Codex, HumanEval, PassAtK, CodeGeneration, FunctionalCorrectness, Sampling, SFT, Copilot, Alignment, ScalingLaw]
---

Reading notes on:
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/pdf/2107.03374)

Almost every coding result reported today, from a frontier model card to an [RLVR paper]({% post_url 2025-11-22-RLVR-Limit %}) arguing about whether RL expands what a base model can do, is phrased in a vocabulary this paper introduced: **pass@k on unit tests**. The Codex paper is remembered for the model behind GitHub Copilot, but its longer-lived contributions are the evaluation choices around it: score code by *running* it, release a hand-written benchmark (**HumanEval**, 164 problems) so the score is not contaminated by GitHub, and estimate pass@k without bias.

The headline numbers: on HumanEval, GPT-3 solves **0%**, GPT-J 6B solves **11.4%**, and Codex-12B solves **28.8%** with a single sample. A further supervised fine-tune on standalone functions, **Codex-S**, reaches **37.7%**. With 100 samples per problem and the unit tests as an oracle, Codex-S solves **77.5%**; without the oracle, picking the sample with the highest mean log-probability still gets **44.5%**. Repeated sampling turns out to be a strong lever, which is the first clear statement of the test-time scaling story that later grew into [best-of-k and multi-agent search]({% post_url 2026-09-23-Test-Time-Multi-Agent-Communication-Discovery %}).

![Panel A: the Codex model lineage, from a GPT-3 family checkpoint fine-tuned on 159 GB of GitHub Python into Codex, then supervised fine-tuning on about 50K curated standalone functions into Codex-S, and a reversed-order fine-tune into Codex-D; HumanEval pass@1 for the 12B models is shown on each card. Panel B: Codex pass@1, pass@10 and pass@100 on HumanEval from Table 1, plotted against parameter count from 12M to 12B on a log axis, with GPT-J 6B marked for comparison](/assets/images/codex_humaneval.svg)

---

## 1. Why Functional Correctness, Not BLEU

Before Codex, code generation was mostly scored by matching a sample against a reference solution, exactly or fuzzily (BLEU). The paper's argument against this is simple: the space of programs that are functionally equivalent to a reference is huge, and match-based metrics cannot see it. It mirrors how developers already judge code. Test-driven development turns requirements into tests before any implementation, and code is merged when it passes them.

The paper backs this up empirically. For Codex-12B samples on HumanEval at temperature 0.8, the BLEU distributions of correct and incorrect solutions overlap heavily on each problem. An incorrect solution is *guaranteed* to disagree with the reference on some input, yet it often scores a higher BLEU than a correct one. So improving BLEU need not improve correctness.

Executing model-written code against tests means running untrusted code. The paper's sandbox uses the **gVisor** container runtime as the main host protection (it emulates host resources to put a security boundary between host and container) plus **eBPF firewall rules** that block inbound and outbound connections except for experiment control. Five years later, the sandbox is the load-bearing layer of agentic RL; see [DeepSeek DSec]({% post_url 2026-09-25-DeepSeek-DSec-Sandbox-Infrastructure-Agentic-RL %}) for what that layer looks like at 380K live sandboxes.

---

## 2. The pass@k Metric and Its Unbiased Estimator

Following Kulal et al., a problem counts as solved under pass@k if *any* of $k$ samples passes the unit tests. Generating exactly $k$ samples per problem and checking them gives a high-variance estimate. Instead, generate $n \ge k$ samples (the paper uses $n = 200$, $k \le 100$), count the $c \le n$ correct ones, and compute

$$\text{pass@}k := \mathop{\mathbb{E}}_{\text{Problems}}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$

The fraction is the probability that $k$ samples drawn **without replacement** from the $n$ are all wrong. Since $c \sim \mathrm{Binom}(n, p)$ with $p$ the true pass@1, Appendix A shows the expectation collapses to exactly $1 - (1-p)^k$:

$$\mathbb{E}_c\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right] = 1 - \sum_{i=0}^{n-k}\binom{n-k}{i}p^i(1-p)^{n-i} = 1 - (1-p)^k$$

The tempting shortcut, plugging the empirical pass@1 $\hat{p}$ into $1 - (1-\hat{p})^k$, is consistent but **biased low**, and the gap does not fully close even when $n > 5k$. It corresponds to drawing $k$ samples *with* replacement from a pool of $n$, which makes them dependent. Using it would make results look better simply by drawing more samples.

Computing the binomials directly overflows, so the paper gives a numerically stable product form:

```python
def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@k
    """
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

This snippet, shipped with the human-eval repo, is what most later eval harnesses copied. The [layered evaluation stack]({% post_url 2026-06-12-LLM-Evaluation-Architecture %}) of a modern lab would look very different without it.

---

## 3. HumanEval

HumanEval is **164 hand-written programming problems**. Each has a function signature, a docstring, a reference body, and on average **7.7 unit tests**. The problems test language comprehension, reasoning, algorithms, and simple math, some at the level of easy interview questions.

Hand-writing matters because the training set is a large fraction of GitHub, which already contains solutions to problems from many public sources. For example, more than ten public repositories hold solutions to the Codeforces problems that make up part of APPS. The three examples in the paper's Figure 2 show the difficulty spread: a single Codex-12B sample passes with probability 0.9, 0.17, and 0.005.

The prompt is a header, the signature, and the docstring. Sampling stops at any of `\nclass`, `\ndef`, `\n#`, `\nif`, or `\nprint`, since the model would otherwise keep writing more functions or statements. All sampling uses nucleus sampling with top-p 0.95.

---

## 4. Code Fine-Tuning: Codex

### 4.1 Data

The corpus was collected in May 2020 from **54 million public GitHub repositories**: 179 GB of unique Python files under 1 MB. Filtering removed files that were likely auto-generated, had an average line length above 100 or a maximum above 1000, or had a small share of alphanumeric characters. The final dataset is **159 GB**.

### 4.2 Training

Codex fine-tunes GPT models up to 12B parameters. Two details are worth noting:

- **Starting from GPT-3 did not improve final quality**, probably because the fine-tuning set is so large, but it did make training **converge faster**, so all experiments start from GPT.
- Training uses the same learning rate as the corresponding GPT model, a 175-step linear warmup, cosine decay, **100B tokens**, Adam with $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$, and weight decay 0.1.

The tokenizer is the GPT-3 text tokenizer plus **extra tokens for runs of whitespace** of different lengths. Whitespace is the biggest source of waste when encoding code with a text tokenizer, and these tokens cut the token count by about **30%**. (See the [tokenizer notes]({% post_url 2025-12-06-Tokenizer %}) for why GPT-2 style BPE handles indentation so poorly.)

### 4.3 Scaling

Test loss on held-out GitHub Python follows a power law in model size, just as GPT-3's does (see [Scaling Law]({% post_url 2024-12-23-scaling-law %})):

$$L(N) = \left(\frac{N}{5.92 \times 10^{7}}\right)^{-0.13}$$

where $N$ is the number of non-embedding parameters. With temperatures tuned per $k$, pass@1 and pass@100 grow smoothly too, roughly as a **sigmoid in log-parameters**.

### 4.4 Temperature depends on k

The best sampling temperature grows with $k$. Larger $k$ rewards *diversity*, because the metric only asks whether any sample is correct. For a 679M model the optimum is $T^* = 0.2$ for pass@1 and $T^* = 0.8$ for pass@100. The bottom panel of the paper's Figure 5 traces this as the upper hull of the per-temperature curves. Reporting pass@1 and pass@100 at a single shared temperature understates one of them.

### 4.5 Picking one sample without an oracle

pass@k assumes an oracle that already knows the unit tests. In autocomplete there are no tests, and the tool should show the user one completion, not a hundred. With $k$ samples at $T = 0.8$ from Codex-12B:

- **Highest mean token log-probability** beats picking a random sample.
- **Highest sum log-probability** can do slightly *worse* than random.
- **Back-translation** with the docstring model (Section 6) beats random but trails mean log-probability, and overfits quickly.

### 4.6 Comparison with other models

| Model | pass@1 | pass@10 | pass@100 |
| :--- | ---: | ---: | ---: |
| GPT-Neo 125M | 0.75% | 1.88% | 2.97% |
| GPT-Neo 1.3B | 4.79% | 7.47% | 16.30% |
| GPT-Neo 2.7B | 6.41% | 11.27% | 21.37% |
| GPT-J 6B | 11.62% | 15.74% | 27.74% |
| TabNine | 2.58% | 4.35% | 7.59% |
| Codex-12M | 2.00% | 3.62% | 8.58% |
| Codex-25M | 3.21% | 7.1% | 12.89% |
| Codex-42M | 5.06% | 8.8% | 15.55% |
| Codex-85M | 8.22% | 12.81% | 22.4% |
| Codex-300M | 13.17% | 20.37% | 36.27% |
| Codex-679M | 16.22% | 25.7% | 40.95% |
| Codex-2.5B | 21.36% | 35.42% | 59.5% |
| Codex-12B | **28.81%** | **46.81%** | **72.31%** |

GPT-Neo and GPT-J are trained on The Pile, which is about 8% GitHub code. Even that much code goes a long way: GPT models of comparable size score near 0%. Data specialization buys a lot of parameter efficiency, though. GPT-Neo-2.7B roughly matches **Codex-85M (30× fewer parameters)**, and GPT-J-6B roughly matches **Codex-300M (20× fewer)**. The largest free TabNine model, a commercial autocomplete system, sits near Codex-12M.

A small inconsistency in the source: the abstract and introduction quote GPT-J at 11.4% pass@1, while Section 3.4 and Table 1 give 11.6% (11.62%). The table above uses the Table 1 values.

### 4.7 APPS

APPS has 5000 training and 5000 test problems, mostly full programs that read stdin and write stdout, a poor match for Codex's function-level training. The paper reports only "strict accuracy" and adds two contest conventions:

- **Filtered pass@k:** sample 1000 solutions, keep those passing the 3 public input/output examples in the task statement, and compute pass@k over the filtered set.
- **Timeouts:** report separately the solutions that fail no test but time out on some (3-second limit).

To make up for not fine-tuning on APPS, one input/output example is appended to the docstring as a formatting hint ("1-shot"). Codex-12B values below; numbers in parentheses also count the passing-but-timed-out programs. Sampling is at $T = 0.6$ for every $k$, so raw pass@1 could be higher with a lower temperature.

| Setting | Introductory | Interview | Competition |
| :--- | ---: | ---: | ---: |
| GPT-Neo 2.7B (fine-tuned) raw pass@1 | 3.90% | 0.57% | 0.00% |
| GPT-Neo 2.7B (fine-tuned) raw pass@5 | 5.50% | 0.80% | 0.00% |
| 1-shot Codex raw pass@1 | 4.14% (4.33%) | 0.14% (0.30%) | 0.02% (0.03%) |
| 1-shot Codex raw pass@5 | 9.65% (10.05%) | 0.51% (1.02%) | 0.09% (0.16%) |
| 1-shot Codex raw pass@100 | 20.20% (21.57%) | 2.04% (3.99%) | 1.05% (1.73%) |
| 1-shot Codex raw pass@1000 | 25.02% (27.77%) | 3.70% (7.94%) | 3.23% (5.85%) |
| 1-shot Codex filtered pass@1 | 22.78% (25.10%) | 2.64% (5.78%) | 3.04% (5.25%) |
| 1-shot Codex filtered pass@5 | 24.52% (27.15%) | 3.23% (7.13%) | 3.08% (5.53%) |

1-shot Codex-12B is comparable to a GPT-Neo fine-tuned on APPS. Filtered pass@1, which uses the public tests as a cheap verifier, beats raw pass@100. On harder problems, many solutions are correct but too slow.

---

## 5. Supervised Fine-Tuning: Codex-S

GitHub Python includes classes, config files, scripts, and data files, all far from "write a function from its docstring." Codex-S closes that distribution gap with a set of *correct standalone functions* drawn from two sources:

1. **Competitive programming and interview-prep sites (10,000 problems).** Problem statements become docstrings. Since full test suites are usually hidden, unit tests are built from the examples in the statement or extracted by submitting incorrect solutions.
2. **Continuous integration tracing (about 40,000 problems).** For GitHub repos using travis or tox, plus PyPI packages, the pipeline follows the CI config to set up the environment and run integration tests, using `sys.setprofile` to record inputs and outputs of **every function called**. Those records become unit tests. Builtins and library calls are traced too, so these problems look like the building blocks of command-line utilities: less about algorithms, more about following the spec. Only about 40,000 came out of millions of candidates, since many functions do not take inputs and return outputs, and most runtime objects cannot be pickled and restored outside the sandbox.

**Filtering with the model itself.** Some curated prompts underspecify the function, so a valid solution could fail the tests; some are stateful. The fix: generate 100 samples per problem with Codex-12B and **drop any problem where none pass**, treating it as ambiguous or too hard. The check is rerun several times to remove stateful or nondeterministic tasks. The same "drop the zero-pass prompts" filter now shows up in nearly every RL data pipeline; [Never Give Up]({% post_url 2026-09-19-Never-Give-Up-Adaptive-Sampling-Hard-Problems %}) is about what that filter leaves on the table.

**Training.** Problems use the HumanEval format. Shorter prompts in a batch are **left-padded** so the first solution tokens line up in context. The loss is NLL on the reference solution with **prompt tokens masked out**. Learning rate is 1/10 of the Codex fine-tune with the same schedule, run until validation loss plateaus (under 10B tokens).

**Results.**

- Codex-S prefers slightly **higher temperatures** for all $k > 1$, likely because it models a narrower distribution than Codex. The paper uses $T^* = 0$ for pass@1 and $T^* = 1$ for pass@100. This is the SFT mode-sharpening effect described in [The Distributional Lens of Post-Training]({% post_url 2026-06-13-Distributional-Lens-Post-Training %}), measured as a shift in the best temperature.
- Averaged over model sizes, Codex-S beats Codex by **6.5 points on pass@1** and **15.1 points on pass@100**, and is one to two orders of magnitude more parameter-efficient.
- Ranking 1 to 100 Codex-S-12B samples by mean log-probability beats random ranking by **11.6 points** on average, over 2 points more than the same gain for Codex.

---

## 6. Docstring Generation: Codex-D

Code-to-docstring is motivated by safety: a model that can describe the intent of generated code. The training examples reuse the Codex-S problems with the order flipped (signature, reference solution, then docstring), and the NLL is on the docstring.

There is no unit test for a docstring, so samples are **graded by hand**: correct if the docstring uniquely and accurately specifies the body. That is 10 samples per problem, 1640 in total, from Codex-D-12B at $T = 0.8$. Copying the body into the docstring counts as wrong; incorrect unit tests generated along with it are ignored. The common failures are leaving out an important detail ("answer must be to two decimal places") and over-conditioning on the function name, inventing a problem that has nothing to do with the body.

| Model | pass@1 | pass@10 |
| :--- | ---: | ---: |
| Codex-S-12B | 32.2% | 59.5% |
| Codex-D-12B | 20.3% | 46.5% |

Codex-D is lower than Codex-S but comparable. The paper has no strong hypothesis for which direction should be easier, and notes that docstrings in the data may be low quality because developers spend less time on them; the model writes things like "I just found this function online" and "This test is not correctly written and it's not my solution."

---

## 7. Limitations

- **Sample inefficiency.** The training set is hundreds of millions of lines of code, far more than any developer sees in a career, yet a strong student finishing an intro CS course should solve more HumanEval problems than Codex-12B.
- **Long chains of operations.** Using 13 deterministic string-manipulation building blocks (for example "convert the string to lowercase", "remove every third character from the string"), the paper builds synthetic docstrings by concatenating $m$ blocks. Pass rate drops by a factor of roughly **2 to 3 for each extra component**, which is exponential decay in chain length. A human who can implement a chain of two can implement a chain of any length.
- **Binding operations to variables.** As in text-to-image models that misbind attributes to objects, Codex mixes up which operation applies to which variable when there are many of each:

```python
def do_work(x, y, z, w):
    """ Add 3 to y, then subtract 4
    from both x and w. Return the
    product of the four numbers. """
    t = y + 3
    u = x - 4
    v = z * w
    return v
```

Codex-12B never decrements `w` and does not return the product of all four numbers.

- Codex also suggests syntactically invalid or undefined code and calls functions, variables, and attributes outside the codebase's scope, and it gets worse as specifications become longer and more system-level.

---

## 8. Broader Impacts

The hazard analysis is long; the parts that aged most interestingly:

**Misalignment, defined operationally.** A model is misaligned on task X if it is *capable* of X but does not do it. A model that fails because it cannot is merely incompetent. The experiment: put subtle bugs in the prompt's example code, and Codex writes worse code than it can. Adding an instruction to write correct code helps a little but does not fix it, and **the gap grows with model size**. The model continues confused code with confused code, insecure code with insecure code. The paper warns that a capable, misaligned model trained on user approval might write obfuscated code that looks good even on careful inspection.

**Over-reliance and automation bias.** Codex can produce code that looks right but is not. Human oversight is required, and the paper expects automation bias to get harder to guard against as models improve. That concern reads very differently in the era of [Claude Code]({% post_url 2026-02-25-Claude %}), where most of the code in a session is model-written.

**Security.** Codex picks clearly insecure crypto settings in a large fraction of cases (RSA keys under 2048 bits, AES in ECB mode). The authors' assessment is that it does not materially lower the barrier to writing malware at its capability level, though it can write components (for example, recursively encrypting files in a directory) and performed poorly next to even basic SAST tools for vulnerability discovery. Sampling nondeterminism could help polymorphic malware.

**Memorization and legal.** Generations identical to training data occurred in **under 0.1%** of cases in a cited study, and when they did, they were common idioms that appear over and over in the training data.

**Economics and environment.** Engineers do not spend their whole day writing code, which limits near-term productivity gains. Codex imports some packages more often than others, which could favour some package authors. Fine-tuning GPT-3-12B into Codex-12B took about as much compute as the original pretraining, hundreds of petaflop/s-days, and the paper flags that inference for hard problems could eventually dwarf training compute, a prediction that the [reasoning-effort era]({% post_url 2026-07-19-Reasoning-Effort-Inference-Scaling %}) has borne out.

---

## 9. Takeaways

1. **Score code by running it.** Functional correctness on hidden unit tests replaced BLEU, and BLEU is shown to be a poor proxy: wrong programs often outscore correct ones.
2. **pass@k needs the unbiased estimator.** Draw $n \ge k$ samples and use $1 - \binom{n-c}{k}/\binom{n}{k}$; the plug-in $1 - (1-\hat{p})^k$ is biased low.
3. **Temperature must be tuned per $k$.** Higher $k$ wants more diversity ($T^* = 0.2$ vs 0.8 for a 679M Codex). SFT narrows the distribution and shifts the optimum up.
4. **Sampling is a scaling axis.** 28.8% at pass@1 becomes 72.3% at pass@100 for Codex-12B, and 77.5% for Codex-S. Without an oracle, mean log-probability ranking captures part of it (44.5%).
5. **Match the training distribution to the task.** Roughly 50K curated standalone functions (before filtering) for Codex-S add 6.5 points of pass@1 and 15.1 points of pass@100 on average. Filtering the training set with the model's own pass rate is now standard practice.
6. **The failure modes are compositional.** Pass rate falls 2 to 3× per chained operation, and operations get bound to the wrong variables, both still recognizable in modern code models on long specifications.
7. **Misalignment can grow with scale.** Buggy context degrades output more for larger models, an early, measurable instance of "capable but does not."

The road from here runs through [SWE-RL self-play]({% post_url 2025-12-26-Self-Play-SWE-RL %}) and agentic coders like [SWE-1.7]({% post_url 2026-07-10-SWE-1.7 %}), where the unit of evaluation is a repository and a test suite rather than a single function and a docstring, but pass@k and the execution sandbox carried over unchanged.
