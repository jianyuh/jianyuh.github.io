---
layout: post
title: "Looped Transformers: From Programmable Computers to Length Generalization"
date: 2026-07-02
categories: [LLM, Theory, Architecture]
tags: [Looped Transformers, Recurrence, Length Generalization, RASP-L, In-Context Learning, Adaptive Depth, Attention, Universal Computation]
---

Reading notes on two complementary papers that recast the Transformer as a **recurrent, depth-adaptive computer**:
- [**Looped Transformers as Programmable Computers**](https://arxiv.org/pdf/2301.13196) (Giannou et al.) — what a looped Transformer *can compute* in principle.
- [**Looped Transformers for Length Generalization**](https://arxiv.org/pdf/2409.15647) (Fan et al., 2024) — how to *train* one so its compute depth adapts to problem difficulty.

The first is a constructive existence proof: hand-wire the weights and a constant-depth Transformer, wrapped in a loop, becomes a general-purpose computer. The second turns the same architectural idea into a learning recipe that finally cracks length generalization on algorithmic tasks. Read together, they argue that **decoupling computational depth from architectural depth** is the key to algorithmic reasoning.

---

# Part I — Looped Transformers as Programmable Computers

**Looped Transformers as Programmable Computers** by Giannou et al. fundamentally rethinks the Transformer architecture. Instead of viewing it merely as a statistical sequence-to-sequence model, this paper demonstrates that Transformers can function as general-purpose, programmable computers. By hardcoding specific weights and routing the output sequence back into the input in a simple recursive loop, a shallow, constant-depth Transformer (as few as **13 layers**) can emulate an instruction-set computer, perform linear algebra, and even execute in-context learning algorithms like backpropagation.

## The Architecture: The Input Sequence as a Punchcard

To build a computer out of a Transformer network, the input sequence is structured like a traditional programming **"punchcard"** divided into three distinct sections:

*   **Scratchpad:** A cache-like temporary workspace where data is copied, transformed, and manipulated.
*   **Memory:** The central repository for data structures like scalars, vectors, and matrices.
*   **Instructions:** The commands to be executed, complete with pointers to memory and operation directives.

To manage data locations and a program counter, the model utilizes **binary positional encodings**. Each column/token index $i$ is represented as a $\log(n)$-dimensional vector $p_i \in \{-1, +1\}^{\log(n)}$. This binary format allows standard feed-forward ReLU layers to easily increment pointers and keep track of instructions through simple vector additions.

## Core Mathematical Derivations

**1. Read/Write Operations via Attention Permutation.** The fundamental Transformer block combines attention and feed-forward ReLU layers. The attention layer operates as:

$$\text{Attn}(X) = X + \sum_{i=1}^H V^i X \, \sigma_S\!\left((K^i X)^\top Q^i X\right)$$

*(Note: the paper uses the embedding vectors $X$ as columns.)*

To **read** from memory, the attention mechanism is reverse-engineered to act as a permutation matrix. By configuring the Key ($K$) and Query ($Q$) matrices to isolate the positional encodings ($p_i$), the inner product evaluated by the softmax becomes $p_i^\top p_j$. Because $p_i^\top p_i = \log(n)$ and $p_i^\top p_j \le \log(n) - 1$ for $i \neq j$, passing this through a softmax $\sigma_S$ with a sufficiently high temperature $\lambda \ge \frac{\log n}{\epsilon}$ heavily isolates the target vector. This results in a matrix that perfectly routes the requested memory data to the scratchpad with an arbitrarily small error $\epsilon$.

**2. Conditional Branching (if-goto).** To execute loops and logic, the Transformer implements an `if mem[a] <= 0 then goto p` instruction. By storing integers in memory using a 2's complement binary representation, the model simply needs to check the sign bit (the most significant bit, $b_N$) to determine if a number is negative. A shallow ReLU network computes a "flag" based on this bit:

$$\text{flag} = \text{ReLU}(b_N) + \text{ReLU}\!\left(1 + N - \sum_{i=1}^N b_i\right)$$

The program counter is then updated via the feed-forward layers: if the flag is 1, it jumps to instruction $p$; if 0, it increments the counter by 1.

**3. Matrix Multiplication via Softmax Linearization.** To act as a mathematical engine, the framework introduces **FLEQ** (a generalized instruction set that allows arbitrary function calls). To implement matrix multiplication $A^\top B$ inside a single attention block, the authors leverage a **linearization of the softmax** function. If a column vector $z$ is padded with a deliberately massive constant $C$, the softmax calculation expands to:

$$\sigma_S(c x_{ij}) = \frac{e^{c x_{ij}}}{\sum_{j=1}^n e^{c x_{ij}} + n(e^C + 1)}$$

Using Taylor expansions, for a small scaling factor $c$ and large $C$, the exponential terms mathematically collapse so the softmax output is roughly proportional to $(1 + c x_{ij})$. The subsequent value matrix $V$ and feed-forward networks extract this linear approximation, effectively computing exact dot products and outputting $A^\top B$ in just **2 layers**.

**4. Non-Linear Functions via Barron's Theorem.** For non-linear operations (e.g., square roots, inverses, activations), the Transformer relies on **Barron's Theorem (1993)**, which proves that functions with a bounded Fourier integral can be approximated by a linear combination of sigmoids: $f(x) \approx \sum_{i=1}^m c_i \sigma(x^\top a_i)$. Because the attention softmax naturally computes a sigmoid over inner products, the Key and Query matrices are set to project $a_i^\top x$, and the Value matrix applies the coefficients $c_i$. The approximation error scales predictably as $O(1/\sqrt{m})$, where $m$ is the number of attention heads utilized.

## Insights & Implications

**1. The Necessity of the Loop (Depth vs. Time).** Historically, mapping algorithms to Transformers required the network's depth to scale linearly with the length of the program or the number of computational steps. The fundamental insight of this paper is that by wrapping the transformer in a **single recursive loop** (feeding the output back as the next input step), the depth of the model becomes a constant $O(1)$. A 13-layer transformer can thus run Newton's algorithm, matrix inversion, and Stochastic Gradient Descent simply by operating for $T$ iterations.

**2. Attention is a Universal Compute Substrate.** While the attention mechanism is traditionally viewed as a way to mix contextual information across sequence tokens, this mathematical deconstruction proves it is highly versatile. It can be tightly parameterized to execute exact permutations (memory routing), explicit polynomials (linearization), and bounded non-linear approximations. (For a complementary look at how depth-wise information flow can be generalized in modern LLMs, see [Attention Residuals]({% post_url 2026-03-16-attention-residuals %}).)

**3. Demystifying In-Context Learning in LLMs.** Large Language Models like GPT-3 or PaLM exhibit "in-context learning," performing unseen algorithmic tasks and logic on the fly without any weight updates. Because this paper proves that a small, looped Transformer can execute full backpropagation and gradient descent on implicitly defined neural networks, it strongly suggests a mechanistic hypothesis for LLMs. Modern language models may be implicitly developing recursive, loop-like internal subroutines that act as function calls, effectively reading our natural-language prompts as the "instructions" on an algorithmic punchcard.

---

# Part II — Looped Transformers for Length Generalization

Where Part I *hand-wires* a looped Transformer, **Looped Transformers for Length Generalization** (Fan et al., 2024) asks whether the same recurrent structure can be *learned* — and uses it to attack one of the most stubborn failures in LLMs.

## 1. The Length Generalization Bottleneck

Despite the massive scaling of LLMs in compute and data, these models fundamentally struggle with **length generalization**. If an LLM is trained on arithmetic operations of up to 20 digits, it will almost certainly fail catastrophically when evaluated on 30 or 40 digits.

The core issue stems from the standard Transformer architecture: it processes inputs with a **fixed depth**. While techniques like Chain-of-Thought (CoT) and scratchpads allow for elastic computation (generating more intermediate tokens for harder problems), these approaches are still constrained by the fixed depth of the underlying model and require hard-to-acquire intermediate supervision data.

To break this bottleneck, this paper proposes **Looped Transformers** — a recurrent architecture where the same decoder block is iteratively applied to the sequence. By decoupling the computational depth from the architectural depth, the model can dynamically adjust its compute steps based on the difficulty (length) of the problem.

## 2. Theoretical Framework: From RASP-L to n-RASP-L

To formally ground the capabilities of Transformers, the authors rely on **RASP-L** (Restricted Access Sequence Processing), a programming-language model designed to map directly to learnable decoder-only Transformer operations. RASP-L is built on element-wise operations and causal attention simulations, prohibiting arbitrary index arithmetic and control-flow loops.

However, fixed-depth RASP-L programs cannot represent operations whose required computational steps scale with the input length (e.g., addition, parity). The authors solve this by formally defining **n-RASP-L**:

**Definition 3.1 (n-RASP-L).** A program $P$ is an n-RASP-L program if:

1.  There exists a function $T: \mathbb{N} \rightarrow \mathbb{N}$ (determining step count based on length $n$).
2.  $P$ can be decomposed into a sequential application of a base program $P'$ for $T(n)$ steps, flanked by preprocessing and postprocessing steps:

$$P = P_{pre} \circ (P')^{T(n)} \circ P_{post}$$

*(where $P', P_{pre}, P_{post} \in \text{RASP-L}$).*

### Derivations of T(n) for Classic Algorithmic Tasks

The authors provide mathematical proofs showing that standard tasks naturally decompose into n-RASP-L loops:

*   **n-bit Parity ($T(n) = n$):** The loop $P'_{parity}$ shifts the input right by 1 and calculates the XOR of the current answer sequence and the input sequence.
*   **n-symbol Copy ($T(n) = n$):** The loop $P'_{copy}$ simply shifts the sequence to the right by 1 at each step.
*   **n-digit Addition ($T(n) = n + 1$):** The loop $P'_{addition}$ calculates the XOR of two sequences (shifting the result right by 1 as the partial answer) and computes the AND as the carry-on sequence.

## 3. The Looped Transformer Architecture

Unlike Universal Transformers which use both encoders and decoders, this architecture strictly uses a **decoder-only block** applied recurrently.

**Key Architectural Details:**

*   **Recurrence:** A single decoder block (consisting of a set number of layers) is reused for all looped steps.
*   **Input Injection:** At each step, the original input sequence embeddings are added to the output embeddings of the *previous* step. This acts as a residual connection to the original problem, preventing information decay over many loops.
*   **No Positional Embeddings (NoPE):** Because RASP-L operations inherently do not use positional encodings, the architecture drops them entirely, relying purely on relative positional relationships learned via the attention mechanism.
*   **Full-Output Prediction (FOP):** Instead of Next-Token Prediction (NTP), the model uses FOP. The input is the query, and the missing answers are padded with EOS tokens. The model predicts *all* missing tokens concurrently after processing all internal loops, shifting away from token-by-token autoregression.

## 4. The Mathematical Formulation of Training

A highly novel insight of this paper is training a recurrent model to learn intermediate algorithmic steps **without providing ground-truth intermediate (CoT) labels**.

**The Setup.** The dataset consists of input-output pairs $(X, Y)$ and the known number of steps $T(n)$ required to solve that specific length. The training loss minimizes the cross-entropy $L$ after strictly $T_i$ steps of the loop $M_\theta$:

$$\mathbb{E}_{D}\left[ L\left(f_{T_i}(M_\theta, \{(x_l)_{l=1}^{L_i}\}_i), \{(y_l)_{l=1}^{L_i}\}_i\right) \right]$$

*where $f_{T_i}(M_\theta, X) = M_\theta(M_\theta(\dots M_\theta(X)))$ applied $T_i$ times.*

**Insight on Implicit Step-Learning.** Why does this work without intermediate labels? If a problem of length $n$ takes $T$ steps to output the right answer, a sub-problem of length $n-1$ will map to the exact same decoder block after $T-1$ steps. Consequently, the shared decoder block is forced to learn generalized intermediate representations that satisfy variable lengths at variable steps.

## 5. Adaptive Inference: Knowing When to Stop

At inference time, the model must dynamically adapt its depth. The authors define two stopping criteria:

1.  **Oracle:** The step count $T(n)$ is provided.
2.  **Maximum Confidence:** The model halts when the output sequence achieves the lowest cross-entropy loss over a batch of test sequences. Mathematically:

$$T = \arg\min_{t \in [1, T_{max}]} L\left(f_t(M_\theta, \{(x_l)_{l=1}^L\}_{i=1}^B), \{(\hat{y}_l^t)_{l=1}^L\}_{i=1}^B\right)$$

*(where $\hat{y}$ is the decoded sequence at step $t$).*

## 6. Empirical Results and Domain Insights

The empirical results are striking. When trained on lengths up to 20, standard Next-Token Prediction (NTP) fails entirely when tested on length 30. The adaptive **FOP-Loop** model achieves near-perfect length generalization up to **40 digits** on Parity, and maintains near-perfect accuracy on Addition and Copy.

**Critical Insights from the Ablations:**

*   **Input Injection is Non-Negotiable:** Ablations removing input injection ("FOP-Loop-Adaptive-WO") show a significant degradation in tasks like Addition and Binary Sum. Keeping the original context continuously accessible is critical for long-horizon loops.
*   **Implicit Convergence:** Even though the model is not explicitly trained to converge and stop, the loss curves on tasks like Addition, Copy, and Multiplication naturally smooth out and converge to stable outputs after solving the task.
*   **Pause Tokens Are Insufficient:** While adding "pause tokens" (NTP-Pause / FOP-Pause) allows the model to compute extra parallel states before outputting, they strictly increase *horizontal* compute. They fail to match the power of true *sequential* adaptive depth.

### Limitations & The Path Forward

While revolutionary for simple algorithmic tasks, the n-RASP-L framework currently does not formally support complex algorithms requiring multiple, distinct, nested loops. Furthermore, the step-dependent training requires $T(n)$ to be pre-calculated and injected into the training data — an assumption easier to satisfy than dense CoT labels, but still an extra requirement compared to pure end-to-end learning.

---

## Putting It Together

The two papers are two halves of the same argument:

1.  **Expressivity (Giannou et al.):** A constant-depth Transformer *looped* over time is Turing-complete in practice — it can route memory, branch, multiply matrices, and approximate non-linearities. Depth is not the resource that matters; **iteration count is**.
2.  **Learnability (Fan et al.):** That same depth-vs-time decoupling, when *trained* with input injection, NoPE, and full-output prediction against a known step schedule $T(n)$, is exactly what lets a model generalize to lengths far beyond its training distribution.

The shared lesson: **the loop is the point.** Fixed-depth Transformers conflate "how big the model is" with "how long it gets to think." Looping separates them — Giannou et al. show the ceiling this unlocks, and Fan et al. show how to climb toward it by learning. Both also sharpen the mechanistic reading of in-context learning: if recurrent, loop-like subroutines are what execute algorithms, then scaling *iteration* (test-time compute) may matter as much as scaling *parameters*. For how this architectural lineage fits the broader evolution of LLM design, see [LLM Architecture Evolution]({% post_url 2025-12-05-LLM-Arch %}).
