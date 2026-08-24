---
layout: post
title: "CUDA Agent: Agentic RL for High-Performance CUDA Kernel Generation"
date: 2026-03-03
categories: [Systems, CUDA]
tags: [CUDA, KernelBench, RL, Agentic, OpenHands, GPU Kernels]
---

Reading notes on:
- [CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA Kernel Generation](https://arxiv.org/pdf/2602.24286v1)

CUDA kernel optimization sits in an awkward gap between compiler heuristics and human wizardry. Static systems like `torch.compile` already do well on routine patterns, but once the workload becomes fused, irregular, or tightly constrained by registers, shared memory, and launch overhead, the search space stops looking like ordinary code generation and starts looking like systems research. CUDA Agent's claim is that this gap can be attacked directly with a tool-using RL agent: a **23B/230B MoE** model trained to inspect code, compile kernels, benchmark them on real GPUs, and keep iterating until the implementation is both correct and faster than the baseline.

That makes it a concrete instance of the broader shift from "generate code" to "optimize inside an execution harness." Here the harness is narrow and brutally measurable: write CUDA, run correctness checks, profile it, and get rewarded only if the kernel actually wins.

![CUDA Agent system overview: synthetic fused tasks, skill-integrated agent loop, and stable RL warm-up](/assets/images/cuda_agent_system_overview.svg)

---

## 1. The Three-Part Design

CUDA Agent works because three pieces are co-designed rather than bolted together.

### Data synthesis: CUDA-Agent-Ops-6K

High-performance CUDA training data is scarce, so the paper builds a task set instead of waiting to collect one. The pipeline starts from PyTorch and `transformers` operators, wraps each as a standalone `torch.nn.Module`, then asks an LLM to fuse **1 to 5** primitives into a single operator. That is the important move: the model is not trained only on textbook kernels, but on compositions where the whole point is to remove intermediate writes and re-map work across registers, shared memory, and thread layouts.

The resulting tasks are filtered aggressively:
- they must run in both eager and compiled PyTorch,
- they must be deterministic,
- they must not admit trivial constant-output hacks, and
- their eager runtime must land between **1 ms and 100 ms**.

A final AST-based decontamination pass drops anything too close to the KernelBench test set, using a maximum similarity threshold of **0.9**. The final corpus is **CUDA-Agent-Ops-6K**, a dataset meant to teach the model fusion pressure rather than isolated operator syntax.

### Skill-integrated agent loop

At inference and training time, the model does not emit a kernel in one shot. It operates in a ReAct-style loop with bash, file editing, and code search tools inside an OpenHands-style environment. The workflow is effectively a domain-specific `SKILL.md`:
1. Profile the PyTorch baseline.
2. Write `model_new.py`, `binding.cpp`, and CUDA kernels.
3. Compile and verify correctness.
4. Iterate until the custom path beats `torch.compile` by at least **5%**.

This is the same broad systems pattern as [Self-Play SWE-RL: Superintelligent Agents via Autonomous Bug Discovery]({% post_url 2025-12-26-Self-Play-SWE-RL %}), except the evaluator is much tighter. The agent is not rewarded for plausible code or passing unit tests alone; it is rewarded for physically better execution on GPUs.

### CPU-GPU decoupled sandboxing

The evaluation harness splits responsibilities across two tiers:
- a CPU-side Docker host handles compilation, terminal control, and string-level checks;
- a GPU worker pool of **128 NVIDIA H20s** runs correctness and profiling under isolated allocations.

That separation matters. It makes the timing loop reproducible enough to train against, and it closes off the obvious reward-hacking path where an agent tampers with the evaluator instead of improving the kernel.

---

## 2. Rewards Have to Be Discrete and Defensive

A naive reward like `t_compile / t_generated` sounds natural, but it is badly biased. Easy tasks can produce huge ratios for unimportant wins, while hard but meaningful fusion problems may yield only modest numeric gains. CUDA Agent replaces that with milestone-based rewards:

$$
r = \begin{cases}
-1 & \text{if numerical correctness check fails} \\
3 & \text{if } b(t, t_{eager}) \land b(t, t_{compile}) \\
2 & \text{if } b(t, t_{eager}) \\
1 & \text{otherwise}
\end{cases}
$$

with

$$
b(t, t_0) = \mathbb{I}\left[\frac{t_0 - t}{t_0} > 5\%\right].
$$

So the agent only gets the top reward if it clears a real speed threshold against both eager and compiled baselines. That removes a lot of timing noise from the learning signal.

The sandbox adds five guardrails:
- read-only verification and profiling scripts,
- interception of sneaky fallbacks to `torch.nn.functional`,
- correctness checks on **5** random inputs,
- synchronized warmups and repeated measurements, and
- no web access inside the agent environment.

This is the part I found most convincing. CUDA Agent is not just "RL for coding"; it is RL with an evaluator designed to make cheating harder than real optimization.

---

## 3. Why PPO Collapses on CUDA

The paper reports that straightforward multi-turn PPO training collapses after **17 steps**. The reason is not mysterious once you see the domain mismatch: high-performance CUDA is far outside the model's pretraining comfort zone, representing **less than 0.01%** of typical data.

That means the policy must put mass on very rare token sequences, sometimes at probabilities near the numerical floor, e.g. $\pi_\theta(a_t \mid s_t) \approx 10^{-9}$. In that setting, even small train-rollout precision differences can explode the importance ratio

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta^{old}}(a_t \mid s_t)}.
$$

That failure mode rhymes with the precision-mismatch story in [Jet-RL and the Precision Mismatch in Reasoning Models]({% post_url 2026-01-26-FP8-RL %}) and the stability analysis in [First-Order Approximation for Stable LLM-RL Training]({% post_url 2025-12-02-Stabilize-RL %}): once probabilities live near the floor, "small" numerical differences stop being small from PPO's point of view.

---

## 4. The Warm-up Strategy

The fix is a staged initialization scheme rather than brute-force online RL from the base model.

### Step 1: single-turn warm-up

The model first learns a simpler PyTorch-to-CUDA mapping task with standard single-turn RL. This does not solve the full agent problem, but it moves the policy away from the worst part of the distribution gap.

### Step 2: actor warm-up via rejection fine-tuning

The warmed single-turn model is used to collect multi-turn agent trajectories. Those trajectories are filtered twice:
- keep only runs with positive final reward,
- reject looping or malformed tool-use behavior.

The actor is then initialized by supervised training on the retained traces:

$$
\mathcal{L}_{RFT}(\theta) = -\mathbb{E}_{\tau \sim \mathcal{D}'} \left[ \sum_{t=1}^{T} \log \pi_\theta(a_t \mid s_t, a_{<t}) \right].
$$

This gives PPO a sane starting policy instead of asking it to invent stable tool-use structure online.

### Step 3: critic warm-up via value pretraining

The critic is pretrained on the collected trajectories with GAE-style targets:

$$
V_t^{target} = V_\phi(s_t) + \hat{A}_t,
\qquad
\hat{A}_t = \sum_{l=0}^{T-1-t} (\gamma \lambda)^l \delta_{t+l},
$$

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t).
$$

The paper uses $\gamma = 1$, $\lambda = 0.95$, and $V_\phi(s_T) = 0$. The critic minimizes

$$
\mathcal{L}_{VP}(\phi) = \frac{1}{2}\mathbb{E}_{\tau \sim \mathcal{D}} \left[ \frac{1}{T} \sum_{t=0}^{T-1} \left( V_\phi(s_t) - V_t^{target} \right)^2 \right].
$$

Only after both warm-ups does the model run multi-turn PPO with the clipped objective

$$
\mathcal{L}_{CLIP}(\theta) = \mathbb{E}_{\tau \sim \mathcal{D}} \left[ \frac{1}{T} \sum_{t=0}^{T-1} \min \left( \rho_t(\theta)\hat{A}_t, \text{clip}\left(\rho_t(\theta), 1-\epsilon_{lower}, 1+\epsilon_{higher}\right)\hat{A}_t \right) \right]
$$

using $\epsilon_{lower} = 0.2$, $\epsilon_{higher} = 0.28$, a global batch size of **1024**, and a stable **150-step** run. In domains this sparse and this sharp-edged, the hard part is not just exploration but initializing a policy and critic that do not immediately destabilize each other.

---

## 5. Results on KernelBench

CUDA Agent is evaluated on **250 KernelBench tasks** across Level 1 to Level 3, under the same interactive agent loop used for the baselines.

| Difficulty Split | Metric | Seed 1.6 (Base) | GLM 4.6 | Kimi K2 | Gemini 3 Pro | Claude Opus 4.5 | **CUDA Agent** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall (Weighted)** | **Pass Rate**<br>**Faster vs. Compile**<br>**Speed-up vs. Compile** | 74.0%<br>27.2%<br>0.69x | 75.6%<br>19.2%<br>0.57x | 66.8%<br>22.8%<br>0.66x | 91.2%<br>69.6%<br>1.42x | 95.2%<br>66.4%<br>1.46x | **98.8%**<br>**96.8%**<br>**2.11x** |
| **Level 1** | **Pass Rate**<br>**Faster vs. Compile**<br>**Speed-up vs. Compile** | 90.0%<br>51.0%<br>1.25x | 86.0%<br>32.0%<br>0.73x | 85.0%<br>39.0%<br>1.00x | 95.0%<br>72.0%<br>1.51x | 96.0%<br>72.0%<br>1.54x | **100.0%**<br>**97.0%**<br>**1.87x** |
| **Level 2** | **Pass Rate**<br>**Faster vs. Compile**<br>**Speed-up vs. Compile** | 74.0%<br>16.0%<br>0.50x | 76.0%<br>11.0%<br>0.42x | 65.0%<br>15.0%<br>0.65x | 93.0%<br>76.0%<br>1.46x | 98.0%<br>69.0%<br>1.60x | **100.0%**<br>**100.0%**<br>**2.80x** |
| **Level 3** | **Pass Rate**<br>**Faster vs. Compile**<br>**Speed-up vs. Compile** | 42.0%<br>2.0%<br>0.40x | 54.0%<br>10.0%<br>0.62x | 34.0%<br>6.0%<br>0.29x | 80.0%<br>52.0%<br>1.17x | 88.0%<br>50.0%<br>1.10x | **94.0%**<br>**90.0%**<br>**1.52x** |

Two numbers matter most. First, the model is not merely correct; it is usually **faster than compile**, which is the whole point. Second, the gap widens on sequence-style fused workloads, where static heuristics struggle most. Level 2 reaches a perfect **100% faster-than-compile** rate and **2.80x** geometric mean speedup.

---

## 6. What the Agent Actually Learns

The case studies are the strongest evidence that the agent is doing systems work rather than memorized syntax.

### Diagonal matrix left-multiply

For $D = \operatorname{diag}(A)$ and dense matrix $B$, the naive path treats the problem like GEMM. CUDA Agent notices that the product is just row-wise scaling:

$$
C[i, j] = A[i] \times B[i, j].
$$

That collapses complexity from $O(N^2 M)$ to $O(N M)$ and avoids materializing the diagonal matrix. The generated kernel is a single grid-stride loop and reaches **73.31x** speedup over `torch.compile`.

### Fused matmul, reduction, divide, and scale

The paper gives a chain of operations of the form

$$
Y_i = \sum_j \frac{X_i W_j^T}{2}.
$$

Instead of launching a GEMM followed by elementwise and reduction kernels, the agent rewrites it as

$$
Y_i = X_i \cdot \left(\sum_j W_j^T \right) / 2.
$$

Now the hot path is "reduce the weights once, then do a dot product." The implementation uses coalesced accesses, `float4` vectorized loads, and shared-memory tree reductions, yielding **24.04x** speedup.

### ResNet BasicBlock

On the hardest case, the agent combines algebra and library awareness:
- fold BatchNorm into convolution weights and bias,
- call cuDNN's fused convolution-bias-activation path,
- run in **TF32** to use Tensor Cores,
- fuse the residual add and final ReLU into one custom kernel with `float4` loads.

That stack reaches **3.59x** speedup over `torch.compile`. The interesting part is that the agent is willing to mix custom kernels with vendor libraries when that is the fastest route, which is exactly how human CUDA experts think. It lives in the same optimization world as [Educational Materials for GEMM Optimizations on CPUs and GPUs]({% post_url 2024-11-11-educational-materials-gemm-optimizations %}) and [High-Performance Matmul Kernels on NVIDIA Hopper]({% post_url 2025-12-29-Hopper-GEMM %}): tile shapes, memory traffic, vectorized loads, and launch boundaries matter more than elegant source code.

---

## Takeaways

1. CUDA Agent is really three ideas glued together: synthetic fused tasks, a tool-using execution harness, and a stabilization recipe for RL in a brutally low-probability domain.
2. The evaluator design is as important as the policy. Discrete rewards, isolated sandboxes, and anti-fallback checks make reward hacking harder than genuine optimization.
3. The warm-up pipeline is the real algorithmic novelty. Without actor and critic initialization, PPO collapses before the agent learns anything useful.
4. The strongest result is not the pass rate but the **96.8% faster-than-compile** number. That means the model is discovering kernels that are both valid and physically better.
5. This is a narrow but important preview of what agentic systems can do when the loop is grounded in hard measurements rather than text-only judgments.
