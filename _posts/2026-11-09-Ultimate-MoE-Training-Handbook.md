---
layout: post
title: "The Ultimate MoE Training Handbook: Routers, Balancers, and Where the Step Goes"
date: 2026-11-09
categories: [MoE, Training]
tags: [MoE, Megatron-LM, Load Balancing, Expert Parallelism, Router, Quantile Balancing, LatentMoE, Muon, MTP, All-to-All, RL, Serving, Hugging Face]
---

Reading notes on:
- [The Ultimate MoE Training Handbook](https://huggingface.co/spaces/huggingface/moe-blogpost)

Hugging Face's new handbook is the MoE sequel to the [Smol Training Playbook]({% post_url 2025-11-29-Smol-Train %}). The playbook trained a dense SmolLM3 and only listed what makes distributed MoE training hard: expert placement, load balancing, and all-to-all communication. This handbook starts from that list. Its opening observation sets the stakes: on the Artificial Analysis index, **18 of the 20 highest-scoring open-weights models are MoEs**, including the top model of each of the five leading labs.

Going from dense to MoE adds three things to manage: a **router** that picks each token's experts, a **load** that must stay balanced across those experts, and, once the experts live on different GPUs, an **all-to-all exchange** in every MoE layer. The handbook takes every setting that touches one of these, explains what it does, reports what frontier models chose, and, for most of them, measures it on their own cluster (AWS P5, 8×H100 nodes over EFA) at a pinned Megatron-LM commit.

What makes it more useful than a survey is the structure: each part ends with a **symptom** (slow, OOM, blows up, learns badly), a **signal to watch**, and a **recommendation**. An appendix runs the index backwards, from a dashboard symptom to the setting behind it. These notes follow the same order and keep the numbers. For the Megatron Core paper that describes much of the machinery underneath, see my [notes on MoE training in Megatron Core]({% post_url 2026-03-11-MoE-Megatron %}).

---

## 1. Before the model: test the cluster, then build a dense reference

### Measure the collective you will actually use

With Megatron's `alltoall` dispatcher, each MoE layer sends tokens to the GPUs holding their experts and returns them with a second all-to-all. A healthy all-reduce says nothing about how fast that runs, so the handbook tests all-to-all directly: DCGM level-1 diagnostics on every node, then `alltoall_perf` from nccl-tests with an 8 GiB buffer per rank.

The baseline has to depend on node count, because more of the exchange crosses the inter-node fabric as you add nodes. Their sweep reached **77.14 GB/s on 2 nodes, 39.25 on 4, and 30.91 on 8**. A practical policy is to require 90% of the baseline at the same node count (about 69 GB/s for two nodes). A single fixed threshold of 33 GB/s would accept a slow two-node allocation and reject a healthy eight-node one. During training, `--gpu-sniff-test-interval <n>` reruns a GPU check plus an all-to-all over the expert-parallel group every `n` iterations.

The most expensive gotcha is silent: NCCL can fall back to TCP sockets even when EFA is requested. Two of their runs set `FI_PROVIDER=efa`, named an OFI plugin, still logged `Using network Socket`, and ran **about 25× slower** (1.7 against 42.0 TFLOP/s). NCCL reached EFA only once the EFA/OFI libraries came first on `LD_LIBRARY_PATH` and the plugin was selected by name (`NCCL_NET_PLUGIN=ofi`); passing a path made NCCL build an invalid library name. Always grep the startup log for `Using network Libfabric`.

### A dense baseline, and an MoE matched to it

The dense reference is a 16-layer model, hidden size 2,048, FFN width 8,192, a 128,256-token vocabulary: **about 1.22B parameters**. It trains on FineWeb-Edu with the Llama 3.2 tokenizer, micro-batch 8, global batch 512, sequence length 4,096 (2,097,152 tokens per step), WSD schedule peaking at 5e-4, full activation recomputation.

The MoE replaces every FFN with **64 experts of width 1,024, top-8**. The expert width is chosen so the active FFN parameters match exactly. A bias-free SwiGLU expert has three matrices:

$$
3 \times 2048 \times 1024 = 6{,}291{,}456, \qquad 8 \times 6{,}291{,}456 = 50{,}331{,}648 = 3 \times 2048 \times 8192 .
$$

All 64 experts store 8× the dense FFN, but attention, embeddings and the rest stay the same size, so the whole model grows to **6.86B total with 1.221B active** (against 1.219B for dense), about 5.6× the parameters at nearly the same active count. The MoE flags are `--expert-model-parallel-size 8 --num-experts 64 --moe-ffn-hidden-size 1024 --moe-router-topk 8 --moe-token-dispatcher-type alltoall --moe-grouped-gemm --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-3 --moe-router-dtype fp32`.

### A parity check you can trust

An MoE with one expert and top-1 routing computes the same function as the dense model it came from. Upcycled from the dense checkpoint, their 1-expert MoE stayed **within 0.00087 of the dense run over 15 aligned steps**. From a fresh initialization the two runs sat 2.72 apart at iteration 4, which says nothing. The parity run also shows the expected fingerprints: a load-balancing loss of exactly 1 and about 32,770 zeros in the gradient (the router weights, which get no gradient with a single expert).

Deterministic mode makes such comparisons meaningful. With `--deterministic-mode` (which requires dropping cross-entropy fusion and setting `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`), two runs printed identical LM loss, balancing loss, grad norm and gradient zeros at every iteration; without it the LM loss differed from the first iteration, by up to 0.00063. The handbook's rule: **switch on deterministic mode before comparing two runs or chasing a NaN**. This is the same discipline as Megatron's [bitwise-deterministic pretraining work]({% post_url 2026-10-12-Megatron-Core-Bitwise-Deterministic-Pretraining %}).

---

## 2. The router: four knobs that shape the weights, one that changes the winners

The router turns each token into one logit per expert. Between those logits and the weights that mix expert outputs sit four choices: the gating function, the output scale, the sharpness, and the precision. In plain top-k routing the first three never change *which* experts win, only how much each counts. **Precision is the exception**: a bf16 router can round two close logits to the same value and change the winners.

![Router pipeline in five stages: (1) logits in fp32, the only stage that can change which experts win because bf16 rounds close logits into ties; (2) gating function, softmax, sigmoid or sqrt(softplus), same ranking but different weights, and the expert bias needs sigmoid or sqrt(softplus); (3) top-k selection, where a balancing bias is added for selection only; (4) renormalize the k weights to sum to 1, which lowered loss from 2.219 to 2.183; (5) multiply by the routed scaling factor and add the shared expert at weight 1. Below each stage, the signal to watch: z-loss and fp32 flip rate, top weight with entropy and Max1/Max2, MaxVio, routed output size, and the shared expert's share of 1/(1 + factor)](/assets/images/moe_handbook_router.svg)

| Model | Gating | Output scale | Router precision (inference code) |
| --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | sqrt(softplus) | renormalized, 1.5 at k = 6 | fp32 |
| Gemma 4 26B-A4B | softmax | renormalized + learned per-expert scale, k = 8 of 128 | bf16, softmax in fp32 |
| GLM-5.3 | sigmoid | renormalized, 2.5 at k = 8 | fp32 |
| Inkling | sigmoid, normalized jointly with 2 shared experts | 8.0 × learned global scale | bf16 |
| Kimi K3 | sigmoid | renormalized, 1.0 at k = 16 of 896 | fp32 |
| Qwen3.8-Flash-Next | softmax | renormalized, no factor, k = 10 | bf16, softmax in fp32 |
| Step-5-Preview | sigmoid | renormalized, 3.0 at k = 8 of 352 | fp32 |

### Gating function

Softmax, sigmoid and sqrt(softplus) rank experts identically; they differ in the weights. As logits grow, softmax hands more and more weight to the top expert, while sigmoid saturates and gives confident experts equal weights. sqrt(softplus) lands between: for positive logits, the ratio of two weights tends to the square root of the ratio of their logits. DeepSeek-V4 switched to it from sigmoid (see my [DeepSeek-V4 architecture notes]({% post_url 2026-04-26-DeepSeek-V4-Arch-Train %})).

The practical constraint: Megatron refuses the loss-free expert bias (section 3) with softmax, because softmax's normalization makes the gap between two experts depend on every other expert's score. **Use sigmoid or sqrt(softplus) with the expert bias; otherwise softmax, Megatron's default.**

### Output scale

With softmax, the top-k weights sum to less than 1 by a different amount per token, so the layer's output size moves with them. Renormalizing the k chosen weights to sum to 1 fixes that, and in Krajewski et al. it **lowered validation loss from 2.219 to 2.183** at granularity 8. One trap: softmax at top-1 with renormalization always gives weight 1, so the router gets no gradient. Megatron requires `--moe-router-pre-softmax` there.

Even renormalized, the output shrinks when weight is spread. With all weight on one expert, the output has that expert's RMS; with equal weights on k uncorrelated experts it drops to $1/\sqrt{k}$ of that. A routed scaling factor near $\sqrt{k}$ restores it: DeepSeek-V3 uses 2.5 at k = 8, Kimi K2 used 2.827 ≈ $\sqrt{8}$. But shipped models disagree (Kimi K3 went to 1.0 at k = 16), so the handbook's verdict is that **the scaling factor has no settled value**.

### Sharpness

A linear softmax router gets sharper as the token norm or the router weights grow. Three controls: logit soft-capping (Grok 2 at 30), normalizing the router input (Gemma 4: RMSNorm, a learned scale and $1/\sqrt{d}$), and gating logit normalization, which standardizes logits across experts. Skywork-MoE used the last against a gate so flat the layer averaged its top-k. The signal is **Max1/Max2**, the top expert's weight over the second's, falling toward 1.

### Precision: always fp32

Top-k orders ties unpredictably. At top-8 of 64 the 8th logit sits about 3.5 below the logsumexp; bf16 has a coarse grid there, and fp32 rounds 65,536× more finely. Z-loss pushes logits down but bf16 rounds the resulting values in steps of the same length, so it does not remove the flips; only fp32 does. The training evidence is about stability: a bf16 router diverged in Switch Transformers while fp32 trained at bf16 speed. Only 2 of 10 labs say what precision they train the router in, and 5 of 10 run the router matmul in bf16 at inference.

Logged z-loss is divided by its coefficient, so its square root is the RMS logsumexp: their run logged 20.9, about 4.6. Recommendation: **always `--moe-router-dtype fp32`**, and add `--moe-z-loss-coeff 1e-3` (ST-MoE's value) for a softmax router whose logits drift up.

### Who selects, and does it need to be learned?

**Expert choice** (each expert takes its best tokens) balances perfectly but leaks the future: whether a token gets an expert depends on later tokens, which do not exist at inference. **Expert threshold** keeps loads near even using cutoffs from earlier batches; in a 575M model it reached loss 2.844 against about 2.89 for token choice. Megatron trains token choice.

**Hash routing** replaces the learned choice with a fixed table keyed by token id. DeepSeek-V4 hashes its first 3 MoE layers. Hash layers matched Switch and BASE up to 4.5B, but Clark et al. found they stop improving as more layers become MoE layers, and no bias can rebalance them. DeepSeek-V4.1-Flash dropped hash layers and kept only Engram n-gram memory, which picks no experts (see my [V4.1-Flash notes]({% post_url 2026-09-14-DeepSeek-V4.1-Flash-KV-Cache-Compression %})). If you use `--moe-num-hash-layers`, load a real table; Megatron's default is a round-robin placeholder, and nothing corrects a bad table during training.

---

## 3. Load balancing: what imbalance costs, and five ways to fix it

### The cost, measured

When tokens crowd onto a few experts, the GPU holding them sets the step time. The handbook routes the same model three ways:

- **Forced-even** routing (`--moe-router-force-load-balancing`, random logits): **149.4 TFLOP/s**, the best case.
- **Forced-biased** routing, a dial of growing shared bias: at the far end, **100.5 TFLOP/s**, with the busiest expert receiving every token (MaxVio 7, as bad as top-8 of 64 can get).
- A **freshly initialized learned router** ran its first 40 iterations as slowly as the worst forced case.

The balance metric throughout is **MaxVio**, from Wang et al.:

$$
\text{MaxVio} = \frac{\max_i c_i}{\operatorname{mean}(c)} - 1 ,
$$

where $c$ is the vector of tokens per expert. 0 is perfectly even; 1 means the busiest expert gets twice the average. Note that averaging micro-batch MaxVio does not give global-batch MaxVio; you have to sum counts first. Forced-even routing is a **control run only**: it shows the speed to aim for, never a way to train.

### Aux loss

The oldest fix: coefficient × expert count × $\sum_i f_i P_i$, each expert's token share times its mean router probability. Its gradient competes with the LM loss, so the coefficient is a trade-off. In Wang et al.'s 1B model (64 experts, top-6), aux loss at 1e-3 gave **MaxVio 0.72 at perplexity 9.56**, against **0.04 and 9.50** with the loss-free bias; at 1e-2, MaxVio fell to about 0.1 but perplexity rose to about 9.69. Yuan3.0's 10B ablation tells the same story: test loss rose with the coefficient from 1e-3 to 0.1, while 1e-4 still left the busiest expert with 13.6× the tokens of the least used.

A subtle trap: **the aux loss can read balanced while the load is skewed**. With flat router probabilities the product $f_i P_i$ reads the same as for an even load whatever the token shares are. Sigma-MoE-Tiny's lower layers hit exactly this. So read balance from MaxVio, not from the loss.

### Scope: sequence, micro-batch, or global batch

Pretraining pipelines often pack a micro-batch from one domain. A loss that demands even load inside that micro-batch spreads code tokens over every expert, so no expert specializes in code. Balancing over the global batch lets domain experts form, which lowered perplexity in Qiu et al., but each GPU's micro-batch load becomes uneven: their 43B model's step ran **about 5.8% slower**, roughly 1 point of that from communication and the rest from imbalance. The flags are `aux_loss` (micro-batch, the default recommendation), `seq_aux_loss`, and `global_aux_loss`, and they can be combined with one coefficient each.

### Loss-free bias

Each expert gets a bias added to its score only for selection; weights still come from the raw scores, so no gradient is touched. Once per step, Megatron moves each bias up or down by a fixed **update rate** (the sign update). It matched a batch-wise aux loss in DeepSeek-V3's ablations at 1B and 3B, and it is the balancer behind [DeepSeek-V3]({% post_url 2024-12-26-deepseek-v3 %}).

The catch is that the fixed step must suit the expert count. In Li et al.'s sweep with rate 1e-3, the busiest expert grew to **about 50× the mean at 1,024 experts** (aux loss at 1e-2 stayed under about 7×), and above 128 experts the bias also hurt LM quality. Alternatives exist: LongCat-Flash runs a linear (PID-style) update, ZAYA1-8B feeds load errors to AdamW, and Moonlight and Trinity Mini subtract the mean step so biases do not drift (Megatron stores the bias in fp32, so drift costs nothing there).

A batch-wide bias can still let one sequence pile onto a few experts. DeepSeek-V3 adds a tiny sequence-wise loss (α = 1e-4) against that. Chang's **causal bias** builds a per-sequence bias only from tokens already routed, avoiding future leakage, though the evidence is small-scale. For multimodal batches, DeepSeek-V4.1-Flash keeps **one bias per modality**, since a shared bias can balance the sum while text and images each crowd onto their own experts.

### Quantile balancing

Instead of stepping toward the right bias, solve for it: for each expert, rank the micro-batch's tokens by score and read off the quantile that only the target share beats. On a 19B MoE with 768 experts (Jin et al.), it balanced sooner than the sign update at about the same LM loss, halving average MaxVio. **Kimi K3** runs it over the global batch with histograms summed across GPUs, because the sign update struggled once it reached 896 experts per layer (see my [Kimi K3 notes]({% post_url 2026-07-18-Kimi-K3 %})).

The risk is again scope: Neitemeier et al.'s 7.5B run balanced the global batch to MaxVio 0.92 while the worst layer's expert-parallel micro-batch sat at **local MaxVio 5.96**. Megatron also refuses it next to any aux loss, router fusion, or group-limited routing.

### Head to head

The handbook ran five router configurations from init on the same model (one seed) and took median throughput over the last 20 of 120 iterations:

![Panel A: median throughput over iterations 101 to 120 for router configurations on the 6.86B-A1.22B MoE: forced-even control 149.4 TFLOP/s, quantile balancing 139, aux loss 1e-3 125, quantile balancing with EMA 0.9 121, the most-skewed forced-bias control 100.5, and the sigmoid loss-free bias at rate 1e-3 99. Panel B: published balance numbers, Wang et al. 1B aux loss MaxVio 0.72 at perplexity 9.56 vs loss-free bias 0.04 at 9.50; Li et al. bias at 1e-3 reaching about 50x the mean load at 1,024 experts; Neitemeier et al. global MaxVio 0.92 hiding local MaxVio 5.96; Qiu et al. global-batch balancing costing 5.8 percent step time](/assets/images/moe_handbook_balancers.svg)

Quantile balancing led at 139 TFLOP/s, then aux loss at 125, quantile with EMA 0.9 at 121, and the bias-only run at 99. Two caveats the authors flag themselves: the runs used an older Megatron commit that accumulated the bias's token counts in bf16 (a rounded count could flip an update's sign; fixed at the pin), and **quantile balancing also ended with the highest training LM loss** of the five, so its throughput lead alone does not prove faster training to the same quality. This is the same territory as [UltraEP]({% post_url 2026-08-12-UltraEP-Exact-Load-Balancing-Rack-Scale-MoE %}), which attacks imbalance on the systems side with per-micro-batch replicas instead.

---

## 4. The experts: count, width, latents, sharing, activation

| Model | Routed (of total), width | Latent | Shared | Activation |
| --- | --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | 6 of 384, 2,304 | none | 1 × 2,304 | SwiGLU, clamp 10 |
| GLM-5.3 | 8 of 256, 2,048 | none | 1 × 2,048 | SiLU gate |
| Kimi K3 | 16 of 896, 3,072 | 7,168 to 3,584, norm before up-proj | 2 × 3,072 | SiTU-GLU, caps 4 and 25 |
| LongCat-Flash | 12 of 768, 2,048 (256 identity) | none | none | SiLU gate |
| MiMo-V2.6-Pro-RL | 8 of 384, 2,048 | none | none | SiLU gate |
| Nemotron 3 Ultra | 22 of 512, 5,120 | 8,192 to 2,048 | 1 × 10,240 | ReLU² |
| Step 5 Preview | 8 of 352, 1,536 | none | 1 × 1,536 | SwiGLU, clamp 7 |

### Granularity and sparsity

At fixed total and active parameters, cutting experts finer (twice as many, half as wide, twice the top-k) lowered loss in Liew et al.'s 18-layer model **from 2.334 to 2.297, up to 128 experts of d/4 with top-8**, and finer cuts gained nothing while still running smaller GEMMs and dispatching more copies of each token. LongCat found that once experts held about 12× the active parameters, extra parameters did more as **n-gram embedding tables** than as more experts, with a rule of at most 50% of parameters in tables (see my [LongCat notes]({% post_url 2025-12-08-LongCat %})).

Sarkar et al. add the systems view: choosing sparsity by model FLOPs alone points to the sparsest model, but accounting for real training throughput on their cluster moves the optimum to a less sparse configuration, **0.031 nats** better than the sparsest. More sparsity can lower hardware utilization without raising FLOPs per token. Small expert GEMMs are exactly what [SonicMoE]({% post_url 2025-12-19-SonicMoE %}) targets.

### LatentMoE

Each dispatched copy carries $d$ numbers. [LatentMoE]({% post_url 2026-01-31-LatentMoE %}) projects tokens down to $\ell$ before dispatch and back up after combine, so at the same expert parameters, compute and traffic the layer can afford $\alpha = d/\ell$ times more experts and an α times larger top-k. In NVIDIA's 73B-A8B ablation it scored **4.6 points higher on MMLU-Pro** at matched size. The costs are two extra $d \times \ell$ projections (serving throughput from 12% lower at concurrency 1 to slightly higher elsewhere) and a new instability: Kimi K3 saw routed-branch activations blow up at 2.8T scale and fixed it with an RMSNorm before the up-projection (`--moe-use-norm-before-up-proj`).

### Shared experts

The evidence by loss is small and inconsistent: about 0.01 to 0.02 lower in Pangu Ultra MoE and expert-threshold models, nothing or slightly worse in OLMoE and Li et al. Sharing also reduces routing freedom: with 32 experts and 4 active, reserving one slot leaves **4,495** ways to choose the other 3 against **35,960** to choose 4. And mixing weights matter. Without a gate, Megatron adds the shared output at weight 1 while routed weights sum to the scaling factor, so the shared expert takes $1/(1 + \text{factor})$ of the output: **29% at a factor of 2.5**, half at 1.0. The recommendation for a fair comparison: give it the routed width and replace one routed slot.

### Activation and clamps

Step 3.5 Flash saw one or two experts develop rapidly growing output norms while the loss barely moved. SwiGLU and ReLU² outputs grow with the square of the input scale; a hard clamp bounds it, and SiTU-GLU bends away smoothly before the bound. K-EXAONE 2.0 measured a peak hidden unit of 6,862 with its clamp disabled. In Megatron: `--activation-func-clamp-value 10` (DeepSeek V4.1 Flash's limit) or Kimi K3's `--activation-func-tanh-clamp-scale 4 --activation-func-tanh-clamp-scale-linear 25` (derivation in my [Kimi K3 architecture notes]({% post_url 2026-08-04-Kimi-K3-Architecture-Deep-Dive %})). The signal is the **max-to-median ratio of expert output norms per layer**.

---

## 5. Layout and block wiring

A dense layer makes no all-to-all. Leading dense layers (DeepSeek-V3 keeps 3, Kimi K3 and MiMo-V2.6-Pro keep 1) replace only the earliest MoE layers; alternating dense and MoE (Llama 4 Maverick) halves the all-to-alls. In Ling's 60-layer models, making the first 1 to 3 layers dense barely changed efficiency at matched FLOPs. The other argument, that dense first layers fix early-layer imbalance, rests on thin evidence: Motif 3's first MoE layers after two dense ones still balanced worst. Watch the first MoE layers specifically.

Wiring decides how much all-to-all can hide. In a standard block the router waits for attention and the next layer waits for combine. A **parallel block** (North Mini Code, Moondream 3) feeds attention and the MoE the same input so dispatch overlaps attention; a **shortcut-connected MoE** (LongCat-Flash, ScMoE) hides combine too. In the handbook's illustrative timing the exposed all-to-all share falls **from 33% to 7%**.

---

## 6. Starting, optimizing, and keeping it stable

### Upcycling

Megatron splits the dense 8,192-wide MLP into 8 slices and copies each into 8 of the 64 experts, then starts a fresh optimizer and schedule. The head start is real but temporary: in OLMoE a from-scratch run caught up after **500B tokens, 25% of the dense model's 2T**; in Komatsuzaki et al. it took **120% of the dense checkpoint's compute**. Use it when a dense checkpoint exists and the MoE run is short. Growing an MoE by copying experts has its own trap: exact copies tie, so every token picks an expert and its copy at equal weight; perturb the copies (K-EXAONE 2.0 uses a random rotation, Drop-Upcycling re-initializes half).

### Muon and the router

AdamW's step follows each expert's gradient history. Muon orthogonalizes each update and scales by matrix shape, so **a rarely picked expert can move about as far as the busiest one**, and keeps moving along momentum in steps where it got no tokens. Megatron, like DeepSeek-V4, orthogonalizes each expert's matrix separately. Moonlight (3B active, 16B total) did significantly better with Muon than AdamW at about 1.2T tokens. The open choice is the router: DeepSeek-V4 keeps it on Muon, Qwen3.8-Next moved it to AdamW after Muon destabilized it early (see my [Qwen3.8-Next notes]({% post_url 2026-08-27-Qwen3.8-Next-GDN-QSA-Gated-Residual %})). Start at `--muon-extra-scale-factor 0.2`: the default 1.0 diverged on a dense 7B in Mellum2's tests. Broader optimizer context in my [SOAP and Muon notes]({% post_url 2026-07-26-SOAP-Muon-Higher-Order-Optimizers %}).

### Multi-token prediction

MTP adds one transformer layer per extra token, fed by the main hidden state and the next token's embedding through separate RMSNorms. The cost lands in one place: Megatron puts it on the last pipeline stage, which already holds the output head. On their 256-node runs, early steps took **about 37% longer** with one MTP layer, and 8 pipeline stages did not close the gap. Ling 2.0 measured an MTP layer at about 1.7× a standard MoE layer's compute and gained **40% end to end** by repartitioning around it. Place it with `--pipeline-model-parallel-layout`.

### Failures the loss does not show

Three failures can leave the loss flat: **routing collapse** (MaxVio climbs), **dead experts** (smallest output norm far below the layer median), and **rogue experts** (largest far above it; Step 3.5 Flash's layer 45 went from about 14 to about 5,700× the median). The handbook's universal checklist: z-loss for the router logits, MaxVio per layer, max and min expert output norm over the median per layer, plus the default NaN check and `--rerun-mode validate_results`.

---

## 7. Where an MoE step spends its time

In the one step they traced on one node, **half the time was exposed all-to-all**. But that step had a balancing loss of 3.2 (1 is balanced), and the same iteration with forced-even routing ran **40% faster**: much of the "communication" was ranks waiting for the overloaded one. The diagnostic recipe: trace one step per config, and if the all-to-all is large, re-trace with forced balance. If it shrinks, fix balancing first; if not, work on the interconnect and overlap. The levers below each come from their own setup, so the handbook warns **not to add their gains together**.

![Panel A: diagnosis order for a slow MoE step. Trace one step at your node count; re-trace with forced load balancing; if the all-to-all shrinks, fix balancing first, and if it stays, confirm NCCL uses the fabric and keep EP inside a node, then turn on the EP overlap, then use selective recompute and offloading if memory runs out. Panel B: measured speed change against each lever's own baseline: forced-even routing +40 percent on the same iteration, EP overlap up to +35 percent on two nodes, permute fusion +3.4 percent (step time minus 3.3 percent), full recompute minus 24 percent, EP16 across two nodes minus 59 percent versus EP8 in one, and an NCCL socket fallback minus 96 percent (1.7 vs 42.0 TFLOP/s)](/assets/images/moe_handbook_step.svg)

| Lever | Measured effect |
| --- | --- |
| `--moe-permute-fusion` | permute/copy kernels 11.45% → 7.14% of step; step time −3.3% |
| EP16 across 2 nodes vs EP8 in 1 node | 59% lower throughput (EFA working) |
| NCCL fallback to sockets | about 25× slower (1.7 vs 42.0 TFLOP/s) |
| `--overlap-moe-expert-parallel-comm` | up to +35% throughput on 2 nodes; 5 to 10 GB more reserved on rank 0 |
| Full recompute at micro-batch 2 | −24% throughput, and it blocks the overlap |
| Fine-grained offloading (NVIDIA, DeepSeek-V3 shapes) | −18% peak reserved memory, −2% throughput |
| FP8 for large MoE training (Megatron Core team) | about 10 to 25% end to end |

**Expert parallelism inside a node** is the default: start there, add nodes as data-parallel replicas. When the experts must span nodes, the **overlap** pairs one micro-batch's forward with another's backward (the same idea as DualPipe), and `--delay-wgrad-compute` lets the backward dispatch wait only for input gradients.

**Memory.** Activations grow with top-k because the dispatcher copies each token's hidden state once per expert. The handbook's recipe is selective: recompute the cheap modules (`--recompute-modules layernorm moe_act`) and offload the large inputs to pinned CPU memory on a side stream (`--offload-modules core_attn attn_proj expert_fc1`). Under CUDA graphs, dropless MoE layers must reserve buffers for the worst load, often 10× more than they keep; the experimental `--moe-paged-stash` computes in one shared buffer and stashes only real tokens into pages.

**Group-limited routing.** DeepSeek-V3 scores each node by its top-2 expert scores, keeps the best 4 of 8 nodes, then picks top-k within them, so a token reaches at most 4 nodes. The saving depends on the dispatcher: `alltoall` sends one copy per expert, while the flex dispatcher with DeepEP sends one per node and forwards over NVLink.

**Capacity versus replicas.** A capacity factor drops overflow silently: at 1.5, Pangu saw about 6% of tokens dropped at 20B and 8% at 718B, and the dropless run kept lower loss throughout. Replicas balance without touching the loss: UltraEP took the busiest rank from **1.30 to 2.86× the mean down to 1.01 to 1.03**, and DeepSeek's serving engine keeps the slowest GPU within 1.09× the mean. Keep dropless as the default and always for fine-tuning, where drops cut into instruction-response pairs.

**Precision.** Follow DeepSeek-V3's split (linear layers in FP8 forward and backward; embedding, head, norms, attention core and router in BF16 or FP32), and keep relative loss difference against a BF16 run below about 0.25%, as DeepSeek's 16B and 230B runs and Ling-1T did. Under NVFP4, also count exactly-zero expert weight-gradient elements, which reached 7% of parameters in Nemotron 3 Super. Background in my [MXFP8]({% post_url 2025-12-07-MXFP8-Train %}) and [NVFP4]({% post_url 2025-11-20-NVFP4-Train %}) notes.

---

## 8. After training: serving and RL

![Panel A: MMLU change when serving top-4 on Qwen3.6-35B-A3B, trained at top-8 with 81.65 percent MMLU: renormalizing over the four survivors costs 4.65 points, dividing by the top-8 softmax mass costs 3.10, dividing by the top-16 mass costs 0.35 (not significant), and no renormalization at top-6 costs 27.45. Panel B: storing gpt-oss-20b's routed experts in 4 bits shrinks the model from 39 to 12.8 GiB but the weight bytes read per decode token only from 6.72 to 3.45 GiB. Panel C: DeepSeek-V4.1-Flash decode at 64 requests per GPU, EP64 versus EP8: 15,700 versus 12,600 tokens per second per GPU and 246 versus 198 tokens per second per user](/assets/images/moe_handbook_serving.svg)

### Serving a different k

Dropping experts at serving saves compute but not memory, and renormalization makes it tricky: survivors share the trained total, so each weighs more. Chen and Yao keep the top $k_1$ experts but divide by the softmax mass of the top $k_2$. On Qwen3.6-35B-A3B (256 experts, trained top-8, a flat router whose top 8 hold only 0.182 of the mass), against 81.65% MMLU at top-8, top-4 scored **4.65 points lower** renormalized over the survivors, **3.10 lower** divided by the top-8 mass, and only **0.35 lower** (not significant) divided by the top-16 mass. With no renormalization, top-6 lost 27.45 points. On Gemma-4-26B-A4B (top-8 holds 0.302), dropping renormalization at top-6 raised perplexity by 0.7% against 356% on Qwen. Serving *more* experts than trained did not help OLMoE.

### Low-bit experts

Storing only the routed experts in 4 bits shrinks gpt-oss-20b from **39.0 GiB to 12.8 GiB**, but the weight bytes a batch-1 decode token reads fall only from **6.72 to 3.45 GiB**, because each token reads 4 of 32 experts but all of attention and the unembedding. gpt-oss (MXFP4 at 4.25 bits), Kimi K3 (MXFP4 through SFT and RL), DeepSeek-V4 (MXFP4 QAT) and GLM-5 (INT4 QAT in SFT) all quantize after pretraining, so the pretraining config stays as it is.

### Expert parallelism at serving

DeepSeek sizes the phases in opposite directions for V4.1-Flash: **prefill at EP2 to EP8** (smallest that fits), **decode at EP64 to EP128** (largest available). At 64 requests per GPU, EP64 decodes about 15,700 tokens/s/GPU at 246 tokens/s/user against EP8's 12,600 and 198, about 25% higher on both; at 256 requests per GPU the two nearly meet. The price: the smallest deployment worth running is 64 prefill plus 64 decode GPUs, and one failed GPU takes down the whole instance.

### Attention-FFN disaggregation

Splitting attention (KV-heavy, few parameters) and experts (parameter-heavy, stateless) into separate pools is the next step beyond [prefill/decode disaggregation]({% post_url 2025-03-30-prefill-decoding-disagg %}). But the published gains shrink under fair comparison: FastAFD's 1.41× over colocated EP4 fell to **0.99×** when Qiu et al. re-provisioned both layouts for the same 72 GPUs. What training decides here is sparsity: Step-3's bound for staying compute-bound on an expert pool is 0.058 activated/routed at DeepSeek-V3's width and depth on H800.

### RL routing mismatch

The rollout engine and trainer can route the same token differently. Ma et al. found about 10% of routers disagreeing between SGLang and Megatron on Qwen3-30B-A3B, with **94% of tokens differing in at least one layer**; separately, one gradient update changed about 10% of activated experts. [R3]({% post_url 2025-12-28-R3 %}) replays the rollout's experts in the trainer, cutting train-inference KL from 1.5e-3 to 7.5e-4 (dense Qwen3-8B: 6.4e-4). Whether it pays in reward is contested: Ma et al. saw no replay run crash, while a later Qwen on-policy study saw no gain, and the GSPO authors dropped it for sequence-level ratios. Megatron offers `--moe-enable-routing-replay` and `--batch-invariant-mode` for the kernels' share of the gap (more on that in my [MoE numeric parity notes]({% post_url 2026-04-08-MoE-Numeric-Parity %})). One sharp warning: NVFP4 rollouts against a BF16 trainer collapsed after about 150 steps, traced to the **quantized activations**, not the weights.

---

## Takeaways

1. **Test the all-to-all, not the all-reduce**, with a per-node-count baseline, and grep for `Using network Libfabric`. A socket fallback costs 25×.
2. **Keep a dense twin and determinism on.** A 1-expert upcycled MoE should track dense within about 1e-3.
3. **Router precision is the one router setting that changes winners**: always fp32. Gating, scale and sharpness change weights, and the routed scaling factor has no consensus value.
4. **Read balance from MaxVio at the scope you balance**, never from the aux loss value, and judge balancers by TFLOP/s against a forced-even control (149.4 vs a learned router starting near 100).
5. **Balancer choice depends on expert count.** Aux loss around 1e-3 is the safe default; the sign-update bias gets too coarse past 128 experts; quantile balancing is what Kimi K3 moved to at 896, with a global/local gap to watch.
6. **Much of "communication time" is waiting on an overloaded rank.** Forced balance made their traced step 40% faster. Fix balance before tuning the network.
7. **Many recommendations rest on one ablation in one lab.** The authors are explicit about this, and ask for configs and logs, including negative results.

---

## Appendix: a condensed flag set

The handbook ends with every recommended flag grouped by section. A starting subset for a single-node run, with conditional lines commented out:

```bash
# monitoring
--moe-per-layer-logging --save <dir> --save-interval <m>
--save-tokens-per-expert-interval <n>          # tokens per expert -> MaxVio
--log-throughput --log-params-norm --log-num-zeros-in-grad --log-max-attention-logit
# router
--moe-router-dtype fp32
# --moe-z-loss-coeff 1e-3                       # softmax router whose logits drift up
--moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 1e-3
# --moe-router-enable-expert-bias --moe-router-score-function sigmoid \
#   --moe-router-bias-update-rate <rate>        # loss-free bias; tune to expert count
# --moe-router-load-balancing-type quantile_balancing
# shape
--num-experts 64 --moe-ffn-hidden-size 1024 --moe-router-topk 8
# --moe-latent-size <l> --moe-use-norm-before-up-proj
# --activation-func-clamp-value 10              # if expert norms diverge
# --moe-layer-freq "([0]+[1]*15)"               # one leading dense layer
# training
# --optimizer muon --muon-extra-scale-factor 0.2 --moe-router-skip-muon
# --mtp-num-layers 1 --mtp-loss-scaling-factor 0.1
--rerun-mode validate_results
# systems
--expert-model-parallel-size 8                  # EP inside one node
# --moe-permute-fusion
# --overlap-moe-expert-parallel-comm --delay-wgrad-compute
# --recompute-granularity selective --recompute-modules layernorm moe_act
# --fine-grained-activation-offloading --offload-modules core_attn attn_proj expert_fc1
```
