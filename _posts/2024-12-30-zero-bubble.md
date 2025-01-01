---
layout: post
title: "Summary on Zero Bubble"
date: 2024-12-30
categories: [LLM, pipeline parallelism, zero bubble]
tags: [LLM]
---

Read [Zero Bubble](https://arxiv.org/pdf/2401.10241).


# Summary

* **Pipeline parallelism (PP) is a key technique for large-scale distributed training but suffers from efficiency losses due to pipeline bubbles.** Pipeline bubbles refer to the idle time in a pipeline when workers are waiting for computations from other workers.

* This paper introduces a novel scheduling strategy called **Zero Bubble (ZB)** that aims to eliminate pipeline bubbles in synchronous training.

* **The key idea behind ZB is to split the backward computation into two parts:** one for computing gradients for the input (B) and another for computing gradients for the parameters (W). This split allows for greater flexibility in scheduling, enabling the placement of W passes to fill pipeline bubbles.

* The paper proposes two handcrafted schedules: **ZB-H1**, which minimizes peak memory usage, and **ZB-H2**, which achieves zero bubbles but requires more memory.

* **An automatic scheduling algorithm is developed to handle real-world scenarios.** This algorithm takes into account factors such as computation time for each pass (F, B, W), communication time, memory limits, and the number of pipeline stages and microbatches.

* **To completely eliminate bubbles during the optimizer step, a post-update validation mechanism is introduced.** This replaces the traditional all-reduce synchronization, allowing for uninterrupted computation flow.

* **Experiments demonstrate the superiority of ZB over baseline methods like 1F1B and interleaved 1F1B.** ZB-2p, the automatically searched schedule with a relaxed memory limit, consistently achieves the highest throughput.

* **ZB-1p, designed for minimal memory usage, offers comparable performance to interleaved 1F1B while requiring less communication.**

* **A more memory-efficient zero-bubble schedule called ZB-V is proposed.** This schedule divides the model into chunks and assigns them to workers in a "V" shape pattern, leading to faster memory clearance and balanced memory usage.

* **ZB-V consistently outperforms 1F1B and ZB-1p while achieving similar performance to ZB-2p with half the memory consumption.**

* **Overall, ZB significantly improves the efficiency of pipeline parallelism, paving the way for faster and more scalable training of large models.**

# 1F1B vs.  Zero Bubble

- 1F1B (one-forward-one-backward) is a pipeline parallelism scheduling strategy where workers alternate between executing one forward pass and one backward pass. This approach was first introduced in the context of asynchronous pipeline parallelism but was later adopted for synchronous settings. 1F1B improves upon the earlier GPipe strategy by offering faster memory clearance, leading to lower peak memory usage with similar bubble ratios.

- Zero Bubble (ZB) distinguishes itself from 1F1B by further refining the scheduling process to completely eliminate pipeline bubbles. While 1F1B reduces bubble size compared to GPipe, ZB aims to achieve zero bubbles, thereby maximizing pipeline efficiency. The key innovation lies in splitting the backward computation into two separate passes: one for calculating gradients with respect to the input (B) and another for calculating gradients with respect to the parameters (W). This decoupling provides more scheduling flexibility, allowing for strategic placement of W passes to fill the gaps (bubbles) in the pipeline.

Here's a table summarizing the key differences between 1F1B and Zero Bubble:

| **Feature**           | **1F1B**                                  | **Zero Bubble (ZB)**                                        |
| --------------------- | ---------------------------------------- | ------------------------------------------------------- |
| **Backward Pass**     | Treated as a single unit                | Split into two parts: **B (input gradient)** and **W (parameter gradient)** |
| **Pipeline Bubbles**  | Reduced compared to GPipe, but still present | Aims to **completely eliminate bubbles**                      |
| **Scheduling Flexibility** | Limited                                  | **Increased** due to the split backward pass                  |
| **Memory Usage**       | **Lower peak memory** than GPipe           | **ZB-H1**: similar to 1F1B, **ZB-H2** and **ZB-V**: higher than 1F1B |
| **Throughput**         | **Lower** than ZB                           | **Higher** than 1F1B                                        | 

