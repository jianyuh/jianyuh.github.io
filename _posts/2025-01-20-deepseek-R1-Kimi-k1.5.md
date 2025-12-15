---
layout: post
title: "Summary on DeepSeek R1 and Kimi k1.5"
date: 2025-01-20
categories: [DeepSeek R1, Kimi k1.5]
tags: [DeepSeek, Kimi]
---

Read [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) and [Kimi-k1.5](https://github.com/MoonshotAI/Kimi-k1.5).

# DeepSeek-R1 Summary

DeepSeek-R1 presents a paradigm shift in training reasoning models (Chain-of-Thought). Unlike previous approaches that rely heavily on Supervised Fine-Tuning (SFT) with human-annotated data or Process Reward Models (PRM), DeepSeek demonstrates that **reasoning capabilities can emerge purely through Reinforcement Learning (RL)** given sufficient scale and appropriate incentives,.

The work introduces two primary models:
1.  **DeepSeek-R1-Zero:** A model trained via pure RL on the base model without any SFT cold start. It exhibits powerful reasoning but suffers from poor readability and language mixing,.
2.  **DeepSeek-R1:** A refined pipeline incorporating a "cold start" (small SFT dataset) followed by multi-stage RL and rejection sampling. It achieves performance comparable to OpenAI-o1-1217.

## Technical Methodology

### A. The Core Algorithm: Group Relative Policy Optimization (GRPO)
Instead of standard PPO (Proximal Policy Optimization), the authors utilize **GRPO**.
*   **Critic-Less Architecture:** Standard RL requires a value function (critic) model, usually the same size as the policy model, doubling memory/compute costs. GRPO eliminates the critic.
*   **Baseline Estimation:** It estimates the baseline using the average reward of a group of outputs sampled from the same query.
*   **Objective Function:**
    The objective maximizes the advantage of sampled outputs relative to the group average, clipped to prevent drastic policy shifts, minus a KL-divergence term to maintain stability relative to the reference model.
    $$J_{GRPO}(\theta) = \frac{1}{G} \sum_{i=1}^{G} \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \dots \right) - \beta D_{KL} \right)$$
    Where the advantage $A_i$ is normalized using the mean and standard deviation of the group's rewards.

### B. DeepSeek-R1-Zero (Pure RL)
*   **Input:** DeepSeek-V3-Base.
*   **Reward System:** Purely rule-based to avoid "reward hacking" common with neural reward models.
    1.  **Accuracy Reward:** Deterministic checks (e.g., LeetCode compilers, math answers in specific boxes).
    2.  **Format Reward:** Enforcing the `<think>` and `</think>` tags to capture the reasoning process.
*   **Result:** The model autonomously increased its "thinking time" (context length), evolving from thousands of tokens to tens of thousands, enabling self-verification and error correction,.

### C. DeepSeek-R1 (The Hybrid Pipeline)
To fix the readability issues of R1-Zero, the authors devised a 4-stage pipeline:
1.  **Cold Start (SFT):** Fine-tuning V3-Base on a small set of high-quality, readable Chain-of-Thought (CoT) data (thousands of samples) to define output structure and readability.
2.  **Reasoning-Oriented RL:** Applying GRPO to the cold-started model. A **language consistency reward** is added here to fix the "language mixing" issue observed in R1-Zero.
3.  **Rejection Sampling & General SFT:**
    *   The checkpoint from Stage 2 generates 600k reasoning samples via rejection sampling (keeping only correct answers).
    *   Combined with 200k non-reasoning samples (writing, QA) to protect general capabilities.
    *   The base model is retrained on this combined dataset.
4.  **RL for All Scenarios:** A final RL stage using reward models (not just rule-based) to align with human preferences for helpfulness and harmlessness.

## Key Insights & "Aha Moments"

**1. Emergence of Self-Verification:**
The most striking finding is the **"Aha Moment"** (Table 3). Without being explicitly taught to verify its work, R1-Zero learned to interrupt its generation, re-evaluate previous steps, and correct errors to maximize the reward,. This suggests that extended test-time compute is not just about length, but about the *quality* of the search trajectory.

**2. The Failure of Process Reward Models (PRM) & MCTS:**
The authors explicitly note unsuccessful attempts with PRMs and Monte Carlo Tree Search (MCTS),.
*   **PRM issues:** Defining fine-grained steps is hard; model-based rewards lead to hacking; manual annotation is unscalable.
*   **MCTS issues:** The search space for token generation is exponentially larger than Chess/Go, making it difficult to optimize the value model iteratively.
*   *Insight:* Simple RL on the final outcome (Rule-Based Reward) proved more effective than complex step-by-step supervision.

**3. Distillation vs. RL on Small Models:**
The paper presents a counter-intuitive finding regarding small models.
*   Running massive RL on a small base model (e.g., Qwen-32B) yields minimal gains.
*   **Distillation is superior:** Fine-tuning a small model on the *outputs* of DeepSeek-R1 (the large teacher) results in state-of-the-art performance for that size class.
*   *Implication:* Large models are required to *discover* reasoning patterns, but small models can effectively *learn* them once discovered.

## Performance & Benchmarks

*   **Reasoning:** DeepSeek-R1 achieves **79.8% Pass@1 on AIME 2024** and **97.3% on MATH-500**, effectively matching OpenAI-o1-1217.
*   **Coding:** Achieves 2029 Elo on Codeforces (96.3 percentile).
*   **Distilled Models:** The **DeepSeek-R1-Distill-Qwen-7B** achieves 55.5% on AIME 2024, surpassing the much larger QwQ-32B-Preview and non-reasoning models like GPT-4o-0513.

## Domain Expert Critique

**Strength: Democratization of RLHF/RL**
The use of **GRPO** is a significant engineering win. By removing the need for a critic model, DeepSeek reduces the VRAM requirements for training large-scale RL models by roughly half. This makes replicating these results more feasible for the open-source community.

**The "Black Box" of Distillation**
The paper confirms that "reasoning" as a capability can be compressed. The fact that a 1.5B model can outperform GPT-4o on Math benchmarks (28.9% vs 9.3% on AIME) solely through SFT on R1 data, suggests that reasoning chains are highly transferrable features, unlike "world knowledge" which requires parameter mass.

**Limitation: The Language Barrier**
A noted drawback is language mixing. R1-Zero, having no SFT constraints, would often switch languages mid-thought. While R1 fixes this via a "language consistency reward", it indicates that without SFT constraints, the "optimal path" in the latent space of the model might naturally traverse cross-lingual tokens.

***

### Analogy for Understanding R1-Zero vs. R1

Think of training a reasoning model like teaching a student to solve complex proofs:

*   **DeepSeek-R1-Zero** is like locking a genius student in a room with a textbook and an answer key, but no teacher. They are told, "Keep trying until you get the right answer." The student eventually figures out brilliant, chaotic methods to solve problems—scribbling all over the walls, sometimes switching languages, or inventing new notations—because all that matters is the final answer. They become incredibly smart, but communicating with them is difficult.

*   **DeepSeek-R1** takes that same student but first gives them a **Style Guide** (Cold Start SFT) on how to write clearly. Then, during the practice phase (RL), the teacher not only checks the answer but also deducts points if the student switches languages randomly or writes illegibly (Language Consistency Reward). The result is a student who is just as brilliant but creates solutions that are structured, readable, and polite.

{% comment %}
The world of LLMs is constantly evolving, with new models and techniques emerging to push the boundaries of what's possible. **DeepSeek-R1**, a new family of reasoning models developed by DeepSeek-AI, represents a significant step forward in this evolution. It explores the key innovations and achievements of DeepSeek-R1 based on the provided research paper excerpt.

## Pushing the Limits of Reasoning with Reinforcement Learning
DeepSeek-R1 is built on the foundation of **reinforcement learning (RL)**, a powerful technique where models learn through trial and error based on rewards. The researchers took a bold step by applying RL directly to the base model **without relying on supervised fine-tuning (SFT) as a preliminary step**. This approach, embodied in **DeepSeek-R1-Zero**, allowed the model to freely explore and discover effective reasoning strategies.
- **DeepSeek-R1-Zero** demonstrated remarkable capabilities, including self-verification, reflection, and the ability to generate long and complex chains of thought (CoT).
- Evaluations on benchmarks like AIME 2024 and MATH-500 showcased significant performance improvements through RL.
However, DeepSeek-R1-Zero also faced challenges, particularly in terms of readability and language consistency. To address these issues and further enhance performance, DeepSeek-AI introduced **DeepSeek-R1**, which incorporated a multi-stage training pipeline:
1. **Cold Start:** A small set of high-quality CoT data was used to fine-tune the base model, providing a more readable and structured starting point for RL.
2. **Reasoning-Oriented RL:** Focused RL training on reasoning-intensive tasks further improved performance, while a language consistency reward encouraged human-friendly outputs.
3. **Rejection Sampling and Supervised Fine-Tuning:** The RL checkpoint was used to collect SFT data, combining reasoning examples with data from other domains to enhance general capabilities.
4. **RL for All Scenarios:** A final RL stage focused on improving helpfulness and harmlessness across all task types.

## Results and Open-Source Contribution
DeepSeek-R1's performance is on par with OpenAI's o1-1217, demonstrating its effectiveness across a wide range of tasks, including:
- **Reasoning:** Achieving state-of-the-art results on benchmarks like AIME 2024, MATH-500, and coding competitions.
- **Knowledge:** Excelling in educational tasks, surpassing other closed-source models on benchmarks like MMLU, MMLU-Pro, and GPQA Diamond.
- **General Capabilities:** Demonstrating strong performance in creative writing, question answering, editing, summarization, and long-context understanding.
**DeepSeek-AI has generously open-sourced DeepSeek-R1-Zero, DeepSeek-R1, and six distilled models (1.5B to 70B parameters) based on Qwen and Llama.** This contribution empowers the research community to build upon these advancements and further explore the potential of reasoning in LLMs.

## Distillation: Making Reasoning Accessible
The researchers also explored **distilling** the reasoning capabilities of DeepSeek-R1 into smaller, more efficient models. By fine-tuning open-source models like Qwen and Llama using the data generated by DeepSeek-R1, they achieved impressive results:
- **DeepSeek-R1-Distill-Qwen-7B outperformed non-reasoning models like GPT-4o-0513 across various benchmarks.**
- **DeepSeek-R1-Distill-Qwen-32B and DeepSeek-R1-Distill-Llama-70B surpassed OpenAI's o1-mini on most benchmarks.**
These findings highlight the effectiveness of distillation in making advanced reasoning capabilities accessible to a wider range of users and applications.

## Looking Ahead: The Future of DeepSeek-R1
The DeepSeek-R1 research paper also outlines future directions for the model's development, including:
- **Improving general capabilities in areas like function calling and complex role-playing.**
- **Addressing language mixing issues when handling queries in languages other than Chinese and English.**
- **Optimizing prompt engineering for better performance.**
- **Enhancing performance in software engineering tasks through further RL training.**
DeepSeek-R1 represents a significant step forward in the quest for more intelligent and capable LLMs. By embracing the power of reinforcement learning and open-source collaboration, DeepSeek-AI has opened up new possibilities for the future of reasoning in artificial intelligence.

{% endcomment %}

# Kimi-k1.5 Summary

Kimi k1.5 is a new multimodal LLM trained with reinforcement learning (RL). The researchers behind Kimi k1.5 aimed to explore RL as a new axis for AI scaling, as language model pretraining with next-token prediction is limited by the amount of available training data.

## Key Ingredients of Kimi k1.5
Kimi k1.5’s design and training revolve around several key ingredients:
- **Long Context Scaling:** The researchers scaled Kimi k1.5’s context window for RL to 128k and found that performance continued to improve as the context length increased. To make training with such a large context window more efficient, they used partial rollouts, which sample new trajectories by reusing large chunks of previously generated ones.
- **Improved Policy Optimization:** Kimi k1.5 uses a formulation of RL with long chain-of-thought (CoT) and a variant of online mirror descent for robust policy optimization.
- **Simplistic Framework:** Combining long context scaling and improved policy optimization methods created a simple but effective RL framework. As the context length increases, the number of search steps increases, allowing Kimi k1.5 to achieve strong performance without complex techniques like Monte Carlo tree search, value functions, or process reward models.
- **Multimodalities:** Kimi k1.5 is trained on both text and vision data, enabling it to reason over both modalities.

## Long-CoT Supervised Fine-tuning and RL
Before beginning RL, Kimi k1.5 underwent a long-CoT supervised fine-tuning stage. The researchers created a small but high-quality dataset of reasoning paths for both text and image inputs using prompt engineering. This primed the model to internalize key cognitive processes like planning, evaluation, reflection, and exploration.
**RL with Kimi k1.5 involved training a policy model to accurately solve problems in a dataset by generating a sequence of intermediate reasoning steps (chain-of-thought) and a final answer.** The model is rewarded for arriving at the correct answer, which encourages it to explore different reasoning paths. The researchers argue that traditional RL methods using value functions for credit assignment might not be suitable for this context because penalizing incorrect reasoning steps could hinder exploration.

## Improving Training Efficiency

The authors used several techniques to make RL training more efficient:
- **Length Penalty:** To prevent the model from generating excessively long responses, they introduced a length reward that penalizes longer responses and promotes shorter ones, especially among responses that arrive at the correct answer.
- **Sampling Strategies:**
  - **Curriculum Sampling:** Starts by training on easier tasks and gradually introduces harder tasks.
  - **Prioritized Sampling:** Focuses on problems with low success rates to improve the model’s weakest areas.

## Long2short: Bringing Long-CoT Benefits to Short-CoT Models
Though long-CoT models perform well, they use more tokens during inference. To address this, the researchers developed several “long2short” methods to transfer insights from long-CoT models to shorter ones. These methods included:
- **Model Merging:** Averages the weights of a long-CoT model and a shorter model.
- **Shortest Rejection Sampling:** Samples the same question multiple times and selects the shortest correct response for supervised fine-tuning.
- **Direct Preference Optimization (DPO):** Uses the shortest correct response as a positive sample and longer responses (both correct and incorrect) as negative samples to train the model.
- **Long2short RL:** Applies a length penalty during a second RL training phase to further penalize overly long responses.

## Results and Conclusions
Kimi k1.5 achieved state-of-the-art results on various reasoning and vision benchmarks in both its long-CoT and short-CoT versions. Notably, **the researchers demonstrated the importance of scaling context length for improving reasoning ability**. Their experiments showed that smaller models could achieve comparable performance to larger ones by leveraging long CoTs optimized through RL, though larger models were more token-efficient. They also found that their RL method was more sample-efficient than ReST, which does not apply negative gradients to penalize incorrect responses.
The authors conclude that scaling context length and improving policy optimization are crucial for continued LLM improvement. They believe that future research should focus on improving credit assignment in RL and reducing overthinking without hindering the model’s ability to explore different reasoning paths. They also see potential in long2short methods for improving short-CoT model performance.
