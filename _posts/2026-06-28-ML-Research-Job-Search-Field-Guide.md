---
layout: post
title: "Preparing for the ML Research Job Search: A Field Guide"
date: 2026-06-28
categories: [Career, Machine Learning]
tags: [Job Search, Interviews, ML Research, Career, PhD, Negotiation, LeetCode, PyTorch]
---

Reading notes synthesizing advice from several ML researchers on navigating the ML/AI industry research job search:

- Silvia Sapora — [*ML Job Interviews: The Ultimate Guide*](https://silviasapora.github.io/blog/ml-interviews.html)
- Alisa Liu — [*Notes on the Industry Job Search*](https://alisawuffles.github.io/blog/job-search/)
- Yong Zheng-Xin — [*Surprising lessons from my research scientist job search*](https://yongzx.github.io/blog/2026/06/24/job-search/)
- Nathan Lambert — [*Thoughts on the job market in the age of LLMs*](https://www.interconnects.ai/p/thoughts-on-the-hiring-market-in)

Below is the cohesive "how-to," followed by the candid details — financial traps, return-offer realities, negotiation scripts, and personal timelines — that rarely make it into clean guides.

---

## Study Planning and Strategy

Preparing for an ML industry job search requires treating the preparation process **like a full-time job**. You should allocate **at least a month of regular study time**. Because interview formats and topics vary wildly, a highly effective meta-strategy is to do **targeted preparation for each specific interview**, treating it like you have three days to cram for a midterm.

To organize your studying, try these methods:

*   **Flashcards:** Write your own *physical* flashcards for ML fundamentals. The act of writing them is half the learning, so avoid downloading pre-made decks. Anticipate questions you might be asked and deeply understand the concepts.
*   **LLM Mock Interviews:** Paste the role and company description into Claude or Gemini and ask it to interview you. Claude is highly recommended for providing fair feedback and surfacing practice questions that often overlap with real interviews.
*   **Draw and Document:** Take continuous notes, draw diagrams, do practice problems, and spend dedicated time solidifying your ML foundations. Always record notes *immediately* after an interview to help with future studying.

---

## Mastering the Technical Interviews

Technical interviews form the bulk of the process and generally evaluate technical skills **much more heavily than past research experience**.

*   **General Coding (LeetCode):** Complete the **Blind 75** and optionally the **NeetCode 150**, focusing on Medium-difficulty questions. **Breadth matters more than depth.** You must know basic patterns (DFS, BFS, Graphs, DP, Binary Search) and implement them quickly. Target under **20 minutes** for an optimal solution; if you are stuck for 15 minutes, look up the solution, understand it, and move on.
*   **ML Coding and Debugging:** This is incredibly common and tests your ability to translate math to code. **Fluency in PyTorch is a must**, though occasionally you may be asked to write backward passes in `numpy`. Practice coding strictly *without* AI assistance to mimic interview settings and build muscle memory. Your baseline should be the ability to implement the following from scratch under time pressure:
    *   an end-to-end transformer,
    *   causal/cross/self-attention,
    *   flash attention,
    *   an attention backward pass,
    *   MLP forward and backward passes,
    *   a simple SGD training loop in PyTorch or JAX.

    Reviewing your own codebase and using "Tensor Puzzles" can help with debugging practice.
*   **Technical and Math Discussions:** You will face rapid-fire knowledge questions testing your breadth (e.g., *"What is 5D parallelism?"*) as well as deep-dive discussions on how to design specific experiments. Math interviews may involve pen-and-paper derivations, so brush up on probability, linear algebra, and calculus.

---

## Preparing for Other Interview Types

*   **Research Discussions and Job Talks:** You will be asked to walk through past projects. Take a step back and reflect on *why* you chose those projects, your insights, and future directions, tailoring your pitch to match the keywords the interviewers are looking for. Job talks are usually shorter than academic ones and should focus on a single paper or cohesive direction. Keep in mind that often only **one or two of your papers truly matter** for getting your foot in the door or for these deep dives.
*   **Behavioral Interviews:** Do not underestimate these. Enumerate memorable stories from your work or PhD and map them to common behavioral questions (e.g., conflicts, receiving feedback) beforehand so you don't go blank trying to retrieve memories under pressure.
*   **Wildcards and Work Trials:** Expect the unexpected. You might be asked about system design, parallel programming (like using `asyncio` for concurrency), or how to use AI agents. Additionally, some companies (especially startups and AI safety labs) use **"work trials"** where you spend up to a week working on an open-ended task with the team. These require massive bandwidth and can stall your prep for other interviews.

---

## Topics to Study

Because you can be asked almost anything, your foundational knowledge must be exceptionally broad. Ensure you cover:

*   **LLMs & Transformers:** Flash Attention, LoRA, MoE, Scaling Laws, RoPE, Tokenisation, RLHF, Decoding techniques, Pretraining/Finetuning.
*   **Reinforcement Learning:** Q-Learning, PPO, GRPO, DPO, Bellman Equations, Policy Gradients, Actor-Critic, On-Policy vs Off-Policy, Markov Decision Processes.
*   **Generative Modelling:** GANs, VAEs, Diffusion (Forward/Reverse processes, SDEs), Flow Matching ODEs, Classifier-Free Guidance.
*   **Applied ML & Infrastructure:** Tensor/Pipeline Parallelism, FSDP, DDP, Mixed-precision training, Gradient checkpointing and accumulation, Profiling.
*   **General ML:** CNNs, RNNs/LSTMs, Bias-Variance Tradeoff, Loss/Activation Functions, Optimizers (AdamW, SGD), Regularisation, Overfitting, KL/Jensen-Shannon Divergence, Metrics (AUC-ROC, F1).
*   **Math & Linear Algebra:** Eigenvectors/Eigenvalues, Hessian, Jacobian, Positive Semi-Definite matrices, Rank/Span.

> Several of these map directly to deep dives on this blog if you want to go beyond flashcard depth: [Scaling Laws]({% post_url 2026-06-25-The-Architecture-of-Scaling-Laws %}), Flash Attention and decoding in the [LLM efficiency notes]({% post_url 2026-06-26-Efficiency-in-LLMs-Fast-Inference-Memory-Bandwidth %}), [speculative decoding]({% post_url 2024-12-15-speculative-decoding %}), and optimizers like AdamW/Shampoo in the [SOAP note]({% post_url 2026-06-23-SOAP-Shampoo-Adam-Eigenbasis %}).

---

## Recommended Resources

*   **Courses:** *Stanford CS336: Language Modeling from Scratch* (highly recommended to organize concepts; Homework 1 on building a transformer is crucial). Gilbert Strang's Linear Algebra lectures on YouTube (can be watched at 2× speed).
*   **Books:** *Designing Machine Learning Systems* by Chip Huyen, *The JAX Scaling Book*, and *Reinforcement Learning* by Sutton & Barto.
*   **Articles/Guides:** "Self-Attention & Transformers", "The Illustrated GPT-2", "Backpropagation", and guides on Policy Gradients and model scaling.

---

## Emotional Preparation and Logistics

The interview process can heavily drain your resilience and cause extreme stress or sleep/eating issues.

*   **Protect your well-being:** Maintain regular exercise, a consistent evening routine, and do not isolate yourself socially. Having dinner with friends on nights before interviews can help reset your mind.
*   **Pre-interview rituals:** Establish a comforting routine before calls, whether that's adjusting your background, doing skincare/makeup, or watching comfort videos. **Sleep is paramount**; attempting technical interviews while sleep-deprived will severely hinder your performance.
*   **Mindset:** Remember that the process is **highly stochastic**. Failing a question does not make you a bad researcher, and your worth is not tied to these interviews. Reading mindset-focused books like *The Now Habit* or *Mindset* before you start can be highly beneficial.
*   **Logistics:** Start by interviewing with companies you care *less* about to practice and calibrate. Try to schedule only **one interview per day** to avoid context-switching and fatigue.

---

## A Note for Junior Candidates

If you are aiming for a junior role without a PhD, the most important trait to demonstrate is a **"fanatical obsession with making progress"** in modeling performance. You can stand out by building a portfolio of high-quality, deeply technical blog posts (avoiding superficial "AI slop") and by making meaningful, sustained open-source contributions to libraries like HuggingFace.

---

# The Candid Details

The guide above is the clean "how-to." Below are the specific personal reflections, financial warnings, and contextual nuances that are easy to omit but matter enormously.

### 1. Resume Benchmarks and "Red Flags"

*   **The Bar for Interviews:** Silvia Sapora notes that to consistently get callbacks at top labs, a rough benchmark is having **3+ first-author papers** (at top venues like ICLR, NeurIPS, ICML) and at least one industry internship.
*   **Negative Signals:** Nathan Lambert warns junior researchers against being a **"middle author on too many papers,"** as it dilutes your perceived depth. He also explicitly warns that writing just one **"AI slop" blog post will kill your application**.

### 2. Compensation, Equity, and Tax Traps (Startups vs. Big Tech)

*   Silvia provides a deep dive into the financial risks of startup equity (specifically under UK tax law). **RSUs** (Big Tech) give you actual shares that count as income, whereas startup **Stock Options** only give you the *right* to buy shares.
*   Crucially, if you leave a startup before it goes public, you might have to spend cash to *exercise* the options and instantly owe income tax on the paper gains—**before you have made any actual money**. She advises discounting startup equity significantly when comparing total compensation.

### 3. The Reality of Return Offers and Ghosting

*   **Return offers aren't guaranteed:** Yong Zheng-Xin points out that unlike standard Software Engineering roles, return offers for Research Scientists are rare. He had to go through the full interview loop for OpenAI despite doing the Astra Fellowship there, and noted return offers at Meta were highly dependent on headcount.
*   **Ghosting happens to everyone:** Even with a stellar CV, Silvia applied to Waymo every six months during her PhD and **never heard back**.

### 4. Advanced Negotiation Tactics

*   **Scripting your calls:** Alisa Liu stresses that negotiation is a totally different skillset from passing interviews. She recommends leaning heavily on friends to calibrate your asks and **writing down verbatim quotes and scripts before *every* recruiter call** so you can advocate for yourself comfortably.
*   **Blind auctions don't always work:** While some advice says to hide competing offers, Silvia found that several companies explicitly demanded **proof (like screenshots)** of her other offers before they would increase their numbers. Furthermore, companies track historical data on candidate choices; bluffing that you might take a startup offer over a top lab like Anthropic won't work if they know candidates rarely do that.

### 5. Should You Drop Out of Your PhD?

*   Nathan has a very specific rule for leaving a PhD early: **only do it if you have an offer to do *modeling* research at a frontier lab** (like Gemini, Anthropic, or OpenAI). If the role is in *product* at a frontier lab, you risk getting absorbed into the corporate machine and losing visibility.
*   Alisa reflects on the emotional toll of the job search and advises students to **cherish the PhD**, noting it is a unique time where your only job is to have good ideas without worrying about the corporate world.

### 6. Cold Emailing Secrets

*   When cold emailing hiring managers, Nathan notes that many leaders intentionally make their emails hard to find **as a filter**. The best cold emails don't just use platitudes; they show you have learned from the person's work and they **"inspire action."** Silvia adds that you should never just repeat your CV, but rather explain exactly why you fit *their specific team*.

### 7. Personal Timelines and Wildcards

*   **Volume of Interviews:** The sheer volume is staggering. Alisa did **57 interviews across 11 companies**, plus 46 recruiter calls.
*   **Irrelevant Interviews:** Yong pivoted specifically to AI safety research, but was surprised to find that the vast majority of his interviews had absolutely **nothing to do with AI safety**, proving you must be a well-rounded ML generalist regardless of your sub-field.
*   **Exploding Offers:** Yong warns that if you receive an **"exploding offer"** (an offer with a very short deadline to accept), you may have to ask other companies to drastically accelerate their timelines, potentially forcing you to do **three back-to-back technical interviews in a single day**.
