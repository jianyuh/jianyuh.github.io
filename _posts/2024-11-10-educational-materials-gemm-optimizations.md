---
layout: post
title: "Educational Materials for GEMM Optimizations on CPUs and GPUs"
date: 2024-11-10
categories: [GEMM, optimization, education]
tags: [CPUs, GPUs, HPC, education]
---

## Educational Materials for GEMM Optimizations on CPUs and GPUs

Recently, a colleague reached out to me regarding a paper I published in 2018: [Huang, 2018](https://arxiv.org/pdf/1808.07984). He also pointed me to this directory: [Optimizing SGEMM on NVIDIA Turing GPUs](https://github.com/yzhaiustc/Optimizing-SGEMM-on-NVIDIA-Turing-GPUs?tab=readme-ov-file), where the author referenced my work: "We refer readers to [Huang, 2018](https://arxiv.org/abs/1808.07984) for more details."

This paper was the last project I worked on during my PhD, even after my PhD defense, for a PPoPP submission. Although it was initially rejected, it was eventually accepted by TOMS. There is some history behind this work: in my PhD proposal, I planned to focus on distributed memory GEMM. However, due to several challenges, such as the availability of the IBM Mira Supercomputer and the readiness of collaborative communication primitives, I decided to switch the GEMM optimization platforms to NVIDIA GPUs. At that time, NVIDIA had just released CUTLASS on the V100, which I promptly adopted for this paper, implementing the Strassen algorithm on top of it. This paper is among the first to utilize CUTLASS.

During my PhD, I was quite enthusiastic about contributing to HPC education. As a teaching assistant, I developed step-by-step materials on optimizing GEMM, which later evolved into BLISlab ([BLISlab GitHub](https://github.com/flame/blislab)) and was submitted to ArXiv ([ArXiv Submission](https://arxiv.org/abs/1609.00076)). To my surprise, this tutorial was even cited by the original Triton paper ([Triton Paper](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)).

Before that, my PhD advisor was working on a popular GEMM optimization tutorial for CPUs on older platforms, such as the Intel Pentium 4 processor released in 2008 ([GEMM Optimization on CPUs](https://github.com/flame/how-to-optimize-gemm)). I contributed by converting it to a Wiki format and updating it for more recent hardware. This work was also cited by the TVM tutorial ([TVM Tutorial](https://tvm.apache.org/docs/how_to/optimize_operators/opt_gemm.html)).

Working on these educational materials has been incredibly fulfilling. I am proud that more people can be educated and inspired to explore the frontiers of this field, starting from the foundational basics. It reminds me of my PhD advisor’s educational course titled "From Foundations to Frontiers" ([Course on edX](https://www.edx.org/learn/linear-algebra/the-university-of-texas-at-austin-linear-algebra-foundations-to-frontiers)).
