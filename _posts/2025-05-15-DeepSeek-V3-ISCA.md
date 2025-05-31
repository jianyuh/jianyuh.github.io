---
layout: post
title: "DeepSeek-V3's Hardware-Aware Design"
date: 2025-05-15
categories: [DeepSeek]
tags: [DeepSeek]
---

Read [DeepSeek-V3 ISCA paper](https://arxiv.org/pdf/2505.09343v1).

LLMs are rapidly evolving, bringing us closer to the dream of AGI. Models like GPT4o, LLaMa-3, Claude 3.5 Sonnet, and DeepSeek-V3 showcase incredible progress, often following scaling laws where larger models trained on more data with more compute lead to better performance. However, this rapid scaling has revealed critical limitations in current hardware architectures, including memory capacity, computational efficiency, and interconnection bandwidth. Meeting the demands of these colossal models requires massive training clusters, often featuring tens or hundreds of thousands of GPUs, presenting significant cost barriers.

DeepSeek-V3 stands out as a prime example of how effective software-hardware co-design can enable cost-efficient training and inference at scale, even for smaller research resources. Trained on 2,048 NVIDIA H800 GPUs, DeepSeek-V3 achieves state-of-the-art performance by addressing core challenges like memory efficiency, cost-effectiveness, and inference speed. This paper provides insights into the architectural choices and infrastructure that make DeepSeek-V3 efficient and offers valuable reflections on future hardware directions.

## DeepSeek-V3's Design Principles: Tackling Scaling Challenges

DeepSeek-V3 employs a hardware-aware approach, aligning design decisions with hardware constraints for optimal performance and cost. Key innovations include:

- **Multi-head Latent Attention (MLA)**: LLM inference often involves multi-turn conversations where context is cached in a KV cache to avoid recomputing previous tokens. While efficient for computation, the KV cache can become a significant memory bottleneck due to the shift from compute-bound GEMM to memory-bound GEMV operations. MLA tackles this by compressing the Key and Value (KV) representations of all attention heads into a smaller latent vector. This latent vector is cached instead of the full KV pairs, drastically reducing memory consumption. Compared to models using GQA, DeepSeek-V3's MLA reduces KV cache size significantly, needing only 70 KB per token versus 327 KB for Qwen-2.5 72B and 516 KB for LLaMA-3.1 405B. This efficiency is particularly beneficial for long-context processing.

- **Mixture of Experts (MoE)**: For cost-effectiveness, DeepSeek-V3 utilizes the DeepSeekMoE architecture. MoE models allow for a dramatically increased total parameter count while keeping computational requirements modest by selectively activating only a subset of expert parameters per token. DeepSeek-V3 has 671B parameters but activates only 37B per token during training. This contrasts with dense models that require all parameters to be active. The computational cost per token for training DeepSeek-V3 (671B MoE) is approximately 250 GFLOPS, significantly less than the 72B dense model (394 GFLOPS) or the 405B dense model (2448 GFLOPS). MoE models also offer advantages for single-request scenarios and on-premises deployment, requiring fewer activated parameters, which enables faster inference speeds on less expensive hardware compared to dense models of similar capability.

- **Multi-Token Prediction (MTP)**: To increase inference speed, DeepSeek-V3 introduces an MTP framework. Traditional autoregressive models generate one token at a time, creating a sequential bottleneck. MTP allows the model to generate additional candidate tokens at a lower cost and verify them in parallel, accelerating inference without compromising accuracy. MTP modules are lightweight single layers used to predict subsequent tokens, enabling parallel verification. This approach can significantly improve end-to-end generation latency and increase generation speed (TPS - tokens per second). Practice data shows MTP can increase generation TPS by 1.8x. Predicting multiple tokens also increases the inference batch size, which is crucial for boosting computational intensity and hardware utilization for Expert Parallelism (EP). High token output speed is particularly important for reasoning models.

## Hardware-Driven Design and Co-Design Insights

DeepSeek-V3's design was heavily influenced by the characteristics of the hardware used. The system is built on NVIDIA H800 GPUs, which have reduced NVLink bandwidth compared to H100 GPUs, but compensate with multiple high-speed InfiniBand (IB) NICs for scale-out.

- **Hardware-Aware Parallelism**: The architecture avoids Tensor Parallelism (TP) during training due to limited NVLink bandwidth but uses Pipeline Parallelism (PP) with techniques like DualPipe to overlap computation and communication. Expert Parallelism (EP) is accelerated by leveraging the eight 400Gbps IB NICs per node, enabling high-speed all-to-all communication.

- **Node-Limited Routing**: The disparity between faster intra-node (NVLink) and slower inter-node (IB) bandwidth led to a model co-design strategy called Node-Limited Routing for the TopK Expert Selection. By grouping experts onto specific nodes and algorithmically routing tokens to experts on a limited number of nodes (up to 4 in a setup with 8 nodes), the design minimizes the communication bottleneck over IB.

- **Low-Precision Techniques**: The model utilizes FP8 mixed-precision training to lower computational costs and make large-scale training more practical. While beneficial, this highlights limitations in current hardware regarding FP8 accumulation precision and support for fine-grained quantization. Suggestions for future hardware include increased accumulation precision and native support for fine-grained quantization. The paper also explored **Logarithmic Floating-Point Formats (LogFMT)** for communication compression, offering higher precision than FP8 at the same bit width. Although not ultimately used due to current hardware overheads, this exploration suggests the value of native hardware support for compression/decompression.

- **Multi-Plane Network Topology**: DeepSeek-V3 training uses a Multi-Plane Fat-Tree (MPFT) scale-out network. Each GPU and IB NIC pair is assigned to a distinct network plane. This topology allows scaling to many endpoints (e.g., 16,384 GPUs theoretically) using a cost-efficient two-layer structure. MPFT offers advantages like cost efficiency, traffic isolation, lower latency than three-layer trees, and robustness.

## Reflections and Future Hardware Directions

DeepSeek-V3 experience provides valuable insights for the future of AI hardware design. Key areas for improvement include:

- **Scale-Up and Scale-Out Convergence**: Current systems face challenges like SM resource contention from network handling and forwarding. Future hardware should integrate intra-node (scale-up) and inter-node (scale-out) communication into a unified framework. Suggestions include unified network adapters, dedicated communication co-processors, and flexible forwarding/reduce mechanisms integrated into hardware.

- **Addressing Bottlenecks**: CPU bottlenecks and bandwidth contention between different types of traffic on NVLink/PCIe are limitations. Future designs should support dynamic traffic prioritization, integrate NICs into I/O dies, and use dedicated high-bandwidth fabrics like NVLink for CPU-GPU interconnects.

- **Low Latency Networks**: EP is sensitive to latency. While InfiniBand offers lower latency than RoCE, it is more expensive and less scalable. Recommendations for RoCE include specialized low-latency switches, optimized routing policies like Adaptive Routing, and improved congestion control mechanisms. Other approaches: Utilizing technologies like InfiniBand GPUDirect Async (IBGDA) to reduce latency by allowing GPUs to manage the control plane.

- **Robustness**: Failures in interconnects, hardware components, and silent data corruption pose risks, especially in large systems. Advanced error detection beyond ECC, includes comprehensive diagnostic toolkits, and fault-tolerant protocols.

- **Smarter Networks**: Future interconnects need low latency and intelligence. This includes co-packaged optics, lossless networks with advanced congestion control, adaptive routing, and dynamic resource management.

- **Memory-Semantic Communication**: Improving memory-semantic communication by providing hardware-level ordering guarantees (like an acquire/release mechanism) can reduce latency and complexity compared to software-based synchronization.

- **In-Network Computation and Compression**: There are opportunities for in-network optimization for EP's dispatch (multicast) and combine (reduction) stages. Native hardware support for compression formats like LogFMT could also optimize communication.

- **Memory-Centric Innovations**: The memory bandwidth limitation requires innovations including DRAM-stacked accelerators for high bandwidth and low latency, and System-on-Wafer integration for maximum density and bandwidth.
