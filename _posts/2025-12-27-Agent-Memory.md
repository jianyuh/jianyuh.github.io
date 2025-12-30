---
layout: post
title: "LLM Agent Memory"
date: 2025-12-27
categories: [Agent, Memory]
tags: [Agent, Memory]
---

Reading the following paper:
- [Memory in the Age of AI Agents: A Survey](https://arxiv.org/pdf/2512.13564)

Restructuring of the field of Agent Memory. It argues that traditional taxonomies (e.g., simple short-term vs. long-term distinctions) are insufficient for modern foundation model-based agents. It introduce a unified framework analyzing memory through three lenses: **Forms** (architecture), **Functions** (purpose), and **Dynamics** (operation). It distinguishes agent memory from related concepts like RAG and Context Engineering, emphasizing memory as the substrate for *continual adaptation* and *long-horizon reasoning*.

### Conceptual Boundaries
Critical distinctions often blurred in the literature:
*   **vs. LLM Memory:** "LLM Memory" often refers to architectural optimizations (e.g., KV cache management, long-context windows) or static weights. Agent memory encompasses these but focuses on persistent, evolving external states that survive across tasks.
*   **vs. RAG:** RAG typically retrieves from static external corpora for a single inference. Agent memory involves dynamic read/write operations where the agent’s own experiences and environmental feedback continuously update the store.
*   **vs. Context Engineering:** Context engineering manages the limited resource of the context window (interface optimization). Agent memory manages the persistent cognitive state (internal substrate).

---

### Technical Taxonomy: The "Forms–Functions–Dynamics" Triangle

#### A. Forms: What Carries Memory?
Memory based on its storage medium and topological structure:

1.  **Token-level Memory:** Discrete, editable units (text, JSON, code).
    *   **Flat (1D):** Sequences/logs (e.g., `MemGPT`). Good for broad recall but lacks structure.
    *   **Planar (2D):** Graphs or trees (e.g., `Mem0g`, `Generative Agents`). Encodes relations and causality.
    *   **Hierarchical (3D):** Multi-layered abstractions (e.g., `HippoRAG`, `G-Memory`). Allows vertical navigation between raw details and high-level summaries.
2.  **Parametric Memory:** Information stored in model weights.
    *   **Internal:** Implemented via pre-training or fine-tuning (e.g., `LMLM`).
    *   **External:** Uses adapters or LoRA to inject knowledge without altering the base model (e.g., `WISE`, `K-Adapter`).
3.  **Latent Memory:** Implicit storage in activation states/vectors.
    *   **Generate:** Auxiliary models create compressed memory tokens (e.g., `MemoRAG`).
    *   **Reuse:** Direct caching of KV pairs (e.g., `Memorizing Transformers`).
    *   **Transform:** Compressing/pruning activations (e.g., `H2O`, `SnapKV`).

#### B. Functions: Why Do Agents Need Memory?
Functional roles are categorized by the *type* of knowledge they preserve:

1.  **Factual Memory (Declarative):** "What the agent knows".
    *   **User Factual:** Maintains consistency in persona and user preferences to prevent "identity drift" (e.g., `MemoryBank`).
    *   **Environment Factual:** Tracks world states, documents, and shared knowledge in multi-agent systems.
2.  **Experiential Memory (Procedural):** "How the agent improves".
    *   **Case-based:** Raw trajectories of past successes/failures for replay (e.g., `Memento`).
    *   **Strategy-based:** Abstracted heuristics and workflows (e.g., `Buffer of Thoughts`).
    *   **Skill-based:** Executable code/APIs distilled from experience (e.g., `Voyager`).
3.  **Working Memory:** "What is active now".
    *   Active workspace management (filtering, folding, state consolidation) to handle infinite horizons within finite context windows.

#### C. Dynamics: How Does It Evolve?
The lifecycle of memory is defined by three operators:

1.  **Formation:** Transforming raw context into memory. Includes *Semantic Summarization*, *Knowledge Distillation* (extracting rules), and *Structured Construction* (building KGs).
2.  **Evolution:** The maintenance phase.
    *   **Consolidation:** Merging fragments into insights (e.g., `TiM`).
    *   **Updating:** resolving conflicts when new facts contradict old ones (e.g., `Fast-Slow` updates).
    *   **Forgetting:** Pruning based on time, frequency, or low semantic value (e.g., `MemGPT` eviction).
3.  **Retrieval:**
    *   **Timing:** Moving from instruction-triggered to autonomous/latent triggering.
    *   **Strategy:** Ranging from lexical (BM25) to graph traversal and generative retrieval.

---

### Key Insights & Emerging Frontiers

**1. The Shift to "Generative Memory"**
The field is moving from *Retrieval-Centric* (finding the right text chunk) to *Generative* approaches. Agents should not just retrieve raw data but actively synthesize/reconstruct memory representations tailored to the current context, similar to how human memory reconstructs rather than replays.

**2. RL for Memory Management**
Current systems often rely on heuristic pipelines (e.g., "summarize every $N$ turns"). It predict a shift toward **e2e Reinforcement Learning**, where agents learn *when* to read/write/forget based on task performance reward signals, effectively internalizing memory management policies.

**3. Offline Consolidation (The "Sleep" Mechanic)**
Human memory relies on sleep for consolidation. Future agents may require "offline" cycles to reorganize episodic logs into semantic knowledge or parametric intuition, resolving the stability-plasticity dilemma without the latency constraints of real-time interaction.

**4. Trustworthiness and Privacy**
Agent memory introduces new attack vectors (e.g., indirect prompt injection into long-term storage). "Trustworthy Memory" requires mechanisms for verifiable forgetting, access control, and granular privacy preservation, especially in shared multi-agent memories.

### Assessment
Establish a vocabulary for *Agentic* memory that is distinct from *LLM* memory. By formalizing the transition from passive storage (RAG) to active, self-evolving cognitive substrates (Experiential/Skill memory), it lays the groundwork for agents that can genuinely improve over time rather than just reset after every context window limits.
