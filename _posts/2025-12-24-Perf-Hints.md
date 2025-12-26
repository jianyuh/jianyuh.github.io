---
layout: post
title: "Reading Note on Performance Hints"
date: 2025-12-24
categories: [Performance]
tags: [Performance]
---

Reading the following blog:
- [Performance Hints](https://abseil.io/fast/hints.html) by Jeff Dean and Sanjay Ghemawat.

Knuth’s "premature optimization" quote. While 97% of code requires no optimization, neglecting the **critical 3%** (core libraries, hot paths, inner loops) results in technical debt that is hard to reverse.

*   **The "Flat Profile" Trap:** If performance is ignored during early development, the resulting system often exhibits a "flat profile"—CPU usage is spread evenly across thousands of inefficient operations (allocations, copies, cache misses). There is no single hotspot to fix, forcing engineers to make hundreds of micro-optimizations later.

*   **Back-of-the-Envelope Estimation:** Before implementing complex designs, engineers should estimate latency using resource costs. For example, estimating Quicksort speed by calculating memory bandwidth ($N \times \text{passes}$) versus branch misprediction costs ($N \log N \times \text{mispredict rate}$).

## Hardware Sympathy (2025 Latency Numbers)
Optimization decisions must be grounded in the latency of modern hardware. The 2025 numbers highlight the massive penalty of leaving the CPU cache.

| Operation | Latency (approx.) | Insight |
| :--- | :--- | :--- |
| **L1 Cache Ref** | 0.5 ns | Keep data here. |
| **Mutex Lock (uncontended)** | 15 ns | Expensive in tight loops. |
| **RAM Reference** | 50 ns | ~100x slower than L1. **Cache locality is king.** |
| **Compress 1KB (Snappy)** | 1,000 ns | Compute is often cheaper than I/O. |
| **Read 1MB (SSD)** | 1,000,000 ns | 1 ms. Avoid blocking on I/O. |
| **Packet CA->Netherlands** | 150,000,000 ns | 150 ms. Network is the ultimate bottleneck. |

## Data Structure Selection
It advocates replacing standard STL containers with "hardware-aware" alternatives that prioritize cache locality and reduce pointer chasing.

### A. Maps and Sets
*   **Swiss Tables (`absl::flat_hash_map`):**
    *   *Mechanism:* Uses open addressing and SIMD instructions (SSE2/AVX) to compare 16 hash bytes in parallel (metadata) before checking keys.
    *   *Benefit:* 47% speedup observed in `CodeToLanguage` lookup. Avoids the node-allocation overhead of `std::unordered_map`.
*   **B-Trees (`absl::btree_map/set`):**
    *   *Mechanism:* Stores multiple keys per node.
    *   *Benefit:* Reduces pointer chasing compared to Red-Black trees (`std::map`). Better cache line utilization.
*   **Small/Hybrid Maps:**
    *   `gtl::small_map`: Stores small N (e.g., <10) inline in an array; upgrades to a hash map only when full.
    *   *Replacement:* Replaced `std::set` with `absl::btree_set` for work queues to reduce allocator thrashing.

### B. Vectors and Indices
*   **`absl::InlinedVector`:** Stores small N directly in the object (stack allocation) rather than on the heap. Critical for "hot" temporary vectors.
*   **Indices vs. Pointers:** On 64-bit machines, pointers are 8 bytes. Using `uint32` indices into a flat array saves 50% memory and improves spatial locality.
*   **Bit Vectors:** Use `util::bitmap::InlinedBitVector` instead of `std::vector<bool>` or `hash_set<int>` for dense integer sets. Enables bitwise logic (OR/AND) for set operations.

## Reducing Allocations (The "Silent Killer")
Allocations incur allocator lock contention and scatter data across memory (bad locality).
*   **Hoisting:** Move temporary objects (protos, strings, vectors) *outside* loops. Use `Clear()` to reuse the capacity.
*   **Pre-sizing:** Use `reserve()` on vectors. Avoid `resize()` for complex types as it default-constructs N objects.
*   **Arenas:** Use Arena allocation (especially for Protobufs) to pack unrelated objects into contiguous blocks and enable $O(1)$ bulk deallocation.

## Protocol Buffers (Protobuf) Optimization
Protobufs are convenient but computationally expensive (up to **20x slower** than structs for arithmetic ops).

*   **Type Selection:**
    *   Use `bytes` over `string` for binary data to skip UTF-8 validation.
    *   Use `[ctype=CORD]` for large blobs to prevent linear copying during parsing/serialization.
    *   Use `fixed32` over `int32` for large values (hashes, random IDs) where Varint encoding is inefficient.
*   **Structure:**
    *   Avoid deep nesting (requires recursive allocation/parsing).
    *   Don't use Proto Maps (`map<string, val>`); they are inefficient. Prefer repeated messages.
*   **Parsing:**
    *   **Partial Parsing:** If only a few fields are needed, define a "SubsetMessage" proto containing only those tags to skip parsing the rest.
    *   **View API:** Use `[string_type=VIEW]` to map fields to `absl::string_view` (pointing into the raw buffer) rather than copying to `std::string`.

## Algorithmic Patterns
*   **Graph Cycles:** Inserting nodes in **reverse post-order** makes cycle detection trivial.
*   **Deadlock Detection:** Replaced an $O(\|V\|^2)$ algorithm with a dynamic topological sort ($O(\|V\|+\|E\|)$), enabling scale from 2k to millions of locks.
*   **Hashing vs. Sorting:** Replaced sorted-list intersection ($O(N \log N)$) with hash-table lookups ($O(N)$) for a 21% speedup.

## Execution Efficiency: "Avoiding Work"

### A. Fast Paths
Specialized logic for the common case, falling back to generic logic only when necessary.
*   **Varint Parsing:** Inline the check for `< 128` (1-byte varint). Call a `noinline` slow function for multi-byte integers. This reduces instruction cache pressure.
*   **ASCII Scanning:** Check 8 bytes at a time for the high bit (0x80) to skip UTF-8 processing for ASCII blocks.

### B. Precomputation & Lazy Evaluation
*   **Lazy:** Defer expensive calls like `GetSubSharding` until the operand is confirmed to be relevant.
*   **Precompute:** Pre-calculate lookup tables (e.g., 256-element array for trigrams) instead of computing log-probs in the loop.

### C. Logging
*   **Conditional VLOG:** Logging logic (even disabled) adds branch overhead. Use `if (VLOG_IS_ON(1))` to guard parameter preparation.
*   **Code Bloat:** `ABSL_LOG(FATAL)` generates significant assembly. For non-debug builds, consider `ABSL_DCHECK(false)` or outlining the error string generation.

## Concurrency & Parallelism
*   **Sharding Locks:** A single mutex on a `map` limits throughput. Split the map into $N$ shards (e.g., by `thread_id` or `hash(key)`) with independent locks. This yielded a **69% reduction** in wall-clock time for Spanner's active call tracking.
*   **Critical Sections:** Never perform I/O or RPCs inside a lock. Precompute values outside the lock, acquire, swap pointers/update, and release immediately.
*   **False Sharing:** Use `alignas(ABSL_CACHELINE_SIZE)` to separate frequently written fields (like counters) to prevent cache line invalidation across cores.

## Code Size and Compiler Optimizations
Large code causes instruction cache (i-cache) misses.
*   **Outlining:** Mark error paths and slow paths (like `TF_CHECK_OK` failure strings) as `noinline` or move them to separate functions.
*   **Template Bloat:** Convert template parameters to function arguments (e.g., `template<bool>` $\to$ `bool arg`) to prevent the compiler from generating duplicate binary code for every boolean permutation.
*   **Loop Unrolling:** Manually unroll loops (e.g., CRC32) only when benchmarks prove the compiler isn't doing it effectively.

## API Design for Performance
*   **Bulk APIs:** Replace `Lookup(key)` with `LookupMany(keys)`. This amortizes the cost of virtual function calls and lock acquisition over a batch of items.
*   **Thread-Compatibility:** Design classes to be **thread-compatible** (external sync required) rather than **thread-safe** (internal locks). This avoids forcing single-threaded users to pay for atomic operations they don't need.
