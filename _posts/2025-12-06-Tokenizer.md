---
layout: post
title: "Tokenizer Learning"
date: 2025-12-06
categories: [Tokenizer]
tags: [Tokenizer]
---

Watch Andrey Karpathy's YouTube Video:
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE).


Many "hallucinations" or failures—poor arithmetic, inability to reverse strings, and erratic behavior with specific keywords—are incorrectly attributed to architecture changes when they are actually artifacts of tokenization.

---

### **1. Core Algorithmic Foundation: Byte-Level BPE**

Modern LLMs (GPT sequence, Llama) predominantly use **Byte Pair Encoding (BPE)** over UTF-8 encoded strings.

*   **The Unicode Problem:** We cannot tokenize raw Unicode code points directly. The standard defines ~150,000 characters, which would result in an unwieldy vocabulary size and sparse embedding matrix.
*   **The Byte Solution:** Text is first encoded into UTF-8 bytes. While this shrinks the base vocabulary to 256 (0-255), it elongates the sequence significantly. BPE solves this by iteratively merging the most frequent adjacent pairs of bytes/tokens into new tokens.
*   **Training Loop:**
    1.  Calculate frequency of all adjacent pairs (e.g., `(101, 32)` corresponding to `'e' + ' '`).
    2.  "Mint" a new token (e.g., 256) to represent the most frequent pair.
    3.  Replace all occurrences in the dataset.
    4.  Repeat until the desired vocabulary size (hyperparameter) is reached.

**Key Constraint:** The tokenization training set is distinct from the LLM training set. If the tokenizer is trained on different data (e.g., heavy code or Reddit) than the model, it creates "dead" tokens (see *Section 4*).

---

### **2. Architecture Specifics: GPT-2 vs. GPT-4**

The raw BPE algorithm is insufficient because it creates suboptimal merges across semantic boundaries (e.g., merging a letter with a punctuation mark). To fix this, OpenAI employs **Regex-based Pre-tokenization** to split text into chunks *before* BPE is applied.

#### **GPT-2 Implementation**
*   **Vocab Size:** 50,257 tokens.
*   **Regex Logic:** Splits by contractions (`'s`, `'t`), letters, numbers, and punctuation.
*   **Weaknesses:**
    *   **Case Sensitivity:** It lacked the `(?i)` flag. `'s` (lowercase) was a separate token logic from `'S` (uppercase), leading to inefficient allocation.
    *   **Inefficient Numerals:** It frequently split multi-digit numbers arbitrarily (e.g., 1234 might become `1`, `234` or `12`, `34`), hampering arithmetic performance.

#### **GPT-4 (cl100k_base) Implementation**
*   **Vocab Size:** ~100,277 tokens.
*   **Improvements:**
    *   **Case Insensitivity:** Uses `(?i:'s|'t|...)` to handle contractions consistently across cases.
    *   **Strict Number Merging:** The regex `\p{N}{2,}` ensures numbers are not split into single digits as aggressively, though they are still treated as distinct tokens from letters.
    *   **Whitespace:** Merges multiple spaces more aggressively than GPT-2, likely due to the high volume of Python code (indentation) in the training set.

---

### **3. Comparative Analysis: Tiktoken vs. SentencePiece**

There are two dominant philosophies in the industry:

| Feature | **Tiktoken (OpenAI)** | **SentencePiece (Google/Llama)** |
| :--- | :--- | :--- |
| **Input Level** | Operates strictly on **UTF-8 Bytes**. | Operates on **Unicode Code Points** first. |
| **Fallback** | None needed (bytes cover everything). | **Byte Fallback**: Unknown chars map to UTF-8 bytes. |
| **Mechanism** | Regex split $\rightarrow$ Byte BPE. | Unigram or BPE on chars $\rightarrow$ Fallback to bytes for rare chars. |
| **Pros** | Cleaner implementation; universal. | Better compression for non-English (e.g., CJK). |
| **Cons** | Inefficient for non-English (3x more tokens for Korean vs English). | Complex configuration (normalization, "sentences"). |

**Technical Insight:** Llama 2 uses SentencePiece with `byte_fallback=True`. If it encounters a rare character not in its vocabulary, it decomposes it into byte tokens (e.g., `<0xEC>`), ensuring no "unknown" (`<unk>`) tokens are fed to the model, provided the vocabulary has room for the 256 base bytes.

---

### **4. Pathology: When Tokenization Breaks the Model**

The source identifies several critical failure modes traceable to tokenization:

*   **The "SolidGoldMagikarp" Effect (Unallocated Embeddings):**
    *   *Issue:* Weird tokens cause the model to crash or hallucinate insults.
    *   *Root Cause:* The *tokenizer* dataset contained Reddit data with high-frequency user strings (e.g., "SolidGoldMagikarp"), creating a specific token. The *LLM* training dataset did not contain this string.
    *   *Result:* The embedding vector for that token remains at its random initialization (never updated via backprop). At inference, calling this token triggers undefined behavior.

*   **Trailing Whitespace & "Unstable" Tokens:**
    *   *Issue:* Prompts ending in space trigger warnings or poor performance.
    *   *Root Cause:* In GPT tokenizers, `` " world"`` is a single token. However, `` `` (space) + ``"world"`` are two tokens. A trailing space forces the model to predict the *next* token starting from a fragmented state (the isolated space), which is distributionally rare.

*   **String Manipulation & Spelling:**
    *   *Issue:* GPT-4 cannot reverse `.DefaultCellStyle`.
    *   *Root Cause:* The string is a single token (ID 98518). The attention mechanism sees one "atom" and cannot attend to the internal character consituents. Pre-segmenting the string allows the model to solve it.

### **5. Practical Implementation & Special Tokens**

*   **Special Token Handling:** Tokens like `<|endoftext|>` or FIM (Fill-In-Middle) markers bypass the BPE logic entirely. They are injected via dictionary lookup or a special regex pass.
*   **Model Surgery:** To add special tokens (e.g., for Chat interfaces):
    1.  Resize the embedding matrix (`vocab_size` $\rightarrow$ `vocab_size + N`).
    2.  Resize the final projection layer (`lm_head`).
    3.  Initialize new rows randomly and train (often freezing the rest of the model).
*   **Efficiency:** For structured data, **YAML** is significantly more token-efficient than JSON (99 tokens vs. 116 tokens for identical data), saving context window and compute costs.

