# Component 2.4 — KV-Cache

## Executive Summary

The KV-Cache is the inference-time memory structure that stores previously computed Key and Value tensors so they do not need to be recomputed for each new generated token. Without it, autoregressive decoding would be O(n²) compute per token step — recomputing all past K and V from scratch. With it, each new token only requires a single forward pass through the new token's position, reading the cached K,V for attention.

The KV-Cache trades compute for memory. The tradeoff is severe at long context: for LLaMA-3 8B at 128k tokens, the KV-cache alone consumes ~16 GB of GPU RAM with GQA, and ~64 GB without. This is the primary memory bottleneck in LLM serving.

This component has no training-time existence — it is a pure inference mechanism. Its complexity profile is O(n · d · L) space, growing linearly and without bound as generation continues.

---

## Mathematical Breakdown & Specification

### Definitions

| Symbol | Meaning |
|---|---|
| `t` | Current decoding step (token index being generated) |
| `L` | Number of Transformer layers |
| `h` | Number of KV heads (= g in GQA, = h in MHA) |
| `d_k` | Key head dimension |
| `d_v` | Value head dimension |
| `K_cache[l]` | Cached key tensor for layer `l` |
| `V_cache[l]` | Cached value tensor for layer `l` |

---

### Prefill Phase

Before decoding begins, the full prompt of length `n_prompt` is processed:

```
For each layer l = 1..L:
  X_l = transformer_block_l(X_{l-1})    (full sequence forward pass)
  K_cache[l] = X_l · Wk                 K_cache[l] ∈ ℝ^{n_prompt × h × d_k}
  V_cache[l] = X_l · Wv                 V_cache[l] ∈ ℝ^{n_prompt × h × d_v}
```

At the end of prefill: KV-cache is fully populated for all `n_prompt` positions.

Prefill compute: O(n_prompt² · d) — full quadratic attention over the prompt. This is unavoidable.

---

### Decode Phase (Autoregressive Generation)

At each step t (generating token t):

```
Input: single new token embedding xₜ ∈ ℝ^d

For each layer l = 1..L:
  1. Compute new K, V for position t:
     kₜ = xₜ · Wk        kₜ ∈ ℝ^{h × d_k}
     vₜ = xₜ · Wv        vₜ ∈ ℝ^{h × d_v}

  2. Append to cache:
     K_cache[l] ← concat(K_cache[l], kₜ, dim=seq)
     V_cache[l] ← concat(V_cache[l], vₜ, dim=seq)

  3. Compute query:
     qₜ = xₜ · Wq        qₜ ∈ ℝ^{h × d_k}

  4. Attention against full cache (length t+1):
     scores = qₜ · K_cache[l]ᵀ / √d_k    ∈ ℝ^{h × (t+1)}
     weights = softmax(scores)              ∈ ℝ^{h × (t+1)}
     oₜ = weights · V_cache[l]             ∈ ℝ^{h × d_v}

  5. Output projection + residual (standard block ops)
```

At decode step `t`, attention is over `t+1` positions — growing linearly with each step.

---

### Memory Layout

The cache is typically a pre-allocated tensor to avoid dynamic memory allocation during generation:

```
K_cache ∈ ℝ^{L × max_seq_len × h × d_k}   (pre-allocated)
V_cache ∈ ℝ^{L × max_seq_len × h × d_v}   (pre-allocated)
```

During decode, a pointer tracks the current fill position. New K,V are written at position `t` and the attention call uses `K_cache[:t+1]`.

---

### Memory Footprint Formula

```
bytes_per_token = L × h × (d_k + d_v) × bytes_per_element
               = L × h × 2 × d_k × bytes_per_element    (assuming d_k = d_v)
               = L × h × 2 × (d/h) × bytes_per_element
               = L × 2d × bytes_per_element
```

**Per token, total across all layers:**
```
= L × 2 × d × dtype_bytes
```

| Model | L | d | h_kv (GQA groups) | dtype | Per token |
|---|---|---|---|---|---|
| LLaMA-2 7B | 32 | 4096 | 32 (MHA) | bf16 | 32×2×4096×2 = 524 KB |
| LLaMA-3 8B | 32 | 4096 | 8 (GQA) | bf16 | 32×2×(8×128)×2 = 131 KB |
| LLaMA-3 70B | 80 | 8192 | 8 (GQA) | bf16 | 80×2×(8×128)×2 = 328 KB |
| GPT-3 175B | 96 | 12288 | 96 (MHA) | fp16 | 96×2×12288×2 = 4,718 KB ≈ 4.6 MB |

**At 128k context (LLaMA-3 8B, GQA):**
```
128,000 × 131 KB = 16.8 GB — just the KV-cache
```

This competes with the model weights themselves (8B params × 2 bytes = 16 GB).

---

### Paged Attention (vLLM)

Standard KV-cache pre-allocates `max_seq_len` upfront — wasteful for variable-length requests.

PagedAttention (Kwon et al., 2023) manages KV-cache like virtual memory:
```
KV-cache divided into fixed-size pages (blocks) of K tokens each
Each sequence gets a page table mapping logical positions to physical blocks
Blocks are allocated on-demand and freed when generation completes
Non-contiguous physical blocks: attention reads via indirect lookup
```

Enables:
- Near-zero memory waste from pre-allocation
- Memory sharing for parallel decoding (beam search, speculative decoding)
- Higher serving throughput via better GPU memory utilization

Does not change the asymptotic complexity: still O(n · d · L) total memory per sequence.

---

### Speculative Decoding Interaction

Speculative decoding uses a small draft model to generate k tokens speculatively, then verifies with the main model in a single forward pass. KV-cache must handle:
- Rollback: discard cache entries for rejected speculative tokens
- Branching: multiple candidate continuations sharing a common prefix

Both are managed via the page table in PagedAttention or via explicit sequence-length rollback in simpler implementations.

---

### Quantized KV-Cache

To reduce memory at the cost of slight precision loss:
```
Standard: bf16 → 2 bytes per element
INT8 KV:  8-bit quantized K,V → 1 byte per element (2× reduction)
INT4 KV:  4-bit quantized K,V → 0.5 bytes per element (4× reduction)
```

Quantization applied per-head or per-channel. Quality impact is mild for INT8, noticeable for INT4 on long context.

Used in production serving (TensorRT-LLM, llama.cpp).

---

## Complexity Analysis

| Operation | Time | Space |
|---|---|---|
| Prefill K,V compute | O(n_prompt · d²) per layer | O(n_prompt · h · d_k) per layer |
| Decode: compute kₜ, vₜ | O(d²) per step per layer | O(h · d_k) new elements |
| Decode: attention at step t | O(t · h · d_k) = O(t · d) per layer | O(t · h · d_k) = O(t · d) read |
| **Decode total (n steps)** | **O(n² · d · L)** summed over all steps | **O(n · d · L)** cache size |
| Cache storage | — | **O(n · d · L)** — grows without bound |

**Time:** Decode attention at step t costs O(t·d) — summed over t=0..n gives O(n²·d·L) total, same as prefill without cache. The cache saves constant factor but not asymptotic class.

**Space:** O(n·d·L) — linear in sequence length, unbounded. At 128k context this dominates GPU RAM.

🔴 **KV-Cache is the primary space bottleneck. O(n·d·L) with no upper bound.**

---

## Detailed Test Specification

### Correctness Tests

- [ ] **Cache equivalence:** `decode_with_cache(prompt + [t₁, t₂, t₃])` produces identical logits to `prefill(prompt + [t₁, t₂, t₃])` for all token positions — bit-exact or within bf16 tolerance
- [ ] **Causal consistency:** At decode step t, attention at layer l uses exactly `K_cache[l][0:t+1]` and `V_cache[l][0:t+1]`
- [ ] **No future leakage:** Cached values from future steps are never read — enforce via length tracking, verify by corrupting future cache positions and checking no effect
- [ ] **Append correctness:** After appending kₜ, `K_cache[l][t] == kₜ` exactly

### State Management Tests

- [ ] **Reset:** After `cache.reset()`, all positions are cleared and `cache.size == 0`
- [ ] **Overflow guard:** Attempting to decode beyond `max_seq_len` raises an explicit error, not silent corruption
- [ ] **Prefill → decode continuity:** First decode step after prefill attends correctly to all `n_prompt` prefill positions
- [ ] **Incremental append:** K_cache grows by exactly 1 position per decode step

### Memory Tests

- [ ] **Pre-allocated size:** Total cache memory == `L × max_seq_len × h × (d_k + d_v) × dtype_bytes` within 5%
- [ ] **No reallocation during decode:** GPU memory usage does not spike during decode (pre-allocated)
- [ ] **GQA cache size:** With GQA (g groups), cache size == `L × max_seq_len × g × (d_k + d_v) × dtype_bytes` — verifies KV head count

### Performance Tests

- [ ] **Decode latency scaling:** Latency per decode step grows linearly with t (O(t) attention) — verify empirically at t=1k, 2k, 4k, 8k
- [ ] **Throughput degradation:** Token/sec at t=100k is ≤ 50% of token/sec at t=1k (memory bandwidth bound)

### Integration Tests

- [ ] **PagedAttention compatibility:** With PagedAttention, non-contiguous page layout produces same output as contiguous layout
- [ ] **GQA broadcast:** Expanded K,V (h heads from g groups) in decode attention produces same result as g-head attention with manual head selection

---

## Checklist

- [ ] Executive summary written and reviewed
- [ ] Prefill phase documented
- [ ] Decode phase step-by-step documented
- [ ] Memory layout and pre-allocation documented
- [ ] Memory footprint formula with per-model table
- [ ] PagedAttention documented
- [ ] Speculative decoding interaction documented
- [ ] Quantized KV-cache documented
- [ ] Complexity analysis complete (time and space)
- [ ] All correctness tests written
- [ ] All state management tests written
- [ ] All memory tests written
- [ ] All performance tests written
- [ ] All integration tests written
- [ ] Document reviewed against Transformer.md for consistency
