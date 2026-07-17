# TODO — Component 2.4 HAF-AGE KV-Cache

> **Hierarchical Aggregate Functional Cache with Adaptive Geometric Enrichment**
> Planning mode — awaiting approval before implementation.

---

## Phase 1: Core Data Structures & State (`SPEC §4.1`)

- [ ] **1.1** [`SPEC §4.1`] Define global parameters: `L`, `g`, `d_k`, `d_v`, `W`, `r_max`, `β`, `J_max`
- [ ] **1.2** [`SPEC §4.1`] Implement `HAFCacheBlock` dataclass: `n`, `mu_k`, `mu_v`, `U_k`, `Lambda_k`, `U_v`, `Lambda_v`, `C_kv`, `S_agg`
- [ ] **1.3** [`SPEC §4.1`] Implement `HAFCache` container: `exact_buffer_K`, `exact_buffer_V`, `exact_ptr`, `exact_count`, `S_exact`, `blocks` list, `S_total`
- [ ] **1.4** [`SPEC §4.1`] Implement ring-buffer logic for exact buffer (circular overwrite)

---

## Phase 2: Mathematical Foundations — Spectral Statistics (`SPEC §3.1–3.3`)

- [ ] **2.1** [`SPEC §3.1`] Implement empirical covariance operator `Σ_K = (1/n) K̃^T K̃`
- [ ] **2.2** [`SPEC §3.1`] Implement eigendecomposition wrapper: `Σ = U Λ U^T` with top-`r` truncation
- [ ] **2.3** [`SPEC §3.2`] Implement Hierarchical Exponential Blocking (HEB): level structure, block sizes `B_j = 2^{j-1}W`, rank budgets `r_j = ⌊r / 2^{j-1}⌋`
- [ ] **2.4** [`SPEC §3.3`] Implement `Merge(B_A, B_B) → B_C`: merged means (§3.3 Step 1)
- [ ] **2.5** [`SPEC §3.3`] Implement `Merge`: merged key covariance with rank-truncated eigendecomposition (§3.3 Steps 2–3)
- [ ] **2.6** [`SPEC §3.3`] Implement `Merge`: merged value covariance (§3.3 Step 4)
- [ ] **2.7** [`SPEC §3.3`] Implement `Merge`: merged cross-covariance `C_kv` (§3.3 Step 5)
- [ ] **2.8** [`SPEC §3.3`] Verify merge associativity: `Merge(Merge(A,B), C) == Merge(A, Merge(B,C))`

---

## Phase 3: Adaptive Rank Allocation (`SPEC §3.4`)

- [ ] **3.1** [`SPEC §3.4`] Implement spectral entropy: `p_i = λ_i / Σλ_j`, `H = -Σ p_i log p_i`
- [ ] **3.2** [`SPEC §3.4`] Implement `AdaptiveRank(eigenvalues, β, r_max) → r_eff` using `⌈d_k · (H / H_max) · β⌉`
- [ ] **3.3** [`SPEC §3.4`] Integrate adaptive rank into block compression and merge operations

---

## Phase 4: Clifford-Geometric Enrichment (`SPEC §3.5`)

- [ ] **4.1** [`SPEC §3.5`] Implement truncated grade-{0,1,2} Clifford algebra for `d_k` dimensions
- [ ] **4.2** [`SPEC §3.5`] Implement `embed_multivector(k) → k̃` (grade-1 embedding into `Cℓ_{d_k}^{[0,1,2]}`)
- [ ] **4.3** [`SPEC §3.5`] Implement geometric product `q̃ k̃` returning grade-0 and grade-2 components
- [ ] **4.4** [`SPEC §3.5`] Implement `σ_geo = ||s̃ - s_0||_F` geometric uncertainty metric
- [ ] **4.5** [`SPEC §3.5`] Implement multivector-valued aggregate `S̃` for linear attention
- [ ] **4.6** [`SPEC §3.5`] Verify `⟨q̃, k̃⟩_0 = q·k` (Theorem 6 metric/duality/orientation properties)

---

## Phase 5: Block Compression Protocol (`SPEC §4.4`)

- [ ] **5.1** [`SPEC §4.4`] Implement `ComputeBlockStats(K_block, V_block) → HAFCacheBlock`
- [ ] **5.2** [`SPEC §4.4`] Implement mean computation, centering, and covariance matrices
- [ ] **5.3** [`SPEC §4.4`] Implement key/value eigendecomposition with `AdaptiveRank` integration
- [ ] **5.4** [`SPEC §4.4`] Implement cross-covariance `C_kv = (1/B)(K̃ U_k)^T (Ṽ U_v)`
- [ ] **5.5** [`SPEC §4.4`] Implement `S_agg` precomputation via first-order Taylor (or exact linear attention sum)

---

## Phase 6: Prefill Phase (`SPEC §4.2`)

- [ ] **6.1** [`SPEC §4.2`] Implement `PrefillPopulate(K, V) → HAFCache` for `K ∈ ℝ^{n×d_k}, V ∈ ℝ^{n×d_v}`
- [ ] **6.2** [`SPEC §4.2`] Implement block extraction in `W`-sized chunks
- [ ] **6.3** [`SPEC §4.2`] Implement hierarchical cascade merge after initial block creation
- [ ] **6.4** [`SPEC §4.2`] Compute `S_total = Σ_j S_agg(B_j)` for all blocks
- [ ] **6.5** [`SPEC §4.2`] Set exact buffer with remaining `(n_prompt mod W)` tokens

---

## Phase 7: Decode Phase (`SPEC §4.3`)

- [ ] **7.1** [`SPEC §4.3`] Implement `DecodeStep(C, k_t, v_t, φ) → (C, S_total)`
- [ ] **7.2** [`SPEC §4.3`] Implement exact buffer write + `S_exact` update
- [ ] **7.3** [`SPEC §4.3`] Implement block compression trigger: when `exact_count == W`
- [ ] **7.4** [`SPEC §4.3`] Implement cascade merge: `j=1..J_max`, merge pairs, propagate `S_agg`
- [ ] **7.5** [`SPEC §4.3`] Verify O(1) amortized cost per decode step

---

## Phase 8: Reconstruction Protocol (`SPEC §4.5`)

- [ ] **8.1** [`SPEC §4.5`] Implement `ReconstructPrefix(m) → K̂, V̂`
- [ ] **8.2** [`SPEC §4.5`] Implement per-block reconstruction: sample from Gaussian via `μ + U sqrt(Λ) z`
- [ ] **8.3** [`SPEC §4.5`] Implement exact buffer fallback for recent partial blocks
- [ ] **8.4** [`SPEC §4.5`] Verify reconstruction error bound: `||K - K̂||_F² ≤ B · Σ_{i=r+1}^{d_k} λ_i`

---

## Phase 9: Prefix Caching & Speculative Decoding (`SPEC §4.6–4.7`)

- [ ] **9.1** [`SPEC §4.6`] Implement `ExportPrefix(C, m) → CacheSnapshot`
- [ ] **9.2** [`SPEC §4.6`] Implement snapshot import for cross-sequence sharing (copy-on-write)
- [ ] **9.3** [`SPEC §4.7`] Implement speculative decoding draft: `Speculate(C, draft_tokens)`
- [ ] **9.4** [`SPEC §4.7`] Implement speculative verification + rollback: `Verify(C, checkpoint, accepted_mask)`

---

## Phase 10: Integration with Upstream Components (`SPEC §6`)

- [ ] **10.1** [`SPEC §6.1`] Define interface: `DecodeStep` receives `K_t[g], V_t[g]` from Component 2.1 per group
- [ ] **10.2** [`SPEC §6.2`] Define interface: `S_total` exposed to Component 2.2 for linear attention mode
- [ ] **10.3** [`SPEC §6.2`] Define interface: `σ_geo` exposed to Component 2.2 for RSA-UQ-ACD modulation
- [ ] **10.4** [`SPEC §6.3`] Define interface: `S_total` seeds `SGKDState.S_i` in Component 2.3
- [ ] **10.5** [`SPEC §6.3`] Implement GQA expansion: `repeat_interleave` from `g` groups to `h` heads

---

## Phase 11: Complexity Verification (`SPEC §5`)

- [ ] **11.1** [`SPEC §5`] Verify O(n·d²) prefill time bound
- [ ] **11.2** [`SPEC §5`] Verify O(1) amortized decode step time
- [ ] **11.3** [`SPEC §5`] Verify O(1) space: `W(d_k+d_v) + 2·r_max·(d_k+d_v+r_max)` bounded independent of `n`
- [ ] **11.4** [`SPEC §5`] Measure memory at 1M tokens: must not exceed `1.5×` memory at `W` tokens

---

## Phase 12: Test Suite (`SPEC §7`)

### 12.1 Correctness Tests (`§7.1`)
- [ ] **12.1.1** Aggregate Equivalence: `S_total == Σ φ(k_i) v_i^T` within 1e-4
- [ ] **12.1.2** Causal Consistency: incremental checksum matches per-step update
- [ ] **12.1.3** Reconstruction Bound: Eckart-Young error bound verified per block
- [ ] **12.1.4** Merge Associativity: `Merge(Merge(A,B), C) == Merge(A, Merge(B,C))`
- [ ] **12.1.5** Geometric Enrichment Sanity: `⟨q̃,k̃⟩_0 == q·k`

### 12.2 State Management Tests (`§7.2`)
- [ ] **12.2.1** Exact Buffer Overflow: `W+1` tokens promote oldest to hierarchical block
- [ ] **12.2.2** Empty Cache: `S_total == 0`
- [ ] **12.2.3** Prefill-Decode Continuity: decode step includes all prefill tokens
- [ ] **12.2.4** Reset: `C.reset()` clears all state
- [ ] **12.2.5** Determinism: bit-exact reproducibility

### 12.3 Memory Tests (`§7.3`)
- [ ] **12.3.1** Constant Memory: 1M tokens ≤ 1.5× mem after `W` tokens
- [ ] **12.3.2** No Reallocation: zero heap allocations during decode
- [ ] **12.3.3** GQA Correctness: cache memory == `g/h` of MHA baseline
- [ ] **12.3.4** CPU Cache Locality: block matrices ≤ 32KB (L1 fit)

### 12.4 Performance Tests (`§7.4`)
- [ ] **12.4.1** Decode Latency: < 10% variance from 1k to 1M tokens
- [ ] **12.4.2** Throughput: 1M context ≥ 95% of 1k context tokens/sec
- [ ] **12.4.3** Prefill Speed: ≤ 2× standard dense cache time
- [ ] **12.4.4** Merge Amortization: merge time ≤ 5% of total decode time

### 12.5 Integration Tests (`§7.5`)
- [ ] **12.5.1** Prefix Caching: export/import retains `S_total`
- [ ] **12.5.2** Speculative Rollback: partial accept state matches reference
- [ ] **12.5.3** Cross-Attention Export: compressed block as memory vector
- [ ] **12.5.4** Clifford Attention Quality: ≥ 2% improvement on geometric reasoning

---

## Legend

| Status | Meaning |
|--------|---------|
| `[ ]`  | Pending |
| `[/]`  | In progress |
| `[x]`  | Completed |
| `[-]`  | Cancelled |
