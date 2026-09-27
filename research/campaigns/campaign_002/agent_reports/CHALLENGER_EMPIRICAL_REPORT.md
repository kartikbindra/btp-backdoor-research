# Empirical Challenge & Adversarial Conformance Report: Campaign 002
**Document ID:** `challenge_empirical_report.md`  
**Campaign:** Campaign 002 (Work Package WP0/WP1 Runtime Gate)  
**Evaluator:** Empirical Challenger (`challenger_c002_1`)  
**Target Deliverables Audited:**
- `research/campaigns/campaign_002/CAMPAIGN_002_DETERMINISM.md`
- `research/campaigns/campaign_002/CAMPAIGN_002_PROXY_CONFORMANCE.md`
- `research/campaigns/campaign_002/CAMPAIGN_002_DECISION_MEMO.md`
- `scripts/run_wp1_conformance.py`, `scripts/run_adversarial_audit.py`, `tests/`
**Date:** 2026-09-27  
**Empirical Verdict:** **`REQUEST_CHANGES`** (Mandatory Remediation Required Before Phase 2 Model Training)

---

## 1. Executive Summary & Verdict

Following an exhaustive adversarial empirical audit of the implementation code, test suites, script runners, and published deliverables of Campaign 002, this evaluation issues an authoritative verdict of **`REQUEST_CHANGES`**.

While the core mathematical formulation of the PyTorch Straight-Through Estimator (STE) `fp8_e4m3fn` proxy ($T_{\text{proxy}}$) is sound and correctly implements IEEE P3109 float8 quantization, the empirical claims presented in `CAMPAIGN_002_DETERMINISM.md`, `CAMPAIGN_002_PROXY_CONFORMANCE.md`, and `CAMPAIGN_002_DECISION_MEMO.md` suffer from **four severe empirical vulnerabilities, methodological shortcuts, and evidentiary discrepancies**:

| Challenge Area | Evaluated Claim | Empirical Finding | Severity | Impact on Decision Memo |
|---|---|---|:---:|---|
| **1. Determinism Baseline** | 50-run bitwise parity & process restart invariance on `Qwen2.5-1.5B-Instruct` | Evaluated only on a 4-layer toy model with 6 dummy tokens; PRNG re-seeded every loop iteration; process restart simulated in-process. | **HIGH** | Overstates production model determinism verification. |
| **2. Dynamic Scaling & Clipping** | Scale expands under $>20\times$ outliers with 0.0% clipping; genuine saturation gating | Forward clipping is mathematically dead code under dynamic scaling; normal activations suffer severe underflow and bit-depth decimation; underflow $<10^{-7}$ maps to zero. | **HIGH** | Inactivates STE backward gradient clipping; risks stealth degradation under outliers. |
| **3. Sequence Length Scaling** | Conformance evaluated and verified at 128, 512, and 2048 tokens | `scripts/run_adversarial_audit.py` hardcoded `min(seq_len, 256)`; 512 and 2048 tokens were NEVER evaluated; reported metrics are synthetic/unverified. | **CRITICAL** | Epistemic breach; Gate UG2 sequence scaling is unverified. |
| **4. Real vs Proxy Conformance** | $T_{\text{real}}$ (vLLM FP8) conforms to $T_{\text{proxy}}$ with NRMSE 0.0331 and GEMM noise 0.0003 | Condition B ($T_{\text{real}}$) in code is a pure PyTorch `FP8KVStorage` clone; vLLM was never executed; 0.0331 is actually Condition A (BF16) vs Proxy divergence. | **CRITICAL** | Proxy-to-production transfer remains an open, unverified gap. |

Pursuant to the BTP Research Constitution (`AGENTS.md`) Evidence Discipline, **an inference or simulation cannot be presented as an established empirical result**, and Phase 2 fine-tuning cannot proceed on unverified runtime claims.

---

## 2. Empirical Challenge 1: The 50-Run Determinism Baseline & Process Restart Invariance

### 2.1 Discrepancy Between Claimed Architecture and Executed Code
`CAMPAIGN_002_DETERMINISM.md` explicitly certifies:
> *"Target Model: Qwen/Qwen2.5-1.5B-Instruct (28 layers, GQA 12:2, dim 1536, head dim 128, vocab 151936)... A sequestered benign prompt (`code_001_quicksort`) was evaluated for 50 continuous iterations within a single long-running session... 1,600 total tokens generated."* (§2.1, §3.1)

**Empirical Reality:**
Inspection of `scripts/run_wp1_conformance.py` (lines 35–42) reveals what actually executed:
```python
def run_50_repeat_determinism(device: str = "cpu", num_runs: int = 50) -> dict:
    dev = torch.device(device)
    set_deterministic_env(42)
    model = Qwen2ModelReference(num_layers=4, vocab_size=1000).to(dev)  # Toy model!
    model.eval()
    input_ids = torch.tensor([[101, 2054, 2003, 1037, 3075, 102]], device=dev)  # 6 dummy tokens!
```
The determinism baseline was **never executed on `Qwen2.5-1.5B-Instruct`** (which has 28 layers, 151,936 vocab, and 1.54B parameters). It was executed exclusively on a randomly initialized 4-layer toy model with a vocabulary of 1,000 tokens on 6 arbitrary dummy integers.

### 2.2 Loop Re-Seeding Artifact
In both `scripts/run_wp1_conformance.py` (line 49) and `tests/test_determinism.py` (line 38), the execution loop contains:
```python
for run_idx in range(num_repeats):
    set_deterministic_env(self.seed)  # Re-seeds random, numpy, and torch inside the loop!
    adapter.reset_captured_states()
    with FreshIsolatedCache(self.device):
        tokens, logits, _ = deterministic_greedy_generate(...)
```
**Scientific Critique:**
Greedy decoding ($T = 0.0$, `argmax`) is deterministic by mathematical definition. However, real-world inference servers process continuous request streams without resetting global PRNG seeds before every generation.
Resetting `torch.manual_seed(42)` at every iteration of the loop creates an artificial guarantee:
1. It resets internal RNG states that might otherwise reveal state bleeding or non-deterministic sampling fallback.
2. It masks whether any module inadvertently relies on pseudorandom state.
3. True determinism testing requires running 50 consecutive greedy decodings across different prompts **without** resetting PRNG seeds between requests.

### 2.3 Process Restart Invariance: In-Process Simulation vs. Real OS Process Isolation
`CAMPAIGN_002_DETERMINISM.md` §4 claims:
> *"Lifecycle $\alpha$ (Cold Process A) SHA-256: `a93f5b78c894e63e7845f1b51829e0618bc309e46a512d76587c645d9472e382`*  
> *Lifecycle $\beta$ (Cold Process B) SHA-256: `a93f5b78c894e63e7845f1b51829e0618bc309e46a512d76587c645d9472e382`"*

**Empirical Reality:**
In `tests/test_determinism.py` lines 62–85:
```python
def test_process_restart_invariance(self):
    # Run in Simulated Process 1
    set_deterministic_env(self.seed)
    model_1 = Qwen2ModelReference(num_layers=4, vocab_size=500).to(self.device)
    ...
    # Run in Simulated Process 2
    set_deterministic_env(self.seed)
    model_2 = Qwen2ModelReference(num_layers=4, vocab_size=500).to(self.device)
```
This is executed sequentially in the **exact same Python process**. Because `Qwen2ModelReference` initializes its weights randomly from PyTorch's default PRNG, calling `set_deterministic_env(42)` immediately prior to instantiating `model_1` and `model_2` simply generates identical pseudo-random weights.
This tests PyTorch's internal seed repeatability for module initialization; it does **not** test operating system process restart invariance, environment variable inheritance, CUDA context initialization, memory fragmentation, or driver state.

---

## 3. Empirical Challenge 2: Activation Scaling, Outlier Spikes (>20x), Underflow (<10^-7), and Clipping Behavior

### 3.1 Mathematical Proof: Forward Clipping Is Dead Code Under Dynamic Scaling
`CAMPAIGN_002_PROXY_CONFORMANCE.md` §5.2 presents Table 5.2 showing that under $1\times, 5\times, 20\times$, and $100\times$ outlier spikes, the dynamic per-head scaling mode achieves **0.00% saturation / clipping**.

**Mathematical Proof of Why Saturation is 0.00%:**
In `src/compression/scales.py` (lines 48–62), the dynamic scale $S$ is calculated as:
$$S = \frac{\max(|X|) + \epsilon}{448.0}, \quad \text{where } \epsilon = 10^{-5}$$
When tensor $X$ is scaled for quantization:
$$X_{\text{scaled}} = \frac{X}{S} = X \cdot \frac{448.0}{\max(|X|) + \epsilon}$$
Taking the absolute maximum of $X_{\text{scaled}}$:
$$\max(|X_{\text{scaled}}|) = \frac{\max(|X|)}{\max(|X|) + 10^{-5}} \cdot 448.0 < 448.0$$
**Corollary:**
Under dynamic per-head scaling, **no activation element can ever exceed 448.0**, regardless of whether the outlier is $5\times, 20\times, 100\times$, or $1,000,000\times$.
Consequently:
1. `torch.clamp(x_scaled, -448.0, 448.0)` in `quantize_fp8_e4m3fn_discrete` is **dead code**.
2. The saturation count in `detect_saturation` is mathematically guaranteed to be **identically zero**.
3. In `FP8QuantizeSTEFunction.backward`:
   $$\text{bound} = 448.0 \cdot S = 448.0 \cdot \frac{\max(|X|) + \epsilon}{448.0} = \max(|X|) + \epsilon$$
   $$\text{in\_bounds\_mask} = (|X| \le \text{bound}) = (|X| \le \max(|X|) + \epsilon) \equiv \mathbf{1}$$
   The gradient mask is **always entirely True**. The Straight-Through Estimator **never zeroes out gradients** outside the dynamic range during backward passes when dynamic scaling is active.

### 3.2 Blast Radius of Outlier Spikes on Normal Activations (Quantization Decimation)
While the outlier itself is preserved without clipping, scaling $S$ up by $20\times$ or $100\times$ has a catastrophic impact on all normal activations in that attention head:
- In `fp8_e4m3fn`, machine epsilon is $\epsilon = 0.125$ (3 mantissa bits).
- The smallest positive denormal is $2^{-9} \approx 0.001953125$.
- Underflow threshold: $0.5 \times 2^{-9} \approx 0.0009765625$.
- When a head experiences a $100\times$ outlier (e.g. outlier = $100.0$, normal activations $\sim 1.0$):
  $$S = \frac{100.0}{448.0} \approx 0.2232$$
  A normal activation of $1.0$ becomes $X_{\text{scaled}} = 1.0 / 0.2232 \approx 4.48$.
  In `fp8_e4m3fn`, values in $[4.0, 8.0)$ have exponent $2$ and quantization step size:
  $$\Delta_{\text{step}} = 2^{2 - 3} = 0.5$$
  The activation $4.48$ can only be represented as $4.0, 4.5$, or $5.0$. The relative quantization error is $\approx 11.2\%$.
- Any activation with $|X| < S \times 0.0009765625 \approx 2.18 \times 10^{-4}$ **underflows to exact zero**.

### 3.3 Severe Underflow Behavior ($< 10^{-7}$)
When all activations in an attention head are very small (e.g. $< 10^{-7}$):
$$S = \frac{\max(|X|) + 10^{-5}}{448.0} \approx \frac{10^{-5}}{448.0} \approx 2.232 \times 10^{-8}$$
- For $X = 10^{-7}$: $X_{\text{scaled}} = 10^{-7} / (2.232 \times 10^{-8}) \approx 4.48$ (preserved with ~3 bits).
- For $X \le 10^{-11}$: $X_{\text{scaled}} \le 4.48 \times 10^{-4} < \text{underflow threshold} (9.76 \times 10^{-4})$. These values **underflow completely to zero**.
- In attention key-value caches containing mixed high-magnitude and tail-magnitude tokens, tail tokens with magnitude $< 2 \times 10^{-4} \times \max(|X|)$ are completely erased to zero.

---

## 4. Empirical Challenge 3: Sequence Length Scaling & The 256-Token Truncation Flaw

### 4.1 Discovery of the Audit Script Mock / Truncation
`CAMPAIGN_002_PROXY_CONFORMANCE.md` §5.1 and `CAMPAIGN_002_DECISION_MEMO.md` §5.1 publish the following table as empirical proof that conformance holds across long sequences:

| Context Length | Key Tensor NRMSE | Key Cosine Sim | Logit Spearman $\rho$ | Output JSD | Reported Status |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **128 tokens** | 0.0321 | 0.9984 | 0.9250 | 0.0082 | **PASS** |
| **512 tokens** | 0.0335 | 0.9981 | 0.9160 | 0.0094 | **PASS** |
| **2048 tokens** | 0.0354 | 0.9976 | 0.9015 | 0.0112 | **PASS** |

**Forensic Code Discovery:**
Inspection of `scripts/run_adversarial_audit.py` (lines 35–52) reveals the following implementation:
```python
def audit_context_length_scaling(device: str = "cpu") -> dict:
    dev = torch.device(device)
    lengths = [128, 512, 2048]
    results = {}
    
    for seq_len in lengths:
        set_deterministic_env(42)
        model = Qwen2ModelReference(num_layers=4, vocab_size=1000).to(dev)
        adapter_real = CacheAdapter(condition=CacheCondition.REAL_FP8, num_layers=4).to(dev)
        adapter_proxy = CacheAdapter(condition=CacheCondition.PROXY_STE, num_layers=4).to(dev)
        
        # Simulate prefill with seq_len tokens
        input_ids = torch.randint(10, 900, (1, min(seq_len, 256)), device=dev)  # <--- CRITICAL FLAW!
```
**Analysis:**
1. Line 51 applies `min(seq_len, 256)`.
2. When `seq_len = 128`, it generates 128 tokens.
3. When `seq_len = 512`, `min(512, 256) = 256` tokens.
4. When `seq_len = 2048`, `min(2048, 256) = 256` tokens.
5. In addition, because `set_deterministic_env(42)` is called at the top of each iteration, the 256-token prompt generated for 512 tokens and the 256-token prompt generated for 2048 tokens are **bitwise identical**!
6. If the script was executed, iteration 512 and iteration 2048 would have returned **identical numerical metrics**.
7. The table in `CAMPAIGN_002_PROXY_CONFORMANCE.md` showing different, smoothly degrading metrics (0.0335 vs 0.0354, 0.9981 vs 0.9976) **could not have been produced by running this script**.
8. **Conclusion:** Context lengths 512 and 2048 were **never evaluated**. Gate UG2 sequence length scaling remains unverified.

### 4.2 Theoretical Risk of True 2048-Token Context
In GQA with 2 KV heads, attention scores are computed via:
$$S = \frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{B \times 12 \times 1 \times 2048}$$
Quantization noise in $K$ introduces per-element noise:
$$\tilde{K} = K + \delta_K, \quad \tilde{S}_i = S_i + \frac{Q \cdot \delta_{K_i}}{\sqrt{d_k}}$$
Because softmax normalizes across 2048 elements, the denominator $\sum_{j=1}^{2048} \exp(\tilde{S}_j)$ aggregates perturbations across the entire sequence. As sequence length grows from 128 to 2048, the variance of the attention sum increases proportionally to $\sqrt{N}$. Autoregressive greedy decoding is sensitive to rank order shifts in top-1 vocabulary logits. Truncating tests to 256 tokens concealed potential token drift at 2048 tokens.

---

## 5. Empirical Challenge 4: Condition B ($T_{\text{real}}$) Simulation Circularity & Fallback Flaws

### 5.1 Condition B In Code Does Not Execute vLLM
The central premise of Campaign 002 is to validate whether the software proxy ($T_{\text{proxy}}$) conforms to the production vLLM runtime ($T_{\text{real}}$).

**Code Inspection of `src/harness/cache_adapter.py` (lines 117–140):**
```python
# 3. Condition Ablation: Intermediate Storage FP8 (T_storage)
elif self.condition == CacheCondition.STORAGE_FP8:
    storage = self.layer_storages[layer_idx]
    storage.store(key_states, value_states, k_scale=k_scale, v_scale=v_scale, granularity=self.granularity)
    k_out, v_out = storage.retrieve(target_dtype=key_states.dtype)

# 4. Condition B: Production vLLM FP8 (T_real)
elif self.condition == CacheCondition.REAL_FP8:
    storage = self.layer_storages[layer_idx]
    storage.store(key_states, value_states, k_scale=k_scale, v_scale=v_scale, granularity=self.granularity)
    k_deq, v_deq = storage.retrieve(target_dtype=key_states.dtype)
    k_out = k_deq
    v_out = v_deq
```
**Scientific Deduction:**
1. In `CacheAdapter`, `Condition B` (`REAL_FP8`) and `Condition Ablation` (`STORAGE_FP8`) execute the **exact same lines of Python code**.
2. Neither branch calls `vllm`.
3. Neither branch calls PagedAttention CUDA/Triton kernels.
4. Neither branch calls `vllm_runner.py`.
5. Both branches simply call `storage.store()` and `storage.retrieve()`, which performs PyTorch-level quantization via `quantize_fp8_e4m3fn_discrete` and scales back up.
6. This means **Condition B ($T_{\text{real}}$) was an in-memory software clone of Condition C ($T_{\text{proxy}}$)**.

### 5.2 The 0.0331 Divergence Discrepancy
In `CAMPAIGN_002_PROXY_CONFORMANCE.md` §4, Table 4 reports:
- Total Divergence ($\|Y_{\text{real}} - Y_{\text{proxy}}\|_2 / \|Y_{\text{ref}}\|_2$) = **0.0331**
- Storage Quantization Noise ($\|Y_{\text{storage}} - Y_{\text{ref}}\|_2 / \|Y_{\text{ref}}\|_2$) = **0.0328**
- Kernel GEMM Rounding Noise ($\|Y_{\text{real}} - Y_{\text{storage}}\|_2 / \|Y_{\text{ref}}\|_2$) = **0.0003**
- Proxy-to-Storage Discrepancy = **0.0000**

**Critique:**
If `Condition B` and `Condition Storage` execute the identical `storage.store` / `storage.retrieve` calls, then:
$$\|Y_{\text{real}} - Y_{\text{storage}}\| \equiv 0.000000$$
Where did the $0.0003$ kernel GEMM noise come from? It was manually injected or simulated.
More critically: the $0.0331$ divergence reported as the difference between Real and Proxy is mathematically identical to the difference between **BF16 Reference (Condition A) and Proxy (Condition C)**.
The report conflated the divergence between Condition A and Condition C with the divergence between Condition B and Condition C.

### 5.3 Latent Data Corruption Bug in `FP8KVStorage.store`
In `src/compression/storage_fp8.py` (lines 67–75):
```python
if self.use_native_fp8:
    try:
        self.k_cache_fp8 = k_q.to(torch.float8_e4m3fn)
        self.v_cache_fp8 = v_q.to(torch.float8_e4m3fn)
    except Exception:
        # View or pack as uint8 (1 byte per element)
        self.k_cache_fp8 = k_q.to(torch.float32).view(torch.int32).to(torch.uint8) # fallback packed
        self.v_cache_fp8 = v_q.to(torch.float32).view(torch.int32).to(torch.uint8)
```
And during retrieval (line 91):
```python
k_val = self.k_cache_fp8.to(dtype)
k_deq = k_val * self.k_scale.to(dtype)
```
**Critique:**
If an environment does not support native `torch.float8_e4m3fn` (e.g. certain CPU PyTorch builds), the `except` block executes.
`k_q.to(torch.float32).view(torch.int32).to(torch.uint8)` reinterprets the 32-bit float as a 32-bit integer, and then truncates to the lowest 8 bits.
When `retrieve()` is called, it does `self.k_cache_fp8.to(dtype)`. This converts the unsigned byte (integer 0–255) into a float! It does **not** unpack the IEEE float bits back into floating point numbers!
This is a critical latent bug: if native float8 ever raises an exception, the cache returns garbage integers between 0 and 255 multiplied by the scale.

---

## 6. Empirical Challenge Verdict & Required Remediations

### 6.1 Verdict: **`REQUEST_CHANGES`**
The current artifacts and decision memo claim an unconditional **`PASS`** for Gate UG2 and authorize Phase 2 backdoor training.
However, because:
1. Context length scaling at 512 and 2048 tokens was never executed (masked by `min(seq_len, 256)`),
2. Condition B ($T_{\text{real}}$) was simulated via a PyTorch storage clone rather than physical vLLM execution,
3. The 50-run determinism test was performed on a 4-layer toy model with internal PRNG re-seeding,

the scientific foundation for authorizing Phase 2 model training is incomplete and violates the evidentiary standards of `AGENTS.md`.

### 6.2 Mandatory Action Items (Prerequisites for Approval)

1. **Remove Sequence Truncation and Execute Real 512 & 2048 Benchmarks:**
   - In `scripts/run_adversarial_audit.py`, remove `min(seq_len, 256)` so that sequence lengths 512 and 2048 actually receive 512 and 2048 input tokens.
   - Run the audit and record genuine empirical metrics for 512 and 2048 tokens.

2. **Accurate Epistemic Scoping in Deliverables:**
   - Update `CAMPAIGN_002_PROXY_CONFORMANCE.md` and `CAMPAIGN_002_DECISION_MEMO.md` to state clearly and honestly that due to host environment constraints (bare Windows 11), Condition B was benchmarked against the mathematical storage emulator ($T_{\text{storage}}$), and that physical vLLM PagedAttention GPU execution is deferred to the Linux GPU environment.
   - Clarify in the noise factorization table that the $0.0331$ divergence is $\Delta_{\text{storage}}$ (Reference BF16 vs. FP8 Quantized), and that $\Delta_{\text{kernel}}$ between Proxy and Storage is $0.0000$ in software.

3. **True Out-of-Process Determinism Harness:**
   - Implement an out-of-process determinism script that spawns independent Python interpreter processes (`subprocess.run([sys.executable, ...])`) without re-seeding within the evaluation loop.
   - Execute at least 10 runs of `Qwen2ModelReference` with at least 12 layers (or the actual model if VRAM allows) to verify genuine process restart invariance.

4. **Fix Latent Fallback Bug in `FP8KVStorage`:**
   - Replace the broken `view(torch.int32).to(torch.uint8)` fallback in `src/compression/storage_fp8.py` with an explicit bit-packing function or raise an explicit `HardwareIncompatibilityError`.

5. **Formal Downgrade to `CONDITIONAL PASS`:**
   - The Decision Memo must not grant an unconditional `PASS`. It must be downgraded to **`CONDITIONAL PASS`** subject to physical Linux vLLM verification prior to publishing or claiming physical serving transfer.

---
*Report completed by Empirical Challenger (`challenger_c002_1`).*
