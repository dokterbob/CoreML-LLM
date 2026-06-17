# CoreML batching throughput — pplx-embed encoder (Apple Silicon)

**Question.** Does CoreML batching (B>1) raise throughput for the pplx-embed
bidirectional Qwen3-0.6B encoder on Apple Silicon, the way MLX gives ~8× on
larger models? An earlier quick test (L=512, pooled_fp16, warm) showed FLAT
docs/sec across B — ANE ~9/s, GPU ~4.7/s. Is that real, and why?

**Verdict (bottom line).** The flat result is **real, not a measurement bug**, but
it is **device-specific**:

- **ANE (`CPU_AND_NE`) does NOT batch — it gets *worse* with B.** Per-doc latency
  is flat-to-rising; throughput *drops* to ~0.68–0.71× of B=1 at L=128. The ANE is
  a batch-1-oriented fixed-function accelerator: it serializes batch rows and adds
  per-row overhead. No 8×, ever. Confirmed.
- **GPU (`CPU_AND_GPU`) batches, but only modestly and only when one sequence
  under-fills the GPU (small L).** At L=128 it gains up to **1.44×** (B=16); at
  L=512 a single sequence already saturates the GPU, so batching is flat (~1.0×,
  even regressing to 0.92× at B=64). Nowhere near MLX's 8×.
- **CPU/BLAS (`CPU_ONLY`) batches the most — up to ~1.6× at L=128** (B=16), ~1.1×
  at L=512. This is the Accelerate/BLAS GEMM batching control behaving as expected,
  and it is the largest batch win of the three backends — but it is off a slow
  baseline, so in *absolute* docs/sec it never beats batch-1 ANE.

**So: can CoreML batch like MLX's 8×? No — partially at best (~1.4–1.6× on
GPU/CPU at short sequences, nothing on ANE).** The reason is architectural, not a
bug: the fast path (ANE) is the one backend that fundamentally can't batch, and the
backends that *can* batch (GPU/CPU) are the slow paths whose batch gains are small
and saturate by L≈512. MLX's 8× comes from a Metal-kernel GPU engine that keeps the
GPU's ALUs busy by parallelizing across the batch; CoreML's ANE path can't do that,
and CoreML's GPU path only helps while the GPU is under-utilized.

---

## Setup

- Machine: Apple Silicon, macOS 26.5.1 (arm64). coremltools 9.0.
- Model: `PplxEmbedModel(cfg, output_mode="pooled_fp16")`, fp16 residual rescale K=8,
  traced+converted at shape **(B, L)** per cell, `minimum_deployment_target=macOS26`,
  converted with `compute_units=ALL`, then **loaded** under each compute-unit setting
  (`CPU_AND_NE`, `CPU_AND_GPU`, `CPU_ONLY`) so each backend is forced.
- Timing: each (L, B, unit) shape **warmed** (3 predicts) then **median of 8** timed
  predicts. Inputs are fp16 all-ones mask, random int32 ids.
- `per-doc latency = batch_latency / B`, `docs/sec = B / batch_latency`.
- Script: `conversion/experiment_batching.py` (parametrized, reproducible).
- A second independent `--quick` run (runs=6) reproduced the ANE numbers within noise
  (B=1 L=128: 72 vs 71 docs/s; B=1 L=512: 10.4 vs 9.9 docs/s), confirming stability.

## docs/sec (rows = B, cols = compute unit)

### L=128
| B  | CPU_AND_NE | CPU_AND_GPU | CPU_ONLY |
|----|-----------:|------------:|---------:|
| 1  |      71.18 |       18.37 |    20.87 |
| 4  |      49.06 |       20.78 |    28.30 |
| 16 |      50.75 |       26.41 |    33.37 |
| 64 |      48.70 |       23.87 |    32.37 |

### L=512
| B  | CPU_AND_NE | CPU_AND_GPU | CPU_ONLY |
|----|-----------:|------------:|---------:|
| 1  |       9.94 |        4.58 |     6.29 |
| 4  |       9.06 |        4.95 |     6.91 |
| 16 |       8.98 |        4.66 |     6.96 |
| 64 |       7.32 |        4.20 |     6.86 |

## per-doc latency (ms) — the key view (flat = no batch gain)

### L=128
| B  | CPU_AND_NE | CPU_AND_GPU | CPU_ONLY |
|----|-----------:|------------:|---------:|
| 1  |     14.048 |      54.425 |   47.918 |
| 4  |     20.383 |      48.116 |   35.336 |
| 16 |     19.706 |      37.863 |   29.963 |
| 64 |     20.532 |      41.886 |   30.892 |

### L=512
| B  | CPU_AND_NE | CPU_AND_GPU | CPU_ONLY |
|----|-----------:|------------:|---------:|
| 1  |    100.583 |     218.390 |  159.103 |
| 4  |    110.398 |     201.895 |  144.813 |
| 16 |    111.340 |     214.736 |  143.760 |
| 64 |    136.588 |     238.156 |  145.773 |

## batch speedup = docs/s(B) / docs/s(B=1)

### L=128
| B  | CPU_AND_NE | CPU_AND_GPU | CPU_ONLY |
|----|-----------:|------------:|---------:|
| 4  |      0.69× |       1.13× |    1.36× |
| 16 |      0.71× |     **1.44×** |  **1.60×** |
| 64 |      0.68× |       1.30× |    1.55× |

### L=512
| B  | CPU_AND_NE | CPU_AND_GPU | CPU_ONLY |
|----|-----------:|------------:|---------:|
| 4  |      0.91× |       1.08× |    1.10× |
| 16 |      0.90× |       1.02× |    1.11× |
| 64 |      0.74× |       0.92× |    1.09× |

---

## Reading the data

1. **ANE: per-doc latency is flat-to-rising and throughput *falls* below 1.0×.**
   B=1 L=128 is the single fastest cell at **71 docs/s** (14.0 ms). Going to B≥4
   *raises* per-doc latency to ~20 ms (0.68–0.71×). The ANE runs the batch as B
   sequential single-row passes plus marshalling overhead — exactly hypothesis #1
   ("ANE is batch-1 by design"). This is the dominant fact and it holds at **both**
   L=128 and L=512, so it is **not** a small-L under-utilization artifact.

2. **GPU does parallelize the batch — but only while the GPU is under-filled.**
   At L=128 the single-sequence GPU pass under-utilizes the ALUs, so batching to
   B=16 cuts per-doc latency 54→38 ms (**1.44×**). At L=512 one 512-token bidirectional
   pass already fills the GPU, so batching is flat (1.0–1.08×) and even regresses at
   B=64 (0.92×, thermal/occupancy). This is hypothesis #3 resolved: batching helps on
   GPU *only* at small L, and the effect is bounded (~1.4×), nowhere near 8×.

3. **CPU/BLAS batches the most (the control did its job).** `CPU_ONLY` lowers the
   GEMMs onto Accelerate/BLAS, which amortizes per-call overhead across the batch:
   L=128 gains up to **1.60×** (B=16), L=512 up to ~1.11×. Largest *relative* gain of
   the three, but off the slowest baseline, so it never wins in absolute docs/sec.

4. **Why the earlier "flat" quick test looked flat.** It used **L=512** and reported
   ANE (~9/s) and GPU (~4.7/s). At L=512 *both* of those backends are genuinely flat
   (ANE structurally; GPU because 512 tokens already saturate it). The quick test
   wasn't buggy — it just happened to pick the one sequence length where even the
   batchable backend (GPU) has nothing left to give. Had it also probed **L=128 on
   GPU/CPU**, it would have seen the modest 1.4–1.6× gains. The blind spot was
   "only tested L=512, only looked at docs/sec deltas that are real-but-zero there."

## Why no MLX-style 8×

MLX gets ~8× from batching because its Metal GPU kernels run the whole batch as one
big-matrix workload that keeps the GPU's ALUs saturated; the per-token work is
memory-bandwidth/occupancy bound at B=1 and batching fills the machine. On CoreML:

- The **fast path is the ANE**, and the ANE is a fixed-function, batch-1 engine: it
  streams one (C,1,S) tile at a time and there is no batch axis to parallelize over,
  so B>1 is pure serialization — it can never give a batch speedup (it gives a small
  *slowdown*).
- The **CoreML GPU path *can* batch-parallelize** (hypothesis #2 is therefore *false*
  as stated — CoreML GPU is not incapable of batching), **but** for this 0.6B encoder
  a single sequence of L≥~256 already saturates the GPU, so the headroom MLX exploits
  is already gone by the time you batch. The gain you can still capture (small L) tops
  out around 1.4×.

Net: on this model, the throughput-optimal strategy on Apple Silicon is **batch-1 on
the ANE** (71 docs/s at L=128, 10 docs/s at L=512), and **batching is not a lever**
that approaches MLX's 8× — at most ~1.4–1.6× on GPU/CPU at short sequences.

## Device-placement audit (MLComputePlan)

The batched models were compiled (`MLModel.get_compiled_model_path()` → copied to a
stable `.mlmodelc`; `MLComputePlan.load_from_path` aborts on a raw `.mlpackage` in
coremltools 9.0) and every MLProgram op's `preferred_compute_device` was tallied.
This is the *static* compute plan (non-`const` ops only).

| model        | CPU_AND_NE         | CPU_AND_GPU            | CPU_ONLY    |
|--------------|--------------------|-----------------------|-------------|
| L=128, B=1   | **ANE 100%**, CPU 0% | CPU 96%, GPU 4%      | CPU 100%    |
| L=128, B=64  | **ANE 100%**, CPU 0% | CPU 84%, **GPU 16%** | CPU 100%    |
| L=512, B=1   | **ANE 100%**, CPU 0% | CPU 86%, GPU 14%     | CPU 100%    |

(1975 compute ops total per model.)

**This is the decisive control: the batched (B=64) model is 100% on the ANE — it does
NOT fall back to CPU.** So the flat-to-rising ANE per-doc latency is *not* a hidden
device-fallback or a measurement bug: the ANE genuinely accepts the batched graph and
runs it, but serializes the batch axis. Hypothesis #1 confirmed; the earlier "flat"
reading was a true hardware property, not a wrong-device artifact.

The `CPU_AND_GPU` plan is interesting: MLComputePlan's *static preference* keeps most
ops on CPU and only marks more ops GPU as B grows (4%→16% at L=128). Yet the *measured*
GPU latency improves with B at L=128 — i.e. at runtime the GPU does carry the batched
matmuls; the static plan understates GPU use. Either way, the GPU's realized batch gain
caps at ~1.4×.

## Sanity — batching is real (no broadcast bug)

Feeding **distinct** input rows (random ids per row) to the B=4 model yields **distinct
output rows** (pairwise max|diff| ≈ 0.34–0.49, not ~0). And batch **row 0 exactly
matches the standalone B=1 model** on the same input (max|diff| = 0.0000, MATCH). So
each batch element is encoded independently and correctly — the batch timings are
genuine per-document work, the flat throughput is real.

## Control — batch-N vs N × (B=1), total wall time (B=64, `conversion/experiments/control.log`)

Direct head-to-head: one batch-64 predict vs 64 sequential B=1 predicts, total wall time.

| unit \ L    | L=128 batch-64 / 64×(B=1) / speedup | L=512 batch-64 / 64×(B=1) / speedup |
|-------------|-------------------------------------|-------------------------------------|
| CPU_AND_NE  | 1313.7ms / 905.1ms / **0.69×**      | 8326.8ms / 6442.6ms / **0.77×**     |
| CPU_AND_GPU | 2764.4ms / 3530.0ms / **1.28×**     | 16315ms / 14310ms / **0.88×**       |
| CPU_ONLY    | 2021.5ms / 3027.9ms / **1.50×**     | 9325.8ms / 10233ms / **1.10×**      |

This is the cleanest statement of the result: on the **ANE you are better off looping
N single-document predicts** than issuing one batch-N call (0.69–0.77×, i.e. batching
*loses*). On **GPU and CPU, batching helps at short sequences** (1.28× / 1.50× at L=128)
but the GPU gain evaporates by L=512 (0.88×, GPU saturated) while CPU/BLAS keeps a small
edge (1.10×). In *absolute* docs/sec, batch-1 ANE is still the throughput winner
everywhere.
