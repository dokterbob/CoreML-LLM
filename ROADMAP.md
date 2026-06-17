# Roadmap — pplx-embed on CoreML/ANE

Status of the `pplx-embed` fork work. See [`PPLX_EMBED.md`](PPLX_EMBED.md) for the overview.

## Section A — Foundation

- [x] **A0 — Fork & scaffold.**
  - Fork `john-rocky/CoreML-LLM` → `dokterbob/CoreML-LLM` (public); `origin`=fork,
    `upstream`=john-rocky; branch `pplx-embed`.
  - `uv` `pyproject.toml` mirroring `conversion/requirements.txt` pins; `uv sync` resolves;
    HF model loads with `trust_remote_code` under torch 2.11 / transformers 5.5 / coremltools 9.
  - Registry entries `pplx-embed` + `pplx-embed-context` (`conversion/config.py`).
  - Golden fp32 reference `conversion/pplx_embed_reference.py` — quantizers **bit-exact** vs
    `st_quantize.py`; `embed(["hello world"]) → (1,1024) int8`; late chunking matches the
    official `model.encode()` (cosine 1.0, ±1 int8 from matmul pooling).
  - `audit_ane_residency.py` verified on a real pplx-embed model (**99.4% ANE** on plain L512 fp16).
- [x] **A1 — Qwen3 bidirectional encoder, fp16, L=4096, PLAIN.** `models/qwen3_encoder.py` +
  `build_pplx_embed_bundle.py` + `test_pplx_embed_parity.py`.
  - PyTorch parity (L=64): pooled cos min 0.99980, int8 min 0.99967, 0 NaN.
  - fp16 fix: deep `down_proj` accumulation overflows fp16 (~layer 19);
    `apply_fp16_residual_rescale` (K=16) — exact for pre-norm. Verified to plateau
    peak |h| ~12–14k (4.6–6.3× headroom) up to 1015 real tokens.
  - CoreML L=4096 (macOS26, fp16): int8 cos min **0.99912** / mean 0.99961 (gate 0.997);
    build 118 s; latency ~4.3 s warm (full 4096-length forward, token-count-independent).
  - Native int8-output artifact builds (`dtype=INT8 [1,1024]`).
  - Swift bench (`pplx-embed-bench`) reads the native int8 output (Python bridge can't on
    macOS26): int8 cos min **0.99912** / mean 0.99967 (PASS); latency median **4324 ms** at
    L=4096 cpuAndNE — matches Python, i.e. real compute, not bridge overhead → see A2.
- [x] **A2 — fp16 ANE residency.** L=4096 plain (fp16 + int8): **99.80% ANE** / 0.20% CPU
  (4 mask-glue ops). The 4.3 s latency is not a fallback — it is inherent O(L²) attention:
  latency 512→**101 ms**, 1024→**259 ms**, 2048→**1372 ms**, 4096→**4340 ms** (73-tok input,
  cpuAndNE) — blow-up concentrated at the 1024→2048 knee (5.3× per 2× L, super-quadratic; ANE
  memory/tiling), so the bucket strategy is a **43× win** for short inputs. ANE beats GPU 2.3× (1024: 258 ms vs
  601 ms cpuAndGPU; `.all` picks ANE). EnumeratedShapes verdict: the encoder bakes L into
  RoPE/reshapes so flexible shapes need a rewrite — decision already backed by two siblings +
  this bucket data; a flexible-shape spike is optional.
- [x] **A3 — Context variant (late chunking).** `PplxEmbedContextModel`: 3rd input
  `pool_matrix [32,L]` → per-chunk pooling as one matmul → `chunk_embeddings [32,1024]` int8.
  CoreML (L=512, K=8): realistic (sentence) chunks **mean 0.99923 / min 0.99785 (PASS)**;
  **99.80% ANE** (matmul stays on ANE). Native int8 output `[32,1024]` converts.
  Note: degenerate 1–3 token chunks are fp16-encoder-limited (~0.996); this drove the K
  retune — **K=8 is now the default** (was 16): better on short chunks (context 0.9911→0.9987)
  and on plain (0.99967→0.99973), overflow-validated (peak ~37k @455 tok). K=4 overflows.
- [x] **A4/A5 — Weight quant: investigated, rejected.** int8 `linear_quantize_weights` is
  intrinsically broken on this encoder (cos ~0.42, min 0.006 — independent of the K-rescale and of
  granularity; per_block even fails ANE compile). int4 `palettize_weights` is the only survivor at
  **0.905** — still below the 0.990 gate. And it does not matter: weight quant buys only **4–8%**
  latency (512: 102→94 ms; 4096: 4421→4236 ms) — the model is activation/compute-bound, not
  weight-bandwidth-bound. **Decision: ship fp16 + buckets.** Quant flags stay wired (storage-only).
- [x] **A6 — Swift SPM API + parity.** `Sources/CoreMLLLM/PplxEmbed.swift` + `pplx-embed-demo`.
  `[String] → int8/binary/ubinary` (plain) and `[[String]] → per-chunk` (context); tokenize
  (swift-transformers) → smallest-fitting bucket → pad/mask → CoreML → native int8; binary = sign,
  ubinary = packbits (MSB-first); context builds the pool_matrix in Swift. Verified: builds clean,
  demos run, Swift int8 vs Python reference cosine **0.99976/0.99981/0.99827** (PASS).
  - **Dynamic routing (full size range).** `embed()` routes `n > largest bucket` → a flexible
    RangeDim model on GPU (non-padded, up to 8192) built via `--dynamic-upper`; everything else →
    smallest ANE bucket. Encoder refactored to derive seq-len (and batch) from the input. Verified:
    2 tok 0.99976 (ANE), 1141 tok 0.99925 (ANE L2048), **3361 tok 0.99956 (dynamic GPU)**.

## Section B — Extensions

- [x] **B1 — INT4 weight quant: measured (0.905, below gate).** `palettize_weights` group_size=32
  → cos 0.905 (< 0.990); ~4–8% latency. Folded into the A4/A5 weight-quant verdict above.
- [x] **B2 — Bucket expansion.** Built {512,1024,2048,4096}; per-bucket latency (cpuAndNE,
  73-tok): 101/259/1372/4340 ms — O(L²), knee at L=1024→2048 (5.3× super-quadratic). Adding a
  bucket is one CLI flag (`--max-seq-len`), skip-if-exists.
- [ ] **B3 — mMARCO calibration + multilingual retrieval eval.** nDCG@10 across languages,
  fp32/fp16/INT8/INT4, plain + context. *(Deferred — the one remaining open item.)*
- [x] **B4 — W8A8 / A8: not viable.** All variants collapse to **cos ≈ 0** (worse than the ~0.57
  wall and than weight-only int8); asym/sym and rescale-K make no difference. ANE residency fine
  (94%) — purely numerical. Recovery needs full QAT (rotation needs online Hadamards the ANE can't
  fold); and it's moot — int8 activations are only ~9% faster (not bandwidth-bound). See
  `docs/PPLX_EMBED_W8A8.md`.

## Post-plan findings (shape strategy + throughput)

- [x] **Shape strategy verdict.** Flexible shapes are ~10× slower than fixed buckets:
  EnumeratedShapes forces **CPU fallback** (measured: 969 ms @512 vs 101 ms ANE); RangeDim is slow
  even warm (per-shape JIT). **Fixed ANE buckets are the only fast path** and beat the GPU at every
  size (incl. past the knee). The dynamic RangeDim GPU model is purely the **>max-bucket catch-all**.
- [x] **Throughput / batching.** Batching gives **no MLX-style gains** on CoreML: the ANE is
  batch-1 by design (batching *hurts*, 0.69×, 100% on-ANE — not a fallback); CoreML GPU batches but
  saturates at L≥256 (MLX's 8× fills an under-utilized GPU, N/A here); CPU/BLAS best at 1.6× off a
  slow base. **ANE batch-1 at the smallest bucket (~72 docs/s @L128) is the throughput winner.**
  See `docs/PPLX_EMBED_BATCHING.md`.
