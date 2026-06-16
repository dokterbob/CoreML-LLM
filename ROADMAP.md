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
  latency 512→**101 ms**, 1024→**259 ms**, 4096→**4272 ms** (73-tok input, cpuAndNE), so the
  bucket strategy is a **42× win** for short inputs. ANE beats GPU 2.3× (1024: 258 ms vs
  601 ms cpuAndGPU; `.all` picks ANE). EnumeratedShapes verdict: the encoder bakes L into
  RoPE/reshapes so flexible shapes need a rewrite — decision already backed by two siblings +
  this bucket data; a flexible-shape spike is optional.
- [ ] **A3 — Context variant (late chunking).** `pool_matrix [32,L]` in → `chunk_embeddings
  [32,1024]` int8 out; per-chunk cosine ≥ 0.997 (exclude zero rows); ANE residency recorded.
- [ ] **A4 — Weight-only INT8 quant.** `linear_quantize_weights` via the quant abstraction
  (INT4 wired but off). Cosine vs fp32 ≥ 0.990.
- [ ] **A5 — INT8 on ANE.** ANE fraction + latency vs fp16 baseline.
- [ ] **A6 — Swift SPM API + parity.** `PplxEmbed`: `[String] → int8/binary/ubinary` (plain +
  per-chunk context); bucket select + pad/mask; matches Python reference within tolerance.

## Section B — Extensions

- [ ] **B1 — INT4 weight quant.** Flip `palettize_weights` (group_size=32); fidelity + ANE + latency.
- [ ] **B2 — Bucket expansion.** `{256,512,1024,2048,4096}`, skip-if-exists; per-bucket table.
- [ ] **B3 — mMARCO calibration + multilingual retrieval eval.** nDCG@10 across languages,
  fp32/fp16/INT8/INT4, plain + context.
- [ ] **B4 — (experiment) true W8A8 / A8.** Asymmetric activation quant; measure the ~cos 0.57 wall.
