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
- [ ] **A1 — Qwen3 bidirectional encoder, fp16, L=4096, PLAIN.** `models/qwen3_encoder.py` +
  `build_pplx_embed_bundle.py`. Gate: cosine vs fp32 ≥ 0.997 (Swift harness, int8 output).
- [ ] **A2 — fp16 ANE residency + EnumeratedShapes verdict.** Audit L=4096 plain; record
  CPU/GPU/ANE fractions; one fixed-vs-Enumerated comparison logged.
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
