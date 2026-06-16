# pplx-embed → CoreML/ANE (fork of CoreML-LLM)

This branch (`pplx-embed`) adds a **bidirectional Qwen3 encoder** path to convert Perplexity's
embedding models to CoreML and run them on the **Apple Neural Engine** (macOS Tahoe / `macOS26`):

- `perplexity-ai/pplx-embed-v1-0.6b` — **plain** sentence embeddings (mean-pool → 1024-d int8).
- `perplexity-ai/pplx-embed-context-v1-0.6b` — **late chunking** (per-chunk embeddings via a
  `pool_matrix` matmul).

Built **on top of** CoreML-LLM's ANE-first toolkit (Conv2d-1×1 projections, RoPE tables,
MLComputePlan residency auditor, weight-quant configs, Swift runtime). We prefer the fork's
native solutions over importing workarounds.

## Layout

| what | where |
|---|---|
| Working code | this repo (fork), branch `pplx-embed` |
| Model registry | `conversion/config.py` → `pplx-embed`, `pplx-embed-context` |
| Golden fp32 reference (oracle) | `conversion/pplx_embed_reference.py` |
| Encoder (A1) | `conversion/models/qwen3_encoder.py` *(to build)* |
| Bundle build (A1) | `conversion/build_pplx_embed_bundle.py` *(to build)* |
| ANE residency auditor | `conversion/audit_ane_residency.py` |
| Deps (uv) | `pyproject.toml` (mirrors `conversion/requirements.txt`) |

## Quickstart

```bash
uv sync
# golden reference (matches the model's own st_quantize.py exactly)
uv run python -c "import sys; sys.path.insert(0,'conversion'); \
  from pplx_embed_reference import Reference; \
  print(Reference('perplexity-ai/pplx-embed-v1-0.6b').embed(['hello world']).shape)"
# ANE residency of a compiled model
xcrun coremlcompiler compile model.mlpackage /tmp/out && \
  uv run python conversion/audit_ane_residency.py /tmp/out/model.mlmodelc
```

## Key facts

- **Fixed-shape buckets**, one `.mlpackage` per bucket — never EnumeratedShapes/RangeDim (they
  force CPU fallback). Start at `L=4096`.
- **Quantizers match `st_quantize.py` exactly**: int8 = `clamp(round(tanh(x)·127), −128, 127)`
  with **`torch.round`** (half-to-even), not the paper's half-up floor. Verified bit-exact.
- **Fidelity gates:** fp16 ≥ 0.997, weight-quant ≥ 0.990 cosine vs fp32 (exclude zero-norm rows).

See [`ROADMAP.md`](ROADMAP.md) for status and [`CLAUDE.md`](CLAUDE.md) for the working method.
