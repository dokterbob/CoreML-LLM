# CLAUDE.md — working method for the `pplx-embed` fork

This is a fork of `john-rocky/CoreML-LLM` (`origin`=`dokterbob/CoreML-LLM`,
`upstream`=`john-rocky`). The `pplx-embed` branch adds a bidirectional Qwen3 encoder path to put
Perplexity's pplx-embed models on the ANE. See [`PPLX_EMBED.md`](PPLX_EMBED.md) and
[`ROADMAP.md`](ROADMAP.md).

## Where things live

- **Working code:** this repo. New code under `conversion/` (flat scripts + `models/` package),
  matching the fork's existing convention.
- **Fidelity oracle:** `conversion/pplx_embed_reference.py` (HF fp32 → pool → `st_quantize`).
- **Upstream lessons:** broadly-useful lessons also go to CoreML-LLM's own `docs/` in
  **separate commits**.

## Rules

- **Match `st_quantize.py` exactly** — `torch.round` (half-to-even), qmin=−128. Not the paper's
  half-up floor.
- **Fixed-shape buckets only** — never EnumeratedShapes/RangeDim (CPU fallback). Start at L=4096;
  adding a bucket must be a one-line change. Pad to the next bucket with pad-token id + mask.
- **Fidelity gates:** fp16 ≥ 0.997, weight-quant ≥ 0.990 cosine vs fp32; exclude zero-norm rows.
- **Idempotency:** model is in the HF cache (don't re-download); skip-if-exists for `.mlpackage`;
  compile once via `xcrun coremlcompiler`.
- **macOS26 native int8 output isn't readable from the Python bridge** → use the Swift fidelity
  harness for those models.
- **Tooling:** `uv` (`uv sync`, `uv run …`). Research via Perplexity; library docs via Context7.

## Env

- macOS Tahoe, target `ct.target.macOS26`. Python 3.12 via uv. torch 2.11 / transformers 5.5 /
  coremltools 9 (coremltools warns torch 2.11 is untested — kept to match the fork's proven pins;
  revisit only if A1 tracing breaks).
