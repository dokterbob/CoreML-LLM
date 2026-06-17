# pplx-embed — Perplexity embedding models on the ANE

Adds a **bidirectional Qwen3 encoder** path that converts Perplexity's pplx-embed models to CoreML
and runs them on the Apple Neural Engine (macOS Tahoe / `macOS26`):

- `perplexity-ai/pplx-embed-v1-0.6b` — **plain** sentence embeddings (mean-pool → 1024-d int8).
- `perplexity-ai/pplx-embed-context-v1-0.6b` — **late chunking** (per-chunk embeddings via a
  `pool_matrix` matmul; one encoder pass over the whole window).

The encoder is a 28-layer bidirectional Qwen3-0.6B (GQA 16/8, head_dim 128, SwiGLU, QK-norm,
RoPE θ=1e6) built on the existing ANE primitives (Conv2d-1×1 projections, `ANERMSNorm`,
`repeat_kv_ane`, `stable_attention`). Output matches the model's own `st_quantize.py` exactly:
int8 = `clamp(round(tanh(x)·127), −128, 127)` (`torch.round`, half-to-even), plus `binary`
(sign) and `ubinary` (packbits).

## Files

| what | where |
|---|---|
| Model registry | `conversion/config.py` → `pplx-embed`, `pplx-embed-context` |
| Encoder | `conversion/models/qwen3_encoder.py` |
| Bundle builder | `conversion/build_pplx_embed_bundle.py` |
| Golden fp32 reference (oracle) | `conversion/pplx_embed_reference.py` |
| Parity test | `conversion/test_pplx_embed_parity.py` |
| Swift runtime | `Sources/CoreMLLLM/PplxEmbed.swift` (+ `pplx-embed-demo`, `pplx-embed-bench`) |

## Build

```bash
# A fixed-shape ANE bucket (the fast path), plain int8 output:
python conversion/build_pplx_embed_bundle.py --model pplx-embed --max-seq-len 512
# Context (late chunking) variant:
python conversion/build_pplx_embed_bundle.py --model pplx-embed-context --max-seq-len 512
# The flexible GPU catch-all for inputs larger than the biggest bucket (up to 8192):
python conversion/build_pplx_embed_bundle.py --model pplx-embed --dynamic-upper 8192
```

Verify fidelity against the fp32 reference (CPU, fast):

```bash
python conversion/test_pplx_embed_parity.py        # pooled ≥0.999, int8 ≥0.997
```

## Use (Swift)

```swift
let embedder = try await PplxEmbed.load(bundleDir: URL(fileURLWithPath: "output/pplx-embed"))
let vectors = try embedder.embed(["hello world", "bonjour le monde"])   // [[Int8]] (1024-d)
// also: embedBinary / embedUBinary; embedContext([[String]]) for late chunking
```

`embed()` tokenizes, selects the **smallest fixed bucket** that fits, pads/masks, and runs on the
ANE. Inputs larger than the biggest bucket are routed to the flexible RangeDim model on the GPU
(non-padded). Run the CLI demo with `swift run -c release pplx-embed-demo --bundle-dir output/pplx-embed --text "…"`.

## Design notes

- **Fixed-shape buckets, one `.mlpackage` per bucket.** Flexible shapes (EnumeratedShapes/RangeDim)
  force CPU fallback on the ANE and are ~10× slower; fixed buckets stay 99.8% on the ANE. Pad each
  input to the smallest fitting bucket. Latency is O(L²) with a sharp knee at L=1024→2048.
- **Flexible GPU model is the >max-bucket catch-all only.** Built with `--dynamic-upper N`
  (RangeDim 1..N), it runs on the GPU non-padded for unbounded length — correct (cos 0.999) but
  ~10× slower than a fixed bucket, so it's used only when no bucket fits.
- **fp16 residual rescale (K=8).** The 28-layer `down_proj` accumulation overflows fp16; scaling
  `embed_tokens`/`o_proj`/`down_proj` by 1/K is exact for a pre-norm net (scale-invariant norms)
  and keeps activations in range. K=8 is the fidelity/overflow sweet spot.
- **macOS26 native int8 output** is not readable from the Python CoreML bridge; read it in Swift
  (the `pplx-embed-bench` harness does). Fidelity is otherwise measured via a `pooled_fp16`-output
  variant in Python.
- **Throughput:** ANE batch-1 at the smallest bucket is both the lowest-latency and
  highest-throughput path; batching is not a useful lever on CoreML (see below).

See [`PPLX_EMBED_W8A8.md`](PPLX_EMBED_W8A8.md) (weight/activation quantization is not viable for
this model) and [`PPLX_EMBED_BATCHING.md`](PPLX_EMBED_BATCHING.md) (no MLX-style batching gains;
the ANE is batch-1 by design).
