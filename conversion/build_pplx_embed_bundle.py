#!/usr/bin/env python3
"""Build a CoreML bundle for pplx-embed (bidirectional Qwen3 encoder).

Stateless single-forward export — no KV cache, no causal mask. Fixed-length
(`--max-seq-len`, the bucket) input + pad mask. Variable length is handled by
padding to the nearest bucket at runtime (fixed shapes keep it on the ANE;
RangeDim/EnumeratedShapes force CPU fallback).

Output modes:
    pooled_fp16  masked-mean → (1, 1024) fp16   — readable from the Python bridge;
                 quantize to int8 downstream. Use for fidelity measurement.
    int8         masked-mean → tanh → int8       — the deliverable (native int8;
                 read via the Swift harness on macOS26).

Usage:
    python conversion/build_pplx_embed_bundle.py --model pplx-embed --max-seq-len 4096
    python conversion/build_pplx_embed_bundle.py --model pplx-embed --max-seq-len 128 \
        --output-mode pooled_fp16
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.qwen3_encoder import (  # noqa: E402
    Qwen3EncoderConfig,
    PplxEmbedModel,
    load_encoder_weights,
    apply_fp16_residual_rescale,
)

DEFAULT_RESCALE_K = 16.0


def _snapshot_dir(hf_repo: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(
        hf_repo,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.py", "1_Pooling/*"],
    )


def build_bundle(
    hf_repo: str,
    model_name: str,
    output_dir: str,
    max_seq_len: int = 4096,
    output_mode: str = "int8",
    rescale_k: float = DEFAULT_RESCALE_K,
    quantize: str | None = None,
    skip_if_exists: bool = True,
) -> str:
    import coremltools as ct

    os.makedirs(output_dir, exist_ok=True)
    pkg = os.path.join(output_dir, "encoder.mlpackage")
    if skip_if_exists and os.path.exists(pkg):
        print(f"  [skip] {pkg} exists")
        return pkg

    print(f"[1/4] Loading {hf_repo} (config + weights)")
    snap = hf_repo if os.path.isdir(hf_repo) else _snapshot_dir(hf_repo)
    cfg = Qwen3EncoderConfig.from_json(os.path.join(snap, "config.json"), max_seq_len=max_seq_len)
    model = PplxEmbedModel(cfg, output_mode=output_mode).eval()
    load_encoder_weights(model.encoder, snap)
    if rescale_k:
        print(f"[1.5/4] fp16 residual rescale 1/K (K={rescale_k})")
        apply_fp16_residual_rescale(model.encoder, rescale_k)

    print(f"[2/4] Tracing (L={max_seq_len}, mode={output_mode})")
    sample_ids = torch.zeros((1, max_seq_len), dtype=torch.int32)
    sample_mask = torch.ones((1, max_seq_len), dtype=torch.float16)
    with torch.no_grad():
        traced = torch.jit.trace(model, (sample_ids, sample_mask))

    out_dtype = np.int8 if output_mode == "int8" else np.float16
    print(f"[3/4] Converting to CoreML (fp16, macOS26, stateless; out={out_dtype.__name__})")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, max_seq_len), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, max_seq_len), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="embedding", dtype=out_dtype)],
        minimum_deployment_target=ct.target.macOS26,
        compute_units=ct.ComputeUnit.ALL,
    )

    if quantize == "int4":
        op = ct.optimize.coreml.OpPalettizerConfig(nbits=4, granularity="per_grouped_channel", group_size=32)
        mlmodel = ct.optimize.coreml.palettize_weights(
            mlmodel, ct.optimize.coreml.OptimizationConfig(global_config=op))
        print("  applied int4 palettization (group_size=32)")
    elif quantize == "int8":
        op = ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        mlmodel = ct.optimize.coreml.linear_quantize_weights(
            mlmodel, ct.optimize.coreml.OptimizationConfig(global_config=op))
        print("  applied int8 weight quantization")

    if os.path.exists(pkg):
        shutil.rmtree(pkg)
    mlmodel.save(pkg)
    size_mb = sum(os.path.getsize(os.path.join(dp, f))
                  for dp, _, fns in os.walk(pkg) for f in fns) / 1024 / 1024
    print(f"  saved {pkg} ({size_mb:.1f} MB)")

    _write_model_config(output_dir, model_name, hf_repo, cfg, max_seq_len,
                        output_mode, rescale_k, quantize)
    _copy_tokenizer(snap, output_dir)
    print(f"[4/4] bundle ready at {output_dir}")
    return pkg


def _write_model_config(output_dir, model_name, hf_repo, cfg, max_seq_len,
                        output_mode, rescale_k, quantize):
    out_shape = [1, 1024]
    out_dtype = "int8" if output_mode == "int8" else "fp16"
    cfgd = {
        "model_name": model_name,
        "architecture": "qwen3-encoder",
        "tokenizer_repo": hf_repo,
        "parts": {"encoder": "encoder.mlpackage"},
        "io_contract": {
            "inputs": {
                "input_ids": {"shape": [1, max_seq_len], "dtype": "int32"},
                "attention_mask": {"shape": [1, max_seq_len], "dtype": "fp16",
                                   "doc": "1.0 for valid tokens, 0.0 for pad"},
            },
            "outputs": {
                "embedding": {"shape": out_shape, "dtype": out_dtype,
                              "doc": "plain mean-pooled embedding; int8 = clamp(round(tanh(x)*127),-128,127)"},
            },
        },
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "intermediate_size": cfg.intermediate_size,
        "vocab_size": cfg.vocab_size,
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": cfg.rms_norm_eps,
        "max_seq_len": max_seq_len,
        "bucket": max_seq_len,
        "output_mode": output_mode,
        "fp16_residual_rescale_k": rescale_k,
        "pooling": "mean",
        "quantization_weights": quantize or "fp16",
        "matryoshka_dims": [1024, 512, 256, 128],
        "compute_units": "CPU_AND_NE",
    }
    path = os.path.join(output_dir, "model_config.json")
    with open(path, "w") as f:
        json.dump(cfgd, f, indent=2)
    print(f"  wrote {path}")


def _copy_tokenizer(snap, output_dir):
    dst = os.path.join(output_dir, "hf_model")
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(snap):
        if name.startswith("tokenizer") or name in (
            "config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
            "added_tokens.json",
        ):
            shutil.copy2(os.path.join(snap, name), os.path.join(dst, name))
    print(f"  copied tokenizer files → {dst}")


def main():
    from config import MODEL_REGISTRY

    ap = argparse.ArgumentParser(description="Build CoreML bundle for pplx-embed")
    ap.add_argument("--model", default="pplx-embed", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--output-mode", default="int8", choices=["int8", "pooled_fp16"])
    ap.add_argument("--rescale-k", type=float, default=DEFAULT_RESCALE_K)
    ap.add_argument("--quantize", default="none", choices=["none", "int8", "int4"])
    ap.add_argument("--hf-dir", default=None, help="Override HF dir (skip download)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-skip", action="store_true", help="Rebuild even if exists")
    args = ap.parse_args()

    reg = MODEL_REGISTRY[args.model]
    hf_repo = args.hf_dir or reg.hf_repo
    output = args.output or os.path.join(
        ROOT, "..", "output", args.model,
        f"L{args.max_seq_len}-{args.output_mode}" + (f"-{args.quantize}" if args.quantize != "none" else ""))
    quantize = None if args.quantize == "none" else args.quantize
    build_bundle(hf_repo, args.model, output, args.max_seq_len, args.output_mode,
                 args.rescale_k, quantize, skip_if_exists=not args.no_skip)


if __name__ == "__main__":
    main()
