#!/usr/bin/env bash
# Start one vLLM server per model (run each in its own terminal / Slurm job).
# Flags follow the vLLM docs at the time of writing; check them against your installed
# vLLM version (`vllm serve --help`), especially the tool-call parser names.
#   gpt-oss models:  built-in "openai" tool parser (Harmony format)
#   Qwen3 instruct:  "hermes" tool parser
# On 40 GB GPUs: set TP=2 for gpt-oss-120b, and use the -FP8 Qwen checkpoint
# (or TP=2) for Qwen3-30B-A3B.
set -euo pipefail
MODEL=${1:?usage: scripts/serve_vllm.sh gpt-oss-120b|gpt-oss-20b|qwen3-30b-a3b}
TP=${TP:-1}
CTX=${CTX:-32768}
case "$MODEL" in
  gpt-oss-120b)
    vllm serve openai/gpt-oss-120b --port 8000 --tensor-parallel-size "$TP" \
      --max-model-len "$CTX" --enable-auto-tool-choice --tool-call-parser openai ;;
  gpt-oss-20b)
    vllm serve openai/gpt-oss-20b --port 8001 --tensor-parallel-size "$TP" \
      --max-model-len "$CTX" --enable-auto-tool-choice --tool-call-parser openai ;;
  qwen3-30b-a3b)
    vllm serve "${QWEN_CKPT:-Qwen/Qwen3-30B-A3B-Instruct-2507}" --port 8002 \
      --served-model-name Qwen/Qwen3-30B-A3B-Instruct-2507 --tensor-parallel-size "$TP" \
      --max-model-len "$CTX" --enable-auto-tool-choice --tool-call-parser hermes ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
