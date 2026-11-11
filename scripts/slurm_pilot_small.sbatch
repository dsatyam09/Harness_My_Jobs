#!/bin/bash
#SBATCH --job-name=jh-pilot
#SBATCH --partition=h100-debug
#SBATCH --account=r01789
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:58:00
#SBATCH --output=logs/pilot-%j.out
# Serve three 1-3B models on one H100 (vLLM), then run the dev pilot. All real runs.
# Env: TRIALS (default 3), OUT (default runs/pilot3.jsonl)
set -uo pipefail
cd /N/u/sbdubey/Quartz/Harness_my_jobs
module load python/3.12.4
export C_INCLUDE_PATH=/N/soft/rhel8/python/gnu/3.12.4/include/python3.12:${C_INCLUDE_PATH:-}
export VLLM_USE_FLASHINFER_SAMPLER=0   # FlashInfer sampler needs nvcc (JIT); use the PyTorch sampler
export HF_HOME=/N/scratch/sbdubey/hf HF_HUB_OFFLINE=1
mkdir -p logs runs
serve() { # name port hf_id [mem_util] [max_len]
  /N/scratch/sbdubey/envs/vllm/bin/vllm serve "$3" --port "$2" --gpu-memory-utilization "${4:-0.25}" \
    --max-model-len "${5:-16384}" --enable-auto-tool-choice --tool-call-parser hermes \
    > logs/vllm-$1.log 2>&1 &
}
wait_port() { for i in $(seq 1 300); do curl -s localhost:$1/v1/models >/dev/null && { echo "port $1 up"; return 0; }; sleep 5; done; echo "port $1 FAILED"; return 1; }
# start servers one at a time: concurrent startup races on the GPU memory check
serve qwen2.5-3b 8010 Qwen/Qwen2.5-3B-Instruct;      wait_port 8010 || exit 1
serve qwen2.5-1.5b 8011 Qwen/Qwen2.5-1.5B-Instruct;  wait_port 8011 || exit 1
serve qwen3-1.7b 8012 Qwen/Qwen3-1.7B 0.28 12288;    wait_port 8012 || exit 1
source ../Harness_my_jobs_venv/bin/activate
M="qwen2.5-3b qwen2.5-1.5b qwen3-1.7b"
H="configs/harness/h0.yaml configs/harness/h0_guides.yaml configs/harness/h0_sensors.yaml configs/harness/h0_text.yaml configs/harness/full.yaml"
echo "== pilot"; python -m jobharness.run --data data --tasks data/tasks_dev.jsonl --models configs/models.yaml \
  --model $M --harness $H --trials ${TRIALS:-3} --concurrency 16 --out ${OUT:-runs/pilot3.jsonl}
echo "== done"
