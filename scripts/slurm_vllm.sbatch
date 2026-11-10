#!/bin/bash
# Example Slurm job for serving one model on a university GPU cluster.
# Adjust partition, account, GPU type and module names to your cluster's documentation.
#SBATCH --job-name=vllm
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/vllm-%j.out
set -euo pipefail
source ~/venvs/jobharness/bin/activate   # environment with vllm installed
echo "Serving on $(hostname); tunnel with: ssh -L 8001:$(hostname):8001 <login-node>"
bash scripts/serve_vllm.sh "${MODEL:-gpt-oss-20b}"
