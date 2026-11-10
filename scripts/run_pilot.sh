#!/usr/bin/env bash
# Milestone 2 pilot, end to end. Every number in the report comes from these outputs.
# Prerequisites: postings.csv downloaded from Kaggle (see README), vLLM servers running.
set -euo pipefail
CSV=${CSV:?set CSV=/path/to/postings.csv}
MODELS=${MODELS:-"gpt-oss-20b qwen3-30b-a3b"}   # add gpt-oss-120b if you have an 80 GB GPU
TRIALS=${TRIALS:-1}

echo "== Pilot A: dataset audit and sandbox =="
python -m jobharness.data --csv "$CSV" --n 8000 --out data

echo "== Task suite =="
python -m jobharness.tasks --data data --per-family 48 --dev-frac 0.25

echo "== Pilot B: non-LLM baselines =="
python -m jobharness.baselines --tasks data/tasks_dev.jsonl --out runs/baselines.jsonl

echo "== Pilot C: models x harnesses on the dev set =="
python -m jobharness.run --tasks data/tasks_dev.jsonl --model $MODELS \
  --harness configs/harness/h0.yaml configs/harness/h0_guides.yaml configs/harness/h0_sensors.yaml \
  --trials "$TRIALS" --concurrency 8 --out runs/pilot.jsonl

echo "== Analysis =="
python -m jobharness.analyze runs/pilot.jsonl runs/baselines.jsonl --compare h0 h0_sensors --out results

echo "== Pilot D: annotation sheets (fill in by hand, then run annotate kappa) =="
python -m jobharness.annotate export-fit --tasks data/tasks_dev.jsonl --n 50
python -m jobharness.annotate export-fab --tasks data/tasks_dev.jsonl --runs runs/pilot.jsonl --n 50
echo "Done. Tables are in results/tables, the figure in results/figures,"
echo "and results/error_sample.csv is ready for error labeling."
