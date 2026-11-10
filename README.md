# JobHarness

Does an agent harness tuned on a large open model transfer to small self-hosted models?
This repository contains the job-search sandbox, task suite, automatic graders, modular
harness, experiment runner, analysis scripts and CI gate for the JobHarness project
(Applied NLP project; PN Tejeshwar Sarma and Satyam Dubey).

All models are open-weight and self-hosted with vLLM. No commercial APIs are used.

## Layout

| Path | What it is |
|---|---|
| `jobharness/data.py` | Loads the LinkedIn postings CSV, runs the dataset audit (Pilot A), builds the sandbox |
| `jobharness/profiles.py` | Synthetic candidate profiles with fully known facts |
| `jobharness/tasks.py` | Five task families with hidden answers; dev/test split |
| `jobharness/graders.py` | Automatic graders (F1, exact match, nDCG@5, field accuracy, fabrication) |
| `jobharness/env.py` | Sandbox tools: BM25 search, job reading, evidence lookup. Never returns hidden fields |
| `jobharness/harness.py` | Modular tool-calling loop; every component is a YAML switch |
| `jobharness/run.py` | Runs model x harness x task x trial; resumable JSONL logs |
| `jobharness/baselines.py` | Non-LLM reference points |
| `jobharness/analyze.py` | Success rates, bootstrap CIs, McNemar, pass^k, transfer ratio, gap closure, LaTeX tables |
| `jobharness/annotate.py` | Blind annotation sheets and Cohen's kappa |
| `jobharness/gate.py` | CI regression gate |
| `jobharness/mcp_server.py` | The sandbox as an MCP server (for the copilot) |
| `configs/harness/*.yaml` | Harness profiles: `h0`, `h0_guides`, `h0_sensors`, `h0_text`, `full` |
| `configs/models.yaml` | vLLM endpoints and sampling settings |
| `tests/` | Unit and integration tests (synthetic fixture + mock models) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,mcp]"
pytest -q            # 16 tests; uses a synthetic fixture and mock models
```

Install vLLM on the GPU machine separately (`pip install vllm`), following the vLLM docs for
your CUDA version.

Download `postings.csv` from the Kaggle dataset *LinkedIn Job Postings (2023-2024)*
(arshkon/linkedin-job-postings). It is CC BY-SA; do not commit it to the repository.

## Milestone 2 pilot (about one day of compute, a few hours of your time)

```bash
# 1. Serve the models (separate terminals or Slurm jobs; see scripts/slurm_vllm.sbatch)
bash scripts/serve_vllm.sh gpt-oss-20b
bash scripts/serve_vllm.sh qwen3-30b-a3b
bash scripts/serve_vllm.sh gpt-oss-120b      # only if you have an 80 GB GPU (or TP=2)

# 2. Run everything
CSV=/path/to/postings.csv MODELS="gpt-oss-20b qwen3-30b-a3b" bash scripts/run_pilot.sh
```

This produces:

| Output | Pilot | Goes into the report as |
|---|---|---|
| `results/tables/audit.tex`, `data/audit_summary.json` | A: dataset audit | Table 2 |
| `results/tables/tasks.tex` | Task suite statistics | Table 3 |
| `results/tables/success.tex`, `scores.tex` | B + C: baselines and models | Tables 4-5 |
| `results/tables/process.tex`, `finish.tex` | C: how runs fail | Tables 6-7 |
| `results/tables/compare.tex` | C: paired test h0 vs h0_sensors | Table 8 |
| `results/figures/success_by_family.png` | C | Figure 1 |
| `annotations/fit_A.csv`, `fit_B.csv`, `fab_A.csv`, `fab_B.csv` | D: grader validation | after labeling, Table 9 |
| `results/error_sample.csv` | Error analysis | Table 10 (you label it) |

After both of you label the annotation sheets **independently** (read
`annotations/*_GUIDELINES.txt` first):

```bash
python -m jobharness.annotate kappa \
  --fit annotations/fit_A.csv annotations/fit_B.csv annotations/fit_key.csv \
  --fab annotations/fab_A.csv annotations/fab_B.csv annotations/fab_key.csv
```

Copy `results/tables/` and `results/figures/` into the Overleaf project next to
`report/milestone2.tex`.

## Later: the full transfer study

```bash
# Tune H_L on the large model and H_S on Qwen (greedy selection over components, dev set),
# save them as configs/harness/hL.yaml and hS.yaml, then:
python -m jobharness.run --tasks data/tasks_test.jsonl \
  --model gpt-oss-120b gpt-oss-20b qwen3-30b-a3b \
  --harness configs/harness/h0.yaml configs/harness/hL.yaml configs/harness/hS.yaml \
  --trials 3 --out runs/test.jsonl
python -m jobharness.analyze runs/test.jsonl --large gpt-oss-120b --h0 h0 --hl hL --hs hS
```

## Rules for reporting

* Only numbers produced by your own runs go in the report. Mock models (`mock-*`) and the
  synthetic fixture exist for tests; never report numbers from them.
* Report failed and surprising results as they are.
* Keep the test split untouched until the final evaluation.
