# Harness_my_jobs (JobHarness) — project memory for Claude Code

Course project (IU graduate NLP, Applied NLP Project). Team: PN Tejeshwar Sarma and Satyam Dubey (sbdubey@iu.edu).
Currently preparing **Milestone 2 (progress report)**; the final report follows.
Last updated: 2026-11-10.

## Rules (read first)
1. **Never fabricate results.** Every number in a report comes from real runs or real human labels. Say plainly what has not been run.
2. **Open-weight models only, self-hosted with vLLM.** No closed APIs for any reported number.
3. `mock-*` models and `tests/fixtures/postings_synthetic.csv` are for tests only; never in reported results.
4. **`data/tasks_test.jsonl` is locked** for the final evaluation. Tune and decide on `tasks_dev.jsonl` only.
5. Report negative and surprising results as they are.
6. Keep an AI-use log (see bottom); the course requires one checked/corrected/rejected AI output.
7. **Never commit `data/`, `runs/`, Kaggle CSVs, or annotation keys** (CC BY-SA data; already in .gitignore).
8. All candidate profiles are synthetic. No real personal data.
9. Humans (the two students) do the annotation, error-category labels and contribution statements. Claude must not fill human labels.

## The research question
If a job-search agent's harness is tuned on a large open model (gpt-oss-120b), does it help small self-hosted MoE models (gpt-oss-20b, Qwen3-30B-A3B-Instruct-2507), or does the benefit stay with the large model?
SQ1 downward transfer, SQ2 which components transfer (guides / sensors / structure), SQ3 gap closure and cost, SQ4 sub-skills.
Hypotheses: H1 partial transfer; H2 sensors transfer best, long guides/complex tools worst; H3 small-model tuning closes part of the gap cheaply.
Metrics: success rate, transfer ratio R(m)=(S_m(HL)-S_m(H0))/(S_L(HL)-S_L(H0)), gap closure C(m)=(S_m(HS)-S_m(H0))/(S_L(HL)-S_m(H0)), pass^3, process metrics, bootstrap CIs, exact McNemar, Cohen's kappa.

## Milestone 2 grading (100 pts) — what matters
Updated problem 15, methods/data 20, **preliminary evidence 25 (concrete pilot with inspectable artifacts)**, evaluation/interpretation 20, challenges/next steps 10, docs/refs/AI-use/contribution statements 10.
Required sections: identification+updated question, work completed, methods+evidence, preliminary results, evaluation+interpretation, challenges+next steps, updated references, AI-use note, individual contribution statements. Each member submits individually via Canvas (PDF/Word) with working links.
Report template: `report/milestone2.tex` (red `\fillin{...}` placeholders must all be replaced; tables come from `results/tables/*.tex`, figure from `results/figures/`).
Prior milestone lost 2 marks; reason unknown — ask the students for the feedback text.

## Environment (IU Quartz cluster, host h2.quartz.uits.iu.edu)
- Repo: `/N/u/sbdubey/Quartz/Harness_my_jobs` (same as `/geode3/home/u080/sbdubey/Quartz/Harness_my_jobs`).
- Venv (Python 3.12.4, outside repo): `/N/u/sbdubey/Quartz/Harness_my_jobs_venv`.
  Activate: `cd /N/u/sbdubey/Quartz/Harness_my_jobs && module load python/3.12.4 && source ../Harness_my_jobs_venv/bin/activate`
  (After `module load`, the binary is `python3.12`; bare `python3` is still the system 3.6.)
- Raw Kaggle data: `/N/u/sbdubey/Quartz/kaggle_raw/postings.csv` (from `archive.zip`, which also holds companies/jobs/mappings CSVs; only postings.csv is extracted).
- Login node has no GPU. GPU partitions: h100-debug (1h), h100-interactive (4h), h100-single/multi (2 days), v100. Slurm account `r01789`, QOS: allocated, allocated-hopper, hopper. Scratch: `/N/scratch/sbdubey` (use for model weights).
- `scripts/slurm_vllm.sbatch` is a template; it needs editing (partition `h100-*`, account `r01789`, venv path).
- Cluster check output: `/N/u/sbdubey/Quartz/cluster_check_h2_2026-11-07.txt`.

## Status (as of 2026-11-10) — all real, run on Quartz
- 16/16 tests pass in the new env.
- **Pilot A, data audit** (8,000 sampled postings, seed 13): salary filled 23.7%, stated in text when filled 54.9%, consistent 66.7%; remote filled 12.4%, stated 46.8%, consistent 87.3%; work type stated 27.4%; **experience level stated in text only 16.7%** (justifies dropping it as a search constraint). Sandbox kept 4,663 postings; 1,844 have a role cluster; years-required stated in 50.2%.
- **Tasks:** 64 profiles; 5 families; 48 tasks per family reached (12 dev / 36 test each; 60 dev / 180 test).
- **Baselines (dev, n=12 per family):** bm25-top5 0%, title-only 0% (search); regex 75% (extraction, label-aligned); random 0%, skill-overlap 58% (ranking, label-aligned); majority 0% (application); **copy-profile 100% (tailoring)**.

### Issues found, to fix before final results
1. **Tailoring grader is too easy**: the copy-profile baseline passes 12/12. Tailoring currently tests nothing; require job-specific content or tighten coverage/relevance. Document the change.
2. **Search baselines score 0%**: check whether the F1>=0.8 threshold is too strict for small answer sets (2–9) or the baselines are weak; inspect before trusting.
3. Regexes not yet hand-spot-checked on 50–100 real postings.
4. `highest_degree` is graded by exact string match (paraphrase risk).

### Not done yet
- vLLM not installed; no model weights downloaded; no Slurm jobs submitted; **no LLM pilot runs exist**. Awaiting the user's go-ahead to: install vLLM (separate env on an H100 node), download weights to scratch (gpt-oss-20b ~40GB, Qwen3-30B-A3B ~60GB, optionally gpt-oss-120b ~65GB), submit a 1h h100-debug smoke test under account r01789.
- Then: smoke test `--limit-per-family 1`, verify tool-parser flags for the installed vLLM version (gpt-oss: `--tool-call-parser openai`; Qwen: `hermes`), run pilot (dev, h0/h0_guides/h0_sensors, 1 trial), analyze, annotation (humans), error sample (humans), fill report.
- Unimplemented for the full study: `tune.py` (greedy forward selection, <=12 configs, equal budget for H_L on gpt-oss-120b and H_S on Qwen3), ablation generator, mixed-effects model, GPU-hour cost metric, workflow family, TalentCLEF/JobResQA loaders, copilot UI, fairness probe, MLflow, `configs/gate_baseline.json`.

## Commands (run from repo root, venv active)
```bash
python -m jobharness.data --csv ../kaggle_raw/postings.csv --n 8000 --seed 13 --out data --tables results/tables
python -m jobharness.tasks --data data --per-family 48 --profiles-per-cluster 8 --list-size 10 --dev-frac 0.25 --seed 11 --tables results/tables
python -m jobharness.baselines --data data --tasks data/tasks_dev.jsonl --out runs/baselines.jsonl
python -m jobharness.run --data data --tasks data/tasks_dev.jsonl --models configs/models.yaml \
  --model gpt-oss-20b qwen3-30b-a3b --harness configs/harness/h0.yaml configs/harness/h0_guides.yaml configs/harness/h0_sensors.yaml \
  --trials 1 --concurrency 8 --out runs/pilot.jsonl      # resumes; --limit-per-family 1 for smoke tests
python -m jobharness.analyze runs/pilot.jsonl runs/baselines.jsonl --compare h0 h0_sensors --out results
python -m jobharness.annotate export-fit --tasks data/tasks_dev.jsonl --n 50     # then export-fab, kappa
pytest -q
```
Code layout: `jobharness/{data,tasks,graders,env,harness,llm,run,baselines,analyze,annotate,gate,mcp_server,profiles,vocab,latex,mock}.py`; configs in `configs/` (models.yaml, harness/*.yaml); `scripts/serve_vllm.sh`, `scripts/run_pilot.sh`.
Task families: search, extraction, ranking (10 postings, nDCG@5), application, tailoring. Harness components: tools broad/narrow, tool_format native/text, instructions, profile_card, evidence_tool, verify_before_submit, claim_audit, long_text, recovery. Conditions: H0 seed, H_L (tuned on gpt-oss-120b), H_S (tuned on Qwen3-30B-A3B), same budget.
Design change vs Milestone 1: constraint/fabrication code checkers replaced by an evidence-lookup tool (returns sentences, never verdicts) plus model self-checks, because code checkers would leak the answer key.

## Run budget
Pilot ~360–540 runs; tuning ~1,440; transfer matrix 180 tasks x 3 harnesses x 3 models x 3 trials = 4,860; ablations ~4,320. Fallback: 120 test tasks, 2 trials.

## Open questions for the students
1. Which proposal was submitted as Milestone 1: JobHarness or the earlier PromptGuard (safety fine-tuning generalization)? If PromptGuard, Milestone 2 section 1 must justify the change of direction.
2. Text of the professor's feedback behind the 2-mark deduction.
3. Milestone 2 and final deadlines. GitHub repo URL (name: Harness_my_jobs).
4. Which models to run (include gpt-oss-120b?).
Four references were flagged as unverified: Lin et al. author list (2604.25850), Jiang et al. first author (2505.08120), Wilson & Caliskan arXiv ID (2407.20371), gpt-oss model card arXiv ID (2508.10925). Verify before submission.

## AI-use log (for the required note)
- Milestone 1 design (AI-drafted) used code-based checkers that would leak the answer key; redesigned.
- AI-written MCP server imported FastMCP from the 1.x path, failing on SDK 2.x; caught in tests, fixed.
- Session 2026-11-08: Claude found that the copy-profile baseline passes all tailoring tasks (grader too easy) — a flaw in the AI-drafted grader design.
- Add new cases here as they happen.
