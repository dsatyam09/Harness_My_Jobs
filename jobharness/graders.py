"""Automatic graders. Each returns {"success": bool, "score": float, ...details}."""
from __future__ import annotations

import math
import re

from . import vocab

SUCCESS_F1 = 0.8
SUCCESS_NDCG = 0.8
SUCCESS_COVERAGE = 0.5
SUCCESS_RELEVANCE = 0.6  # share of skills named in the bullets that the job actually asks for


def f1(pred: set, gold: set) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def ndcg_at_k(ranking: list[str], labels: dict[str, int], k: int) -> float:
    def dcg(rels):
        return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels[:k]))
    ideal = dcg(sorted(labels.values(), reverse=True))
    return dcg([labels.get(j, 0) for j in ranking]) / ideal if ideal > 0 else 0.0


def _num_eq(a, b, tol=0.01) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return abs(float(a) - float(b)) <= tol * max(abs(float(b)), 1.0)
    except (TypeError, ValueError):
        return False


def _bool(v):
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1"}
    return bool(v)


def grade_search(task, ans):
    pred = {str(x) for x in (ans or {}).get("job_ids", [])}
    gold = set(map(str, task["answer"]["job_ids"]))
    p, r, f = f1(pred, gold)
    return {"success": f >= SUCCESS_F1, "score": f, "precision": p, "recall": r}


def grade_extraction(task, ans):
    gold, ans = task["answer"], ans or {}
    checks = {
        "min_salary": _num_eq(ans.get("min_salary"), gold["min_salary"]),
        "max_salary": _num_eq(ans.get("max_salary"), gold["max_salary"]),
        "pay_period": str(ans.get("pay_period", "")).upper() == str(gold["pay_period"]).upper(),
        "remote_allowed": _bool(ans.get("remote_allowed")) == gold["remote_allowed"],
    }
    acc = sum(checks.values()) / len(checks)
    return {"success": all(checks.values()), "score": acc, "fields": checks}


def grade_ranking(task, ans, k: int = 5):
    ranking = [str(x) for x in (ans or {}).get("ranking", [])]
    labels = {str(a): b for a, b in task["answer"]["labels"].items()}
    valid = sorted(ranking) == sorted(labels)
    score = ndcg_at_k(ranking, labels, k) if valid else 0.0
    return {"success": valid and score >= SUCCESS_NDCG, "score": score, "valid_permutation": valid}


def grade_application(task, ans):
    gold, ans = task["answer"], ans or {}
    checks = {
        "needs_sponsorship": _bool(ans.get("needs_sponsorship")) == gold["needs_sponsorship"],
        "years_with_skill": _num_eq(ans.get("years_with_skill"), gold["years_with_skill"], 0),
        "open_to_remote": _bool(ans.get("open_to_remote")) == gold["open_to_remote"],
        "minimum_salary": _num_eq(ans.get("minimum_salary"), gold["minimum_salary"], 0),
        "highest_degree": str(ans.get("highest_degree", "")).strip().lower()
                          == gold["highest_degree"].lower(),
    }
    acc = sum(checks.values()) / len(checks)
    return {"success": all(checks.values()), "score": acc, "fields": checks}


_YEARS_CLAIM = re.compile(r"(\d{1,2})\+?\s*(?:years|yrs)", re.I)


def fabrication_check(bullets: list[str], profile: dict) -> dict:
    """Skills from the vocabulary that the profile lacks, and inflated year claims."""
    text = " ".join(bullets)
    mentioned = vocab.skills_in_text(text)
    fabricated = sorted(mentioned - set(profile["skills"]))
    max_years = profile["years_experience"]
    inflated = [int(y) for y in _YEARS_CLAIM.findall(text) if int(y) > max_years]
    return {"mentioned": sorted(mentioned), "fabricated_skills": fabricated,
            "inflated_years": inflated, "n_fabricated": len(fabricated) + len(inflated)}


def grade_tailoring(task, ans, profile):
    """Tailored = covers the job-relevant skills AND stays focused on them.

    target_skills = job skills the candidate truly has. coverage = share of target mentioned;
    relevance = share of mentioned skills that are in target. A copy of the whole profile has
    full coverage but low relevance, so it fails (v2, 2026-11-10; v1 had no relevance term).
    """
    bullets = [str(b) for b in (ans or {}).get("bullets", [])]
    fab = fabrication_check(bullets, profile)
    target = set(task["answer"]["target_skills"])
    mentioned = set(fab["mentioned"])
    coverage = len(mentioned & target) / len(target) if target else 1.0
    relevance = len(mentioned & target) / len(mentioned) if mentioned else 0.0
    ok = (len(bullets) == 3 and fab["n_fabricated"] == 0 and coverage >= SUCCESS_COVERAGE
          and relevance >= SUCCESS_RELEVANCE)
    f = 2 * coverage * relevance / (coverage + relevance) if coverage + relevance else 0.0
    return {"success": ok, "score": f if fab["n_fabricated"] == 0 else 0.0,
            "coverage": coverage, "relevance": relevance, "n_bullets": len(bullets), **fab}


def grade(task, ans, profile=None):
    fam = task["family"]
    if fam == "search":
        return grade_search(task, ans)
    if fam == "extraction":
        return grade_extraction(task, ans)
    if fam == "ranking":
        return grade_ranking(task, ans)
    if fam == "application":
        return grade_application(task, ans)
    if fam == "tailoring":
        return grade_tailoring(task, ans, profile)
    raise ValueError(fam)
