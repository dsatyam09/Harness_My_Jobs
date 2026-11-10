"""Skill vocabulary, role clusters and text-matching rules.

Everything the graders treat as "ground truth" that is derived from text
(skills in a posting, salary amounts, remote mentions) is defined here, so
the same rules are used consistently and can be reported in the paper.
"""
from __future__ import annotations

import re
from functools import lru_cache

# Role clusters: title keywords (matched against the visible job title) and
# concrete skills (matched against description text). Generic soft skills are
# excluded on purpose: they appear everywhere and would inflate overlap.
CLUSTERS: dict[str, dict[str, list[str]]] = {
    "data": {
        "titles": ["data analyst", "business analyst", "data scientist", "analytics"],
        "skills": ["python", "sql", "excel", "tableau", "power bi", "statistics",
                   "data visualization", "pandas", "spark", "machine learning",
                   "looker", "etl", "a/b testing", "snowflake"],
    },
    "software": {
        "titles": ["software engineer", "developer", "full stack", "backend", "frontend"],
        "skills": ["java", "javascript", "typescript", "react", "node.js", "c++", "c#",
                   "golang", "aws", "azure", "docker", "kubernetes", "git", "rest api",
                   "microservices", "linux", "ci/cd"],
    },
    "ml": {
        "titles": ["machine learning", "ml engineer", "ai engineer"],
        "skills": ["pytorch", "tensorflow", "deep learning", "nlp", "computer vision",
                   "mlops", "scikit-learn", "llm", "python", "aws"],
    },
    "marketing": {
        "titles": ["marketing"],
        "skills": ["seo", "google analytics", "content marketing", "social media",
                   "hubspot", "email marketing", "copywriting", "branding"],
    },
    "sales": {
        "titles": ["sales", "account executive"],
        "skills": ["salesforce", "crm", "negotiation", "lead generation",
                   "account management", "cold calling", "b2b"],
    },
    "healthcare": {
        "titles": ["nurse", "registered nurse", "medical assistant"],
        "skills": ["patient care", "ehr", "epic", "cpr", "bls",
                   "medication administration", "triage"],
    },
    "finance": {
        "titles": ["accountant", "financial analyst", "accounting"],
        "skills": ["gaap", "quickbooks", "financial analysis", "budgeting",
                   "forecasting", "auditing", "reconciliation", "excel"],
    },
    "pm": {
        "titles": ["project manager", "program manager"],
        "skills": ["project management", "agile", "scrum", "jira", "pmp",
                   "stakeholder management"],
    },
}

ALL_SKILLS: list[str] = sorted({s for c in CLUSTERS.values() for s in c["skills"]})

PAY_MULTIPLIER = {"YEARLY": 1, "HOURLY": 2080, "MONTHLY": 12, "WEEKLY": 52, "BIWEEKLY": 26}

DEGREES = ["High school diploma", "Associate degree", "Bachelor's degree",
           "Master's degree", "PhD"]


@lru_cache(maxsize=None)
def _term_regex(term: str) -> re.Pattern:
    # Custom boundaries so terms such as "c++", "c#", "node.js" and "ci/cd" match.
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", re.I)


def skills_in_text(text: str, vocab: list[str] | None = None) -> set[str]:
    text = text or ""
    return {s for s in (vocab or ALL_SKILLS) if _term_regex(s).search(text)}


def cluster_of_title(title: str) -> str | None:
    t = (title or "").lower()
    for name, c in CLUSTERS.items():
        if any(k in t for k in c["titles"]):
            return name
    return None


# --- Salary --------------------------------------------------------------
_MONEY = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s?([kK])?")


def money_amounts(text: str) -> list[float]:
    out = []
    for num, k in _MONEY.findall(text or ""):
        v = float(num.replace(",", ""))
        if k:
            v *= 1000
        out.append(v)
    return out


def amount_in_text(value: float | None, amounts: list[float], tol: float = 0.02) -> bool:
    if value is None or value != value:  # None or NaN
        return False
    return any(abs(a - value) <= tol * max(value, 1.0) for a in amounts)


# --- Remote / work type / level / sponsorship -----------------------------
_REMOTE = re.compile(r"\b(remote|work from home|work-from-home|wfh|telecommut\w*)\b", re.I)
_NOT_REMOTE = re.compile(r"\b(not (a )?remote|no remote|non-remote|not eligible for remote|"
                         r"remote work is not)\b", re.I)


def remote_in_text(text: str) -> bool:
    text = text or ""
    return bool(_REMOTE.search(text)) and not _NOT_REMOTE.search(text)


_WORKTYPE = {
    "Full-time": re.compile(r"\bfull[- ]time\b", re.I),
    "Part-time": re.compile(r"\bpart[- ]time\b", re.I),
    "Contract": re.compile(r"\bcontract(or)?\b", re.I),
    "Internship": re.compile(r"\binternship\b", re.I),
    "Temporary": re.compile(r"\btemporary\b", re.I),
}


def worktypes_in_text(text: str) -> set[str]:
    return {k for k, rx in _WORKTYPE.items() if rx.search(text or "")}


_LEVEL = {
    "Entry level": re.compile(r"\b(entry[- ]level|junior|new grad)\b", re.I),
    "Associate": re.compile(r"\bassociate\b", re.I),
    "Mid-Senior level": re.compile(r"\b(senior|mid[- ]level|sr\.)\b", re.I),
    "Director": re.compile(r"\bdirector\b", re.I),
    "Executive": re.compile(r"\b(executive|vice president|vp|chief)\b", re.I),
    "Internship": re.compile(r"\bintern(ship)?\b", re.I),
}


def level_in_text(level: str | None, text: str) -> bool:
    rx = _LEVEL.get(level or "")
    return bool(rx and rx.search(text or ""))


_YEARS = re.compile(r"(\d{1,2})\+?\s*(?:-|to)?\s*\d{0,2}\s*(?:years|yrs)", re.I)


def years_required(text: str) -> int | None:
    vals = [int(m) for m in _YEARS.findall(text or "") if 0 < int(m) <= 30]
    return min(vals) if vals else None


_SPONSOR_NEG = re.compile(r"\b(not|unable to|no|cannot|can't|will not|won't|does not|do not)\b"
                          r"[^.]{0,60}\bsponsor", re.I)


def sponsorship_unavailable(text: str) -> bool:
    return bool(_SPONSOR_NEG.search(text or ""))


# --- Evidence sentences (used by the evidence tool, not by graders) --------
ASPECT_KEYWORDS = {
    "salary": ["$", "salary", "compensation", "pay", "per hour", "annually"],
    "remote": ["remote", "on-site", "onsite", "hybrid", "in office", "work from home"],
    "location": ["located", "location", "office", "relocat"],
    "sponsorship": ["sponsor", "visa", "authorized to work", "authorization"],
    "experience": ["years", "experience", "senior", "junior", "entry"],
    "skills": ["experience with", "proficien", "knowledge of", "familiar", "skills"],
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if len(s.strip()) > 3]
