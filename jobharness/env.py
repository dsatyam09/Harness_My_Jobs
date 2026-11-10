"""Job-board sandbox. Tools return only visible fields: title, company,
location and description text. Structured columns stay hidden."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import pandas as pd

from . import vocab

VISIBLE = ["job_id", "title", "company_name", "location", "description"]
_TOK = re.compile(r"[a-z0-9+#./]+")


def tokenize(s: str) -> list[str]:
    return _TOK.findall((s or "").lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs, self.lens = [], []
        df = Counter()
        for d in docs:
            tf = Counter(tokenize(d))
            self.tfs.append(tf)
            self.lens.append(sum(tf.values()))
            df.update(tf.keys())
        n = len(docs)
        self.avg = sum(self.lens) / max(n, 1)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.inv = defaultdict(list)
        for i, tf in enumerate(self.tfs):
            for t in tf:
                self.inv[t].append(i)

    def search(self, query: str, k: int) -> list[int]:
        scores = defaultdict(float)
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.inv[t]:
                tf = self.tfs[i][t]
                denom = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]


class JobBoard:
    def __init__(self, postings: pd.DataFrame, profiles: list[dict]):
        self.df = postings.reset_index(drop=True)
        self.by_id = {str(r["job_id"]): r for r in self.df[VISIBLE].to_dict("records")}
        self.profiles = {p["profile_id"]: p for p in profiles}
        # Title is repeated so title matches outrank incidental mentions.
        self.bm25 = BM25([f"{r['title']} {r['title']} {r['description']}" for r in self.by_id.values()])
        self.ids = list(self.by_id.keys())

    # ---- search ----
    def search(self, query: str, k: int = 10) -> list[dict]:
        k = max(1, min(int(k or 10), 25))
        return [self._card(self.ids[i]) for i in self.bm25.search(query, k)]

    def search_by_title(self, keyword: str, k: int = 25) -> list[dict]:
        kw = (keyword or "").lower()
        hits = [j for j, r in self.by_id.items() if kw and kw in str(r["title"]).lower()]
        return [self._card(j) for j in hits[: max(1, min(int(k or 25), 50))]]

    def _card(self, job_id: str) -> dict:
        r = self.by_id[job_id]
        return {"job_id": job_id, "title": r["title"], "company": r["company_name"],
                "location": r["location"], "snippet": str(r["description"])[:160]}

    # ---- reading ----
    def get_job(self, job_id: str, long_text: str = "full") -> dict:
        r = self.by_id.get(str(job_id))
        if r is None:
            return {"error": f"Unknown job_id {job_id}"}
        desc = str(r["description"])
        if long_text == "truncate":
            desc = desc[:1500] + (" [truncated]" if len(desc) > 1500 else "")
        elif long_text == "key_sections":
            keys = [kw for kws in vocab.ASPECT_KEYWORDS.values() for kw in kws]
            sents = [s for s in vocab.sentences(desc) if any(k in s.lower() for k in keys)]
            desc = desc[:600] + "\n[Key sentences]\n" + "\n".join(sents[:25])
        return {"job_id": str(job_id), "title": r["title"], "company": r["company_name"],
                "location": r["location"], "description": desc}

    def evidence(self, job_id: str, aspect: str) -> dict:
        """Sentence-level evidence lookup. Returns sentences, never a verdict,
        so it cannot leak the hidden answer key."""
        r = self.by_id.get(str(job_id))
        if r is None:
            return {"error": f"Unknown job_id {job_id}"}
        kws = vocab.ASPECT_KEYWORDS.get((aspect or "").lower())
        if not kws:
            return {"error": f"aspect must be one of {sorted(vocab.ASPECT_KEYWORDS)}"}
        sents = [s for s in vocab.sentences(str(r["description"])) if any(k in s.lower() for k in kws)]
        return {"job_id": str(job_id), "aspect": aspect, "location": r["location"],
                "sentences": sents[:12]}

    def profile(self, profile_id: str, part: str = "all") -> dict:
        p = self.profiles.get(profile_id)
        if p is None:
            return {"error": f"Unknown profile_id {profile_id}"}
        if part == "skills":
            return {"profile_id": profile_id, "skills": p["skills"]}
        if part == "preferences":
            keys = ["state", "open_to_remote", "open_to_relocate", "needs_sponsorship",
                    "min_salary", "experience_level", "years_experience", "highest_degree"]
            return {"profile_id": profile_id, **{k: p[k] for k in keys}}
        if part == "history":
            return {"profile_id": profile_id, "work_history": p["work_history"]}
        return {k: v for k, v in p.items() if k != "cluster"}
