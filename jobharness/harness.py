"""Modular agent harness.

Every component is a switch in a YAML profile (see configs/harness/):
  tools               broad | narrow                      (structure)
  tool_format         native | text                       (structure)
  instructions        minimal | workflow                  (guide)
  profile_card        true | false                        (guide)
  evidence_tool       true | false                        (sensor: sentence lookup, no verdicts)
  verify_before_submit true | false                       (sensor: one self-check round)
  claim_audit         true | false                        (sensor: tailoring self-audit)
  long_text           full | truncate | key_sections      (structure)
  recovery            none | retry | retry_loop           (sensor)
  max_steps           int
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass

import yaml

from .env import JobBoard
from .llm import ContextOverflow, Reply
from .profiles import profile_card


@dataclass
class HarnessConfig:
    name: str = "h0"
    tools: str = "broad"
    tool_format: str = "native"
    instructions: str = "minimal"
    profile_card: bool = False
    evidence_tool: bool = False
    verify_before_submit: bool = False
    claim_audit: bool = False
    long_text: str = "full"
    recovery: str = "none"
    max_steps: int = 20

    @classmethod
    def load(cls, path: str) -> "HarnessConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))


MINIMAL = ("You are a job-search assistant with access to a job board and candidate profiles "
           "through tools. Complete the user's task, then call submit_answer with the answer in "
           "the JSON format the task asks for.")

WORKFLOW = MINIMAL + """

Work in this order:
1. List every requirement in the task (title words, remote work, salary floors and pay periods, candidate preferences, skills).
2. Search more than once with different queries so you do not miss relevant postings.
3. Open each candidate posting and check every requirement against its description text. Convert hourly pay at 2,080 hours per year and monthly pay at 12 months.
4. For candidate tasks, read the profile and use only facts it contains. Never invent skills, employers, years or degrees.
5. Only then call submit_answer with exactly the JSON fields requested."""

VERIFY_MSG = ("Before your answer is accepted, check it once more: confirm every item against "
              "the job text or the candidate profile (use tools if needed). Then call submit_answer "
              "again with your final answer, even if it is unchanged.")
AUDIT_MSG = ("Audit your bullets: for each skill, tool, employer and number you mention, name the "
             "profile fact that supports it. Remove or rewrite anything the profile does not "
             "support, then call submit_answer again.")


def _s(t, d=""):
    return {"type": "string", "description": d}


def _i(d=""):
    return {"type": "integer", "description": d}


def tool_specs(cfg: HarnessConfig) -> list[tuple[str, str, dict]]:
    if cfg.tools == "broad":
        specs = [
            ("search_jobs", "Keyword search over job titles and descriptions. Returns job cards.",
             {"query": _s("str", "search keywords"), "k": _i("number of results, max 25")}, ["query"]),
            ("get_job", "Read one job posting: title, company, location and description.",
             {"job_id": _s("str")}, ["job_id"]),
            ("get_profile", "Read a candidate profile.", {"profile_id": _s("str")}, ["profile_id"]),
        ]
    else:
        specs = [
            ("search_by_title", "List postings whose title contains a keyword.",
             {"keyword": _s("str"), "k": _i("max 50")}, ["keyword"]),
            ("search_by_text", "Keyword search over descriptions.",
             {"query": _s("str"), "k": _i("max 25")}, ["query"]),
            ("get_job_header", "Title, company and location of a posting.", {"job_id": _s("str")}, ["job_id"]),
            ("get_job_description", "Description text of a posting.", {"job_id": _s("str")}, ["job_id"]),
            ("get_profile_skills", "Skills listed in a candidate profile.", {"profile_id": _s("str")}, ["profile_id"]),
            ("get_profile_preferences", "Location, remote, salary, sponsorship and degree.",
             {"profile_id": _s("str")}, ["profile_id"]),
            ("get_profile_history", "Work history with years and skills per job.",
             {"profile_id": _s("str")}, ["profile_id"]),
        ]
    if cfg.evidence_tool:
        specs.append(("find_evidence", "Return the sentences of a posting that mention an aspect: "
                      "salary, remote, location, sponsorship, experience or skills.",
                      {"job_id": _s("str"), "aspect": _s("str")}, ["job_id", "aspect"]))
    specs.append(("submit_answer", "Submit the final answer as a JSON object.",
                  {"answer": {"type": "object", "description": "the JSON answer the task asks for"}},
                  ["answer"]))
    return [(n, d, {"type": "object", "properties": p, "required": r}) for n, d, p, r in specs]


def openai_tools(specs):
    return [{"type": "function", "function": {"name": n, "description": d, "parameters": p}}
            for n, d, p in specs]


def text_tool_prompt(specs) -> str:
    lines = ["You can call exactly one tool per reply. Write the call as",
             '<tool_call>{"name": TOOL_NAME, "arguments": {...}}</tool_call>',
             "Available tools:"]
    for n, d, p in specs:
        args = ", ".join(p["properties"])
        lines.append(f"- {n}({args}): {d}")
    return "\n".join(lines)


_CALL_RX = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_text_call(content: str):
    m = _CALL_RX.search(content or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
        return {"id": "t", "name": obj["name"], "arguments": json.dumps(obj.get("arguments", {}))}
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def validate(task: dict, ans) -> str | None:
    """Return an error message if the answer does not match the required schema."""
    if not isinstance(ans, dict):
        return "answer must be a JSON object"
    fam = task["family"]
    need = {"search": ["job_ids"], "extraction": ["max_salary", "pay_period", "remote_allowed"],
            "ranking": ["ranking"], "application": ["needs_sponsorship", "years_with_skill",
            "open_to_remote", "minimum_salary", "highest_degree"], "tailoring": ["bullets"]}[fam]
    missing = [k for k in need if k not in ans]
    if missing:
        return f"missing fields: {missing}"
    if fam == "search" and not isinstance(ans["job_ids"], list):
        return "job_ids must be a list"
    if fam == "ranking" and sorted(map(str, ans["ranking"])) != sorted(task["job_ids"]):
        return "ranking must contain each given job ID exactly once"
    if fam == "tailoring" and (not isinstance(ans["bullets"], list) or len(ans["bullets"]) != 3):
        return "bullets must be a list of exactly three strings"
    return None


class Episode:
    def __init__(self, llm, board: JobBoard, task: dict, cfg: HarnessConfig):
        self.llm, self.board, self.task, self.cfg = llm, board, task, cfg
        self.specs = tool_specs(cfg)
        self.m = {"steps": 0, "turns": 0, "malformed": 0, "invalid_answers": 0, "loops": 0,
                  "overflow": False, "prompt_tokens": 0, "completion_tokens": 0,
                  "verify_rounds": 0, "finish_reason": None}
        self.traj: list[dict] = []
        self.answer = None
        self._checked = False
        self._recent: list[str] = []
        self._consec_bad = 0

    # ---------- tools ----------
    def dispatch(self, name: str, args: dict):
        b, lt = self.board, self.cfg.long_text
        table = {
            "search_jobs": lambda: b.search(args.get("query", ""), args.get("k", 10)),
            "get_job": lambda: b.get_job(args.get("job_id", ""), lt),
            "get_profile": lambda: b.profile(args.get("profile_id", "")),
            "search_by_title": lambda: b.search_by_title(args.get("keyword", ""), args.get("k", 25)),
            "search_by_text": lambda: b.search(args.get("query", ""), args.get("k", 10)),
            "get_job_header": lambda: {k: v for k, v in b.get_job(args.get("job_id", "")).items()
                                       if k != "description"},
            "get_job_description": lambda: b.get_job(args.get("job_id", ""), lt),
            "get_profile_skills": lambda: b.profile(args.get("profile_id", ""), "skills"),
            "get_profile_preferences": lambda: b.profile(args.get("profile_id", ""), "preferences"),
            "get_profile_history": lambda: b.profile(args.get("profile_id", ""), "history"),
            "find_evidence": lambda: b.evidence(args.get("job_id", ""), args.get("aspect", "")),
        }
        allowed = {n for n, _, _ in self.specs}
        if name not in allowed or (name not in table and name != "submit_answer"):
            return None
        return table[name]()

    # ---------- loop ----------
    def system_prompt(self) -> str:
        s = WORKFLOW if self.cfg.instructions == "workflow" else MINIMAL
        if self.cfg.tool_format == "text":
            s += "\n\n" + text_tool_prompt(self.specs)
        return s

    def user_prompt(self) -> str:
        u = self.task["instruction"]
        pid = self.task.get("profile_id")
        if self.cfg.profile_card and pid:
            u += "\n\n" + profile_card(self.board.profiles[pid])
        return u

    def _bad(self, messages, reason: str, reply_msg: str) -> bool:
        """Record a malformed step. Returns True if the episode should stop."""
        self.m["malformed"] += 1
        self._consec_bad += 1
        if self.cfg.recovery == "none" or self._consec_bad > 3:
            self.m["finish_reason"] = reason
            return True
        messages.append({"role": "user", "content": reply_msg})
        return False

    def run(self) -> dict:
        t0 = time.time()
        messages = [{"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": self.user_prompt()}]
        native = self.cfg.tool_format == "native"
        tools = openai_tools(self.specs) if native else None
        while self.m["finish_reason"] is None:
            if self.m["turns"] >= self.cfg.max_steps + 10:
                self.m["finish_reason"] = "max_turns"
                break
            self.m["turns"] += 1
            try:
                r: Reply = self.llm.chat(messages, tools)
            except ContextOverflow:
                self.m["overflow"] = True
                self.m["finish_reason"] = "context_overflow"
                break
            self.m["prompt_tokens"] += r.prompt_tokens
            self.m["completion_tokens"] += r.completion_tokens
            calls = r.tool_calls if native else [c for c in [parse_text_call(r.content)] if c]
            self.traj.append({"turn": self.m["turns"], "content": (r.content or "")[:1500],
                              "calls": [{"name": c["name"], "arguments": c["arguments"][:1500]}
                                        for c in calls]})
            if native:
                msg = {"role": "assistant", "content": r.content or ""}
                if calls:
                    msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                          "function": {"name": c["name"], "arguments": c["arguments"]}}
                                         for c in calls]
                messages.append(msg)
            else:
                messages.append({"role": "assistant", "content": r.content or ""})
            if not calls:
                if self._bad(messages, "no_tool_call",
                             "No valid tool call found. Call a tool; to finish, call submit_answer."):
                    break
                continue
            for c in calls:
                if self.m["finish_reason"]:
                    break
                self._handle_call(c, messages, native)
        self.m["seconds"] = round(time.time() - t0, 2)
        return {"answer": self.answer, "metrics": self.m, "trajectory": self.traj,
                "harness": asdict(self.cfg)}

    def _tool_result(self, messages, native, call_id, name, content):
        text = json.dumps(content, default=str) if not isinstance(content, str) else content
        if native:
            messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        else:
            messages.append({"role": "user", "content": f"TOOL RESULT ({name}):\n{text}"})

    def _handle_call(self, c, messages, native):
        name = c["name"]
        try:
            args = json.loads(c["arguments"] or "{}")
            if not isinstance(args, dict):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            self.m["malformed"] += 1
            self._consec_bad += 1
            if self.cfg.recovery == "none" or self._consec_bad > 3:
                self.m["finish_reason"] = "malformed_arguments"
                if native:  # keep the transcript valid for the API
                    self._tool_result(messages, native, c["id"], name, "error")
                return
            self._tool_result(messages, native, c["id"], name,
                              {"error": "arguments were not valid JSON; try again"})
            return

        self.m["steps"] += 1
        sig = name + json.dumps(args, sort_keys=True)
        self._recent = (self._recent + [sig])[-3:]
        looping = len(self._recent) == 3 and len(set(self._recent)) == 1
        if looping:
            self.m["loops"] += 1

        if name == "submit_answer":
            ans = args.get("answer", args)
            if isinstance(ans, str):
                try:
                    ans = json.loads(ans)
                except json.JSONDecodeError:
                    pass
            err = validate(self.task, ans)
            if err:
                self.m["invalid_answers"] += 1
                self._consec_bad += 1
                if self.cfg.recovery == "none" or self._consec_bad > 3:
                    self.answer = ans
                    self.m["finish_reason"] = "invalid_answer"
                    self._tool_result(messages, native, c["id"], name, {"error": err})
                    return
                self._tool_result(messages, native, c["id"], name, {"error": err})
                return
            self._consec_bad = 0
            want_verify = self.cfg.verify_before_submit
            want_audit = self.cfg.claim_audit and self.task["family"] == "tailoring"
            if (want_verify or want_audit) and not self._checked:
                self._checked = True
                self.m["verify_rounds"] += 1
                note = " ".join(m for ok, m in [(want_verify, VERIFY_MSG), (want_audit, AUDIT_MSG)] if ok)
                self._tool_result(messages, native, c["id"], name, note)
                return
            self.answer = ans
            self.m["finish_reason"] = "submitted"
            self._tool_result(messages, native, c["id"], name, "accepted")
            return

        result = self.dispatch(name, args)
        if result is None:
            self.m["malformed"] += 1
            self._consec_bad += 1
            if self.cfg.recovery == "none" or self._consec_bad > 3:
                self.m["finish_reason"] = "unknown_tool"
                self._tool_result(messages, native, c["id"], name, "error")
                return
            self._tool_result(messages, native, c["id"], name, {"error": f"unknown tool {name}"})
            return
        self._consec_bad = 0
        if looping and self.cfg.recovery == "retry_loop":
            result = {"warning": "You made the same call three times in a row. Try a different "
                                 "action or submit your answer.", "result": result}
        self._tool_result(messages, native, c["id"], name, result)
        if self.m["steps"] >= self.cfg.max_steps and self.m["finish_reason"] is None:
            self.m["finish_reason"] = "max_steps"


def run_episode(llm, board, task, cfg) -> dict:
    return Episode(llm, board, task, cfg).run()
