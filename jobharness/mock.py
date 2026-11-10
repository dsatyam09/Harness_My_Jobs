"""Mock models for unit tests and CI plumbing checks. NOT reportable results.

mock-oracle     reads one tool, then submits the correct answer (tests graders and loop)
mock-notool     replies in plain text and never calls a tool (tests failure handling)
mock-badjson    calls submit_answer with an answer missing required fields
"""
from __future__ import annotations

import json

from .llm import Reply


def oracle_answer(task, board):
    a, fam = task["answer"], task["family"]
    if fam == "search":
        return {"job_ids": a["job_ids"]}
    if fam in ("extraction", "application"):
        return dict(a)
    if fam == "ranking":
        return {"ranking": sorted(task["job_ids"], key=lambda j: -a["labels"][j])}
    if fam == "tailoring":
        p = board.profiles[task["profile_id"]]
        sk = list(a["target_skills"]) or p["skills"][:1]
        chunks = [sk[i::3] for i in range(3)]
        return {"bullets": [f"Applied {', '.join(c) or 'core methods'} to deliver team projects"
                            for c in chunks]}
    raise ValueError(fam)


class _Mock:
    def __init__(self, kind, task=None, board=None):
        self.kind, self.task, self.board, self.turn = kind, task, board, 0

    def bind(self, task, board):
        return _Mock(self.kind, task, board)

    def chat(self, messages, tools=None) -> Reply:
        self.turn += 1
        native = tools is not None
        if self.kind == "notool":
            return Reply(content="I think the answer is probably the first job.")
        if self.kind == "badjson":
            call = {"id": f"c{self.turn}", "name": "submit_answer",
                    "arguments": json.dumps({"answer": {"wrong": 1}})}
        elif self.turn == 1 and self.task.get("profile_id"):
            call = {"id": "c1", "name": "get_profile" if _has(tools, messages, "get_profile")
                    else "get_profile_skills",
                    "arguments": json.dumps({"profile_id": self.task["profile_id"]})}
        else:
            call = {"id": f"c{self.turn}", "name": "submit_answer",
                    "arguments": json.dumps({"answer": oracle_answer(self.task, self.board)})}
        if native:
            return Reply(content="", tool_calls=[call], prompt_tokens=100, completion_tokens=20)
        obj = {"name": call["name"], "arguments": json.loads(call["arguments"])}
        return Reply(content=f"<tool_call>{json.dumps(obj)}</tool_call>",
                     prompt_tokens=100, completion_tokens=20)


def _has(tools, messages, name):
    if tools:
        return any(t["function"]["name"] == name for t in tools)
    return f"- {name}(" in messages[0]["content"]


def mock_llm(name: str):
    return _Mock(name.removeprefix("mock-"))
