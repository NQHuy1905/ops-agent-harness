"""Keyword-overlap search over the mock runbook knowledge base."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.tools.base import Tool, ToolContext

_STOPWORDS = frozenset(
    "a an and are as at be by for from how i in is it of on or the to what when where why with do does".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SNIPPET_LEN = 200
_MAX_RESULTS = 5
_kb_cache: dict[Path, list[dict]] = {}


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS}


def _load_kb(path: Path) -> list[dict]:
    if path not in _kb_cache:
        _kb_cache[path] = json.loads(path.read_text(encoding="utf-8"))
    return _kb_cache[path]


def _snippet(text: str) -> str:
    return text if len(text) <= _SNIPPET_LEN else text[:_SNIPPET_LEN].rstrip() + "..."


class SearchKBInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)] = Field(
        description="Keywords or a short question, e.g. 'payments-api high latency' or 'severity matrix'."
    )


class KBHit(BaseModel):
    id: str
    title: str
    snippet: str
    score: float = Field(ge=0)
    service: str | None = None


class SearchKBOutput(BaseModel):
    results: list[KBHit] = Field(max_length=_MAX_RESULTS)
    total_matches: int = Field(ge=0)


class SearchKnowledgeBaseTool(Tool):
    name = "search_knowledge_base"
    description = (
        "Search the internal operations knowledge base (runbooks, severity matrix, incident policy, "
        "escalation and rollback procedures) by keyword. Returns up to 5 ranked documents with snippets. "
        "Use it to find troubleshooting steps or policies before acting."
    )
    input_model = SearchKBInput
    output_model = SearchKBOutput

    async def run(self, args: BaseModel, ctx: ToolContext) -> dict:
        assert isinstance(args, SearchKBInput)
        docs = _load_kb(Path(ctx.settings.mock_data_dir) / "knowledge_base.json")
        q = _tokens(args.query)
        scored: list[tuple[float, dict]] = []
        for doc in docs:
            strong = _tokens(" ".join([doc["title"], *doc.get("tags", []), doc.get("service") or ""]))
            weak = _tokens(doc["content"])
            score = 2.0 * len(q & strong) + 1.0 * len((q & weak) - strong)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda p: (-p[0], p[1]["id"]))
        hits = [
            {
                "id": d["id"],
                "title": d["title"],
                "snippet": _snippet(d["content"]),
                "score": s,
                "service": d.get("service"),
            }
            for s, d in scored[:_MAX_RESULTS]
        ]
        return {"results": hits, "total_matches": len(scored)}
