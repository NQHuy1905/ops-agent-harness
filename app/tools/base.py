"""Tool abstraction: typed input/output models, validation, and OpenAI-format spec."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.core.errors import ToolInputValidationError, ToolOutputValidationError
from app.storage.repository import RunRepository


@dataclass
class ToolContext:
    """Per-invocation context. ``tool_call_id`` is stable across executor retries."""

    run_id: str
    tool_call_id: str
    repo: RunRepository
    settings: Settings


def summarize_validation_error(exc: ValidationError, max_items: int = 5) -> str:
    """Short, LLM-readable summary such as ``severity: Input should be 'low', ...``."""
    parts = []
    for err in exc.errors()[:max_items]:
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        parts.append(f"{loc}: {err['msg']}")
    extra = len(exc.errors()) - max_items
    if extra > 0:
        parts.append(f"... and {extra} more")
    return "; ".join(parts)


def _strip_titles(node: Any) -> Any:
    """Recursively drop JSON-schema ``title`` keys (but keep properties literally named 'title')."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "title" and not isinstance(v, (dict, list)):
                continue
            if k == "properties" and isinstance(v, dict):
                out[k] = {pk: _strip_titles(pv) for pk, pv in v.items()}
            else:
                out[k] = _strip_titles(v)
        return out
    if isinstance(node, list):
        return [_strip_titles(i) for i in node]
    return node


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]
    output_model: ClassVar[type[BaseModel]]
    requires_approval: ClassVar[bool] = False

    @abstractmethod
    async def run(self, args: BaseModel, ctx: ToolContext) -> dict:
        """Execute the tool. Returns a raw dict; the executor validates it via ``validate_output``."""

    def validate_input(self, raw: dict) -> BaseModel:
        if not isinstance(raw, dict):
            raise ToolInputValidationError(f"arguments must be a JSON object, got {type(raw).__name__}")
        try:
            return self.input_model.model_validate(raw)
        except ValidationError as exc:
            raise ToolInputValidationError(summarize_validation_error(exc)) from exc

    def validate_output(self, raw: object) -> dict:
        try:
            return self.output_model.model_validate(raw).model_dump(mode="json")
        except ValidationError as exc:
            raise ToolOutputValidationError(summarize_validation_error(exc)) from exc

    def spec(self) -> dict:
        """OpenAI ``tools`` entry for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _strip_titles(self.input_model.model_json_schema()),
            },
        }
