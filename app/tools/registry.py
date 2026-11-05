"""Tool registry and default tool set."""

from __future__ import annotations

from app.config import Settings
from app.core.errors import ToolNotFoundError
from app.tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"unknown tool '{name}'. available: {', '.join(self.names())}") from None

    def names(self) -> list[str]:
        return list(self._tools)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def llm_tool_specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]


def build_default_registry(settings: Settings) -> ToolRegistry:
    """Registry with the three built-in tools. ``settings`` is accepted for future per-tool config."""
    from app.tools.create_incident import CreateIncidentTool
    from app.tools.get_service_status import GetServiceStatusTool
    from app.tools.search_knowledge_base import SearchKnowledgeBaseTool

    reg = ToolRegistry()
    for tool in (SearchKnowledgeBaseTool(), GetServiceStatusTool(), CreateIncidentTool()):
        reg.register(tool)
    return reg
