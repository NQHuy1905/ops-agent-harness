"""All database access for runs, steps, tool calls, approvals, spans and mock incidents.

Each method uses its own short transaction so state is durable after every transition
(a crashed process can resume a run from what is persisted here).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.state import RunStatus, StepType, ToolCallStatus
from app.storage.models import (
    ApprovalRow,
    IncidentRow,
    RunRow,
    SpanRow,
    StepRow,
    ToolCallRow,
    utcnow,
)


def args_hash(tool_name: str, args: dict[str, Any]) -> str:
    """Stable hash of tool name + canonical JSON args (used for loop detection and approval binding)."""
    canonical = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class RunRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    # ------------------------------------------------------------------ runs
    async def create_run(
        self, objective: str, max_steps: int, max_wall_time_s: float, llm_provider: str, llm_model: str
    ) -> RunRow:
        row = RunRow(
            id=str(uuid.uuid4()),
            objective=objective,
            status=RunStatus.RUNNING.value,
            step_count=0,
            max_steps=max_steps,
            max_wall_time_s=max_wall_time_s,
            active_elapsed_s=0.0,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        async with self._sf.begin() as s:
            s.add(row)
        return row

    async def get_run(self, run_id: str) -> RunRow | None:
        async with self._sf() as s:
            return await s.get(RunRow, run_id)

    async def list_runs(self, limit: int = 50, offset: int = 0) -> list[RunRow]:
        async with self._sf() as s:
            res = await s.execute(select(RunRow).order_by(RunRow.created_at.desc()).limit(limit).offset(offset))
            return list(res.scalars())

    async def update_run(self, run_id: str, **fields: Any) -> None:
        fields["updated_at"] = utcnow()
        async with self._sf.begin() as s:
            await s.execute(update(RunRow).where(RunRow.id == run_id).values(**fields))

    async def transition_status(self, run_id: str, from_status: RunStatus, to_status: RunStatus, **fields: Any) -> bool:
        """Compare-and-set status. Returns False if the run was not in ``from_status``."""
        values = {"status": to_status.value, "updated_at": utcnow(), **fields}
        if to_status.is_terminal:
            values.setdefault("completed_at", utcnow())
        async with self._sf.begin() as s:
            res = await s.execute(
                update(RunRow).where(RunRow.id == run_id, RunRow.status == from_status.value).values(**values)
            )
            return res.rowcount == 1

    # ------------------------------------------------------------------ steps
    async def append_step(self, run_id: str, type_: StepType, payload: dict[str, Any]) -> StepRow:
        async with self._sf.begin() as s:
            next_idx = (
                await s.scalar(select(func.coalesce(func.max(StepRow.idx), -1)).where(StepRow.run_id == run_id))
            ) + 1
            row = StepRow(run_id=run_id, idx=next_idx, type=type_.value, payload=payload)
            s.add(row)
        return row

    async def list_steps(self, run_id: str) -> list[StepRow]:
        async with self._sf() as s:
            res = await s.execute(select(StepRow).where(StepRow.run_id == run_id).order_by(StepRow.idx))
            return list(res.scalars())

    # ------------------------------------------------------------------ tool calls
    async def add_tool_calls(self, run_id: str, step_idx: int, calls: list[tuple[str, str, dict]]) -> None:
        """Persist the tool calls of one LLM response as ``queued`` (call_id, tool_name, args)."""
        async with self._sf.begin() as s:
            for seq, (call_id, tool_name, args) in enumerate(calls):
                s.add(
                    ToolCallRow(
                        run_id=run_id,
                        call_id=call_id,
                        step_idx=step_idx,
                        seq=seq,
                        tool_name=tool_name,
                        args=args,
                        args_hash=args_hash(tool_name, args),
                        status=ToolCallStatus.QUEUED.value,
                    )
                )

    async def get_tool_call(self, run_id: str, call_id: str) -> ToolCallRow | None:
        async with self._sf() as s:
            res = await s.execute(
                select(ToolCallRow).where(ToolCallRow.run_id == run_id, ToolCallRow.call_id == call_id)
            )
            return res.scalar_one_or_none()

    async def queued_tool_calls(self, run_id: str) -> list[ToolCallRow]:
        async with self._sf() as s:
            res = await s.execute(
                select(ToolCallRow)
                .where(ToolCallRow.run_id == run_id, ToolCallRow.status == ToolCallStatus.QUEUED.value)
                .order_by(ToolCallRow.step_idx, ToolCallRow.seq)
            )
            return list(res.scalars())

    async def pending_approval_call(self, run_id: str) -> ToolCallRow | None:
        async with self._sf() as s:
            res = await s.execute(
                select(ToolCallRow).where(
                    ToolCallRow.run_id == run_id,
                    ToolCallRow.status == ToolCallStatus.PENDING_APPROVAL.value,
                )
            )
            return res.scalars().first()

    async def list_tool_calls(self, run_id: str) -> list[ToolCallRow]:
        async with self._sf() as s:
            res = await s.execute(
                select(ToolCallRow).where(ToolCallRow.run_id == run_id).order_by(ToolCallRow.step_idx, ToolCallRow.seq)
            )
            return list(res.scalars())

    async def update_tool_call(self, run_id: str, call_id: str, **fields: Any) -> None:
        fields["updated_at"] = utcnow()
        async with self._sf.begin() as s:
            await s.execute(
                update(ToolCallRow).where(ToolCallRow.run_id == run_id, ToolCallRow.call_id == call_id).values(**fields)
            )

    async def cancel_unfinished_tool_calls(self, run_id: str, error_type: str) -> None:
        """On a terminal abort, close out calls that will never run (queued) or were interrupted (running)."""
        async with self._sf.begin() as s:
            await s.execute(
                update(ToolCallRow)
                .where(
                    ToolCallRow.run_id == run_id,
                    ToolCallRow.status.in_([ToolCallStatus.QUEUED.value, ToolCallStatus.RUNNING.value]),
                )
                .values(status=ToolCallStatus.FAILED.value, error_type=error_type, updated_at=utcnow())
            )

    async def count_identical_calls(self, run_id: str, tool_name: str, hash_: str) -> int:
        """How many calls with the same tool + args were already dispatched (not merely queued)."""
        async with self._sf() as s:
            return await s.scalar(
                select(func.count())
                .select_from(ToolCallRow)
                .where(
                    ToolCallRow.run_id == run_id,
                    ToolCallRow.tool_name == tool_name,
                    ToolCallRow.args_hash == hash_,
                    ToolCallRow.status != ToolCallStatus.QUEUED.value,
                )
            )

    # ------------------------------------------------------------------ approvals
    async def record_approval(
        self,
        run_id: str,
        call_id: str,
        decision: str,
        edited_args: dict | None,
        reason: str | None,
        decided_by: str,
    ) -> ApprovalRow:
        row = ApprovalRow(
            run_id=run_id,
            call_id=call_id,
            decision=decision,
            edited_args=edited_args,
            reason=reason,
            decided_by=decided_by,
        )
        async with self._sf.begin() as s:
            s.add(row)
        return row

    async def list_approvals(self, run_id: str) -> list[ApprovalRow]:
        async with self._sf() as s:
            res = await s.execute(select(ApprovalRow).where(ApprovalRow.run_id == run_id).order_by(ApprovalRow.id))
            return list(res.scalars())

    # ------------------------------------------------------------------ spans
    async def add_span(self, row: SpanRow) -> None:
        async with self._sf.begin() as s:
            s.add(row)

    async def list_spans(self, run_id: str) -> list[SpanRow]:
        async with self._sf() as s:
            res = await s.execute(select(SpanRow).where(SpanRow.run_id == run_id).order_by(SpanRow.start_ns))
            return list(res.scalars())

    # ------------------------------------------------------------------ incidents (mock external system)
    async def create_incident(
        self, title: str, description: str, severity: str, idempotency_key: str, run_id: str | None
    ) -> tuple[IncidentRow, bool]:
        """Idempotent create. Returns (incident, created). Same key -> existing incident, no duplicate."""
        existing = await self.get_incident_by_key(idempotency_key)
        if existing:
            return existing, False
        row = IncidentRow(
            id=f"INC-{uuid.uuid4().hex[:8].upper()}",
            title=title,
            description=description,
            severity=severity,
            status="open",
            idempotency_key=idempotency_key,
            run_id=run_id,
        )
        try:
            async with self._sf.begin() as s:
                s.add(row)
        except IntegrityError:  # concurrent create with same key
            existing = await self.get_incident_by_key(idempotency_key)
            if existing is None:
                raise
            return existing, False
        return row, True

    async def get_incident_by_key(self, key: str) -> IncidentRow | None:
        async with self._sf() as s:
            res = await s.execute(select(IncidentRow).where(IncidentRow.idempotency_key == key))
            return res.scalar_one_or_none()

    async def list_incidents(self, run_id: str | None = None) -> list[IncidentRow]:
        async with self._sf() as s:
            q = select(IncidentRow).order_by(IncidentRow.created_at)
            if run_id:
                q = q.where(IncidentRow.run_id == run_id)
            res = await s.execute(q)
            return list(res.scalars())
