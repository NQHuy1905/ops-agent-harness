"""REST endpoints. A run executes synchronously inside the request (prototype design):
``POST /runs`` returns once the run completes, fails, hits a limit (200) or pauses for approval (202).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request, Response

from app.api.schemas import (
    ApprovalView,
    ErrorResponse,
    IncidentView,
    RunCreate,
    RunHistory,
    RunTrace,
    ToolCallView,
    ToolInfo,
)
from app.container import Container
from app.core.errors import RunNotFoundError
from app.core.state import ApprovalDecision, RunStatus, RunView, StepType, StepView
from app.observability.tracer import build_span_tree

router = APIRouter()
_ERRORS = {404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}}


def get_container(request: Request) -> Container:
    return request.app.state.container


def _status_code(view: RunView) -> int:
    return 202 if view.status == RunStatus.AWAITING_APPROVAL else 200


@router.get("/health")
async def health(c: Container = Depends(get_container)) -> dict:
    return {"status": "ok", "llm_provider": c.llm.provider, "llm_model": c.llm.model}


@router.get("/tools", response_model=list[ToolInfo])
async def list_tools(c: Container = Depends(get_container)) -> list[ToolInfo]:
    return [
        ToolInfo(
            name=t.name,
            description=t.description,
            requires_approval=t.requires_approval,
            parameters=t.spec()["function"]["parameters"],
        )
        for t in c.registry.all()
    ]


@router.post(
    "/runs",
    response_model=RunView,
    responses={202: {"model": RunView, "description": "Paused: awaiting approval"}, **_ERRORS},
)
async def create_run(body: RunCreate, response: Response, c: Container = Depends(get_container)) -> RunView:
    view = await c.runner.start(body.objective, body.max_steps, body.max_wall_time_s)
    response.status_code = _status_code(view)
    return view


@router.get("/runs", response_model=list[RunView])
async def list_runs(
    limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0), c: Container = Depends(get_container)
) -> list[RunView]:
    return [await c.runner.get_view(r.id) for r in await c.repo.list_runs(limit, offset)]


@router.get("/runs/{run_id}", response_model=RunView, responses=_ERRORS)
async def get_run(run_id: str, c: Container = Depends(get_container)) -> RunView:
    return await c.runner.get_view(run_id)


@router.get("/runs/{run_id}/history", response_model=RunHistory, responses=_ERRORS)
async def get_history(run_id: str, c: Container = Depends(get_container)) -> RunHistory:
    view = await c.runner.get_view(run_id)
    steps = await c.repo.list_steps(run_id)
    calls = await c.repo.list_tool_calls(run_id)
    approvals = await c.repo.list_approvals(run_id)
    return RunHistory(
        run=view,
        steps=[StepView(idx=s.idx, type=StepType(s.type), payload=s.payload, created_at=s.created_at) for s in steps],
        tool_calls=[
            ToolCallView(
                call_id=t.call_id,
                step_idx=t.step_idx,
                seq=t.seq,
                tool=t.tool_name,
                args=t.args,
                status=t.status,
                result=t.result,
                error_type=t.error_type,
                attempts=t.attempts,
                duration_ms=t.duration_ms,
            )
            for t in calls
        ],
        approvals=[
            ApprovalView(
                call_id=a.call_id,
                decision=a.decision,
                edited_args=a.edited_args,
                reason=a.reason,
                decided_by=a.decided_by,
                decided_at=a.decided_at,
            )
            for a in approvals
        ],
    )


@router.get("/runs/{run_id}/trace", response_model=RunTrace, responses=_ERRORS)
async def get_trace(run_id: str, c: Container = Depends(get_container)) -> RunTrace:
    if await c.repo.get_run(run_id) is None:
        raise RunNotFoundError(f"run '{run_id}' not found")
    return RunTrace(run_id=run_id, spans=build_span_tree(await c.repo.list_spans(run_id)))


@router.post(
    "/runs/{run_id}/approval",
    response_model=RunView,
    responses={202: {"model": RunView, "description": "Paused again: another approval needed"}, **_ERRORS},
)
async def decide_approval(
    run_id: str, body: ApprovalDecision, response: Response, c: Container = Depends(get_container)
) -> RunView:
    view = await c.runner.resume(run_id, body)
    response.status_code = _status_code(view)
    return view


@router.get("/incidents", response_model=list[IncidentView])
async def list_incidents(run_id: str | None = None, c: Container = Depends(get_container)) -> list[IncidentView]:
    return [
        IncidentView(
            id=i.id,
            title=i.title,
            description=i.description,
            severity=i.severity,
            status=i.status,
            run_id=i.run_id,
            created_at=i.created_at,
        )
        for i in await c.repo.list_incidents(run_id)
    ]
