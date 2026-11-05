"""Command-line interface: ``harness run "Check payments-api status"``.

Shares the exact same AgentRunner as the API. ``run`` is interactive by default: when the agent
wants to call ``create_incident`` you are asked to approve, reject or edit the call.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Optional

import typer

from app.config import get_settings
from app.container import Container, build_container
from app.core.errors import HarnessError
from app.core.state import ApprovalDecision, RunStatus, RunView
from app.observability.logging import configure_logging
from app.observability.tracer import build_span_tree

app = typer.Typer(add_completion=False, help="Ops agent harness CLI")


async def _container(verbose: bool) -> Container:
    settings = get_settings()
    # Logs go to stderr so stdout stays readable; WARNING unless --verbose.
    configure_logging("INFO" if verbose else "WARNING", settings.log_file, stream=sys.stderr)
    return await build_container(settings)


def _print_view(view: RunView) -> None:
    typer.echo(f"\nrun {view.id}  status={view.status.value}  steps={view.step_count}/{view.max_steps}")
    if view.final_answer:
        typer.echo(f"\nFinal answer:\n{view.final_answer}")
    if view.error_code:
        typer.echo(f"\nError: {view.error_code}: {view.error_detail}")
    if view.pending_approval:
        p = view.pending_approval
        typer.echo(f"\nAwaiting approval for {p.tool} (tool_call_id={p.tool_call_id}):")
        typer.echo(json.dumps(p.args, indent=2))


async def _print_steps(c: Container, run_id: str) -> None:
    for step in await c.repo.list_steps(run_id):
        p = step.payload
        if step.type == "llm_response" and p.get("kind") == "tool_calls":
            calls = ", ".join(tc["function"]["name"] for tc in p["assistant_message"].get("tool_calls", []))
            typer.echo(f"  [{step.idx}] LLM -> tool calls: {calls}")
        elif step.type == "tool_result":
            obs = p["observation"]
            mark = "ok" if obs.get("ok") else f"error {obs.get('error', {}).get('code')}"
            typer.echo(f"  [{step.idx}] {p['tool']}: {mark} (attempts={p['attempts']})")
        elif step.type in ("llm_malformed", "approval_requested", "approval_decided", "run_error"):
            typer.echo(f"  [{step.idx}] {step.type}: {json.dumps({k: v for k, v in p.items() if k != 'raw'})}")


def _ask_decision(view: RunView) -> ApprovalDecision:
    p = view.pending_approval
    assert p is not None
    choice = typer.prompt("Approve? [y]es / [n]o / [e]dit", default="n").strip().lower()
    if choice.startswith("y"):
        return ApprovalDecision(tool_call_id=p.tool_call_id, decision="approve", decided_by="cli-user")
    if choice.startswith("e"):
        edited: dict[str, Any] = {k: typer.prompt(k, default=str(v)) for k, v in p.args.items()}
        return ApprovalDecision(
            tool_call_id=p.tool_call_id, decision="approve", edited_args=edited, decided_by="cli-user"
        )
    reason = typer.prompt("Reason", default="rejected by operator")
    return ApprovalDecision(tool_call_id=p.tool_call_id, decision="reject", reason=reason, decided_by="cli-user")


@app.command()
def run(
    objective: str = typer.Argument(..., help="What the ops assistant should do"),
    max_steps: Optional[int] = typer.Option(None, help="Override MAX_STEPS"),  # noqa: UP045 - typer needs Optional
    max_time: Optional[float] = typer.Option(None, help="Override MAX_WALL_TIME_S"),  # noqa: UP045
    auto_approve: bool = typer.Option(False, help="Approve create_incident without asking (demo only)"),
    no_input: bool = typer.Option(False, help="Do not prompt; leave run awaiting approval"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print JSON logs to stderr"),
) -> None:
    """Start a run and drive it to completion (prompting for approvals)."""

    async def main() -> None:
        c = await _container(verbose)
        try:
            view = await c.runner.start(objective, max_steps, max_time)
            while view.status == RunStatus.AWAITING_APPROVAL and not no_input:
                await _print_steps(c, view.id)
                _print_view(view)
                p = view.pending_approval
                decision = (
                    ApprovalDecision(tool_call_id=p.tool_call_id, decision="approve", decided_by="cli-auto")
                    if auto_approve
                    else _ask_decision(view)
                )
                try:
                    view = await c.runner.resume(view.id, decision)
                except HarnessError as exc:
                    typer.echo(f"Approval error: {exc.code}: {exc.message}", err=True)
                    if auto_approve:  # the same decision would fail again; avoid looping forever
                        break
            typer.echo("\nSteps:")
            await _print_steps(c, view.id)
            _print_view(view)
        finally:
            await c.close()

    asyncio.run(main())


def _decide(run_id: str, decision: ApprovalDecision, verbose: bool) -> None:
    async def main() -> None:
        c = await _container(verbose)
        try:
            view = await c.runner.resume(run_id, decision)
            await _print_steps(c, run_id)
            _print_view(view)
        except HarnessError as exc:
            typer.echo(f"Error: {exc.code}: {exc.message}", err=True)
            raise typer.Exit(1) from exc
        finally:
            await c.close()

    asyncio.run(main())


@app.command()
def approve(
    run_id: str,
    tool_call_id: str,
    edit_json: Optional[str] = typer.Option(None, help="Edited args as JSON"),  # noqa: UP045
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Approve the pending tool call of a run (optionally with edited args)."""
    edited = json.loads(edit_json) if edit_json else None
    _decide(
        run_id,
        ApprovalDecision(tool_call_id=tool_call_id, decision="approve", edited_args=edited, decided_by="cli-user"),
        verbose,
    )


@app.command()
def reject(
    run_id: str,
    tool_call_id: str,
    reason: str = typer.Option("rejected by operator"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Reject the pending tool call of a run."""
    _decide(
        run_id,
        ApprovalDecision(tool_call_id=tool_call_id, decision="reject", reason=reason, decided_by="cli-user"),
        verbose,
    )


@app.command()
def show(run_id: str) -> None:
    """Show run status and execution history."""

    async def main() -> None:
        c = await _container(False)
        try:
            await _print_steps(c, run_id)
            _print_view(await c.runner.get_view(run_id))
        except HarnessError as exc:
            typer.echo(f"Error: {exc.code}: {exc.message}", err=True)
            raise typer.Exit(1) from exc
        finally:
            await c.close()

    asyncio.run(main())


@app.command()
def trace(run_id: str) -> None:
    """Print the span tree of a run as JSON."""

    async def main() -> None:
        c = await _container(False)
        try:
            spans = build_span_tree(await c.repo.list_spans(run_id))
            typer.echo(json.dumps([s.model_dump() for s in spans], indent=2, default=str))
        finally:
            await c.close()

    asyncio.run(main())


if __name__ == "__main__":
    app()
