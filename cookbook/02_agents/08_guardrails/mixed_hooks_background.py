"""
Mixed Hooks with Background Mode
=================================

Demonstrates that guardrails block execution even when mixed with
plain hooks in background mode. This is the scenario where pre_hooks
contains both a logging function and a PII guardrail.

Key behavior:
- Plain hooks are deferred (buffered) until all guardrails pass
- If a guardrail rejects, NO background tasks are queued
- If all guardrails pass, plain hooks are queued as background tasks
"""

from typing import Any, List

from agno.agent._hooks import execute_pre_hooks
from agno.exceptions import InputCheckError
from agno.guardrails import PIIDetectionGuardrail
from agno.run import RunContext
from agno.run.agent import RunInput
from agno.utils.hooks import normalize_pre_hooks


# ---------------------------------------------------------------------------
# Simulate a background task queue (what FastAPI BackgroundTasks does)
# ---------------------------------------------------------------------------
class FakeBackgroundTasks:
    def __init__(self):
        self.tasks: List = []

    def add_task(self, fn, **kwargs):
        self.tasks.append((fn.__name__ if hasattr(fn, "__name__") else str(fn), kwargs))


# ---------------------------------------------------------------------------
# A plain logging hook (non-guardrail)
# ---------------------------------------------------------------------------
def log_request(run_input: RunInput, agent: Any) -> None:
    print(f"  [log_request] Logging input: {run_input.input_content}")


# ---------------------------------------------------------------------------
# Mock agent with background mode enabled
# ---------------------------------------------------------------------------
class FakeAgent:
    def __init__(self):
        self._run_hooks_in_background = True
        self.debug_mode = False
        self.events_to_skip = None
        self.store_events = False
        self.name = "test-agent"


class FakeSession:
    session_id = "s1"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def main():
    agent = FakeAgent()
    pii_guardrail = PIIDetectionGuardrail()
    run_context = RunContext(
        run_id="r1", session_id="s1", session_state={}, metadata={}
    )

    # Mix a plain hook BEFORE the guardrail - this is the key scenario
    hooks = [log_request] + normalize_pre_hooks([pii_guardrail], async_mode=False)

    # -----------------------------------------------------------------------
    # Test 1: Input WITH PII - guardrail should reject, log_request should NOT run
    # -----------------------------------------------------------------------
    print("Test 1: Input with PII (SSN)")
    print("-" * 40)
    bt = FakeBackgroundTasks()
    try:
        list(
            execute_pre_hooks(
                agent=agent,
                hooks=hooks,
                run_response=None,
                run_input=RunInput(input_content="My SSN is 123-45-6789"),
                session=FakeSession(),
                run_context=run_context,
                background_tasks=bt,
            )
        )
        print("  [FAIL] Should have been blocked!")
    except InputCheckError as e:
        print(f"  [BLOCKED] Guardrail rejected: {e.message}")
        print(f"  Background tasks queued: {len(bt.tasks)}")
        if len(bt.tasks) == 0:
            print("  [PASS] No background tasks leaked")
        else:
            print(f"  [FAIL] Tasks leaked: {[t[0] for t in bt.tasks]}")

    # -----------------------------------------------------------------------
    # Test 2: Clean input - guardrail passes, log_request should run in background
    # -----------------------------------------------------------------------
    print("\nTest 2: Clean input (no PII)")
    print("-" * 40)
    bt = FakeBackgroundTasks()
    list(
        execute_pre_hooks(
            agent=agent,
            hooks=hooks,
            run_response=None,
            run_input=RunInput(input_content="What is the weather today?"),
            session=FakeSession(),
            run_context=run_context,
            background_tasks=bt,
        )
    )
    print(f"  Background tasks queued: {len(bt.tasks)}")
    if len(bt.tasks) == 1:
        print(f"  [PASS] log_request queued as background task: {bt.tasks[0][0]}")
    else:
        print(f"  [FAIL] Expected 1 task, got {len(bt.tasks)}")

    print("\n" + "=" * 40)
    print("Mixed hooks demo complete.")


if __name__ == "__main__":
    main()
