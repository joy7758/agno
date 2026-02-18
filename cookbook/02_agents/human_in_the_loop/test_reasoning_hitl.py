"""
Test: Reasoning Model + UserControlFlowTools (get_user_input)
=============================================================
Reproduces the bug where reasoning models (o4-mini) fail with
"No tool output found for function call" when using get_user_input.

Non-interactive: simulates user providing input programmatically.
"""

import os
import sys

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat, OpenAIResponses
from agno.tools import Toolkit
from agno.tools.user_control_flow import UserControlFlowTools

# Simple toolkit that triggers user input
class EmailTools(Toolkit):
    def __init__(self, *args, **kwargs):
        super().__init__(name="EmailTools", tools=[self.send_email], *args, **kwargs)

    def send_email(self, subject: str, body: str, to_address: str) -> str:
        """Send an email to the given address.

        Args:
            subject (str): The subject of the email.
            body (str): The body of the email.
            to_address (str): The address to send the email to.
        """
        return f"Sent email to {to_address} with subject '{subject}'"


def test_model(model, model_name: str):
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    agent = Agent(
        model=model,
        tools=[EmailTools(), UserControlFlowTools()],
        markdown=True,
        telemetry=False,
        db=SqliteDb(db_file="tmp/test_reasoning_hitl.db"),
    )

    # Step 1: Initial run — prompt designed to trigger BOTH get_user_input AND send_email
    # in the same turn (parallel tool calls)
    print("\n[Step 1] Running agent...")
    try:
        run_response = agent.run(
            "Send an email with subject 'Hello' and body 'How are you?' to whoever. "
            "Also use get_user_input to ask for the recipient address at the same time."
        )
    except Exception as e:
        print(f"  FAILED on initial run: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"  is_paused: {run_response.is_paused}")
    print(f"  active_requirements: {len(run_response.active_requirements)}")

    # Dump all messages to see what's in history after pause
    if hasattr(agent, 'run_messages') and agent.run_messages:
        msgs = agent.run_messages.messages
        print(f"\n  === Message history after pause ({len(msgs)} msgs) ===")
        for j, m in enumerate(msgs):
            tc_info = ""
            if m.tool_calls:
                tc_info = f" tool_calls={[tc.get('function',{}).get('name','?') for tc in m.tool_calls]}"
            print(f"  [{j}] role={m.role} tool_call_id={m.tool_call_id}{tc_info} content={str(m.content)[:80] if m.content else 'None'}")
        print(f"  === End message history ===")

    if not run_response.is_paused:
        print("  Agent did not pause (may have made up an address). Trying more explicit prompt...")
        run_response = agent.run(
            "Use the get_user_input tool to ask me for the recipient email address, then send an email with subject 'Test' and body 'Hello'"
        )
        print(f"  is_paused: {run_response.is_paused}")
        if not run_response.is_paused:
            print("  SKIP: Agent still didn't pause. Model may not use get_user_input.")
            return None

    # Step 2: Simulate user providing input (loop in case model asks again)
    max_rounds = 3
    for round_num in range(1, max_rounds + 1):
        print(f"\n[Step 2, round {round_num}] Providing user input and continuing run...")
        print(f"  Requirements: {len(run_response.active_requirements)}")
        for i, req in enumerate(run_response.active_requirements):
            print(f"  Req[{i}]: needs_user_input={req.needs_user_input}, needs_confirmation={req.needs_confirmation}")
            print(f"    tool_name={req.tool_execution.tool_name if req.tool_execution else 'N/A'}")
            print(f"    tool_args={req.tool_execution.tool_args if req.tool_execution else 'N/A'}")
            if req.needs_user_input and req.user_input_schema:
                for field in req.user_input_schema:
                    print(f"    Field: {field.name} (type={field.field_type}, desc={field.description})")
                    # Simulate user typing
                    if "email" in (field.name or "").lower() or "address" in (field.name or "").lower() or "to" in (field.name or "").lower():
                        field.value = "test@example.com"
                    else:
                        field.value = "test@example.com"
                    print(f"      -> Set to: {field.value}")

        try:
            run_response = agent.continue_run(
                run_id=run_response.run_id,
                requirements=run_response.requirements,
            )
        except Exception as e:
            print(f"  FAILED on continue_run: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False

        print(f"  is_paused: {run_response.is_paused}")
        # Debug: show messages in the run
        if hasattr(agent, 'run_messages') and agent.run_messages:
            msgs = agent.run_messages.messages
            print(f"  Messages in history: {len(msgs)}")
            for j, m in enumerate(msgs[-5:]):
                print(f"    msg[{len(msgs)-5+j}] role={m.role} tool_call_id={m.tool_call_id} content={str(m.content)[:100] if m.content else 'None'}")
        if not run_response.is_paused:
            break

    print(f"\n[Result] is_paused: {run_response.is_paused}")
    print(f"[Result] content: {(run_response.content or '')[:200]}")
    if run_response.is_paused:
        print("  WARN: Still paused after max rounds")
    print(f"  SUCCESS")
    return True


if __name__ == "__main__":
    models = [
        # Reasoning model via Responses API
        (OpenAIResponses(id="o4-mini"), "o4-mini (Responses, reasoning)"),
    ]

    results = {}
    for model, name in models:
        result = test_model(model, name)
        results[name] = result

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for name, result in results.items():
        status = "PASS" if result is True else ("SKIP" if result is None else "FAIL")
        print(f"  {status}: {name}")
