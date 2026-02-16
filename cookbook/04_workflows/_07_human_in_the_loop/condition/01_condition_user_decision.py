"""
Condition with User Decision HITL Example

This example demonstrates how to use HITL with a Condition component,
allowing the user to decide which branch to execute at runtime.

When `requires_confirmation=True` on a Condition:
- User confirms -> Execute the `steps` (if branch)
- User rejects -> Execute the `else_steps` (else branch) if provided, otherwise skip

This is useful for:
- User-driven decision points
- Interactive branching workflows
- A/B testing with human judgment
"""

from agno.db.sqlite import SqliteDb
from agno.workflow.condition import Condition
from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput
from agno.workflow.workflow import Workflow


# ============================================================
# Step functions
# ============================================================
def analyze_data(step_input: StepInput) -> StepOutput:
    """Analyze the data."""
    user_query = step_input.input or "data"
    return StepOutput(
        content=f"Analysis complete for '{user_query}':\n"
        "- Found potential issues that may require detailed review\n"
        "- Quick summary is available\n\n"
        "Would you like to proceed with detailed analysis?"
    )


def detailed_analysis(step_input: StepInput) -> StepOutput:
    """Perform detailed analysis (if branch)."""
    return StepOutput(
        content="Detailed Analysis Results:\n"
        "- Comprehensive review completed\n"
        "- All edge cases examined\n"
        "- Full report generated\n"
        "- Processing time: 10 minutes"
    )


def quick_summary(step_input: StepInput) -> StepOutput:
    """Provide quick summary (else branch)."""
    return StepOutput(
        content="Quick Summary:\n"
        "- Basic metrics computed\n"
        "- Key highlights identified\n"
        "- Processing time: 1 minute"
    )


def generate_report(step_input: StepInput) -> StepOutput:
    """Generate final report."""
    previous_content = step_input.previous_step_content or "No analysis"
    return StepOutput(
        content=f"=== FINAL REPORT ===\n\n{previous_content}\n\n"
        "Report generated successfully."
    )


# Define the steps
analyze_step = Step(name="analyze_data", executor=analyze_data)

# Condition with HITL - user decides which branch to take
# The evaluator is ignored when requires_confirmation=True
# User confirms -> detailed_analysis (if branch)
# User rejects -> quick_summary (else branch)
analysis_condition = Condition(
    name="analysis_depth_decision",
    evaluator=True,  # This is ignored when requires_confirmation=True
    steps=[Step(name="detailed_analysis", executor=detailed_analysis)],
    else_steps=[Step(name="quick_summary", executor=quick_summary)],
    requires_confirmation=True,
    confirmation_message="Would you like to perform detailed analysis? (yes=detailed, no=quick summary)",
)

report_step = Step(name="generate_report", executor=generate_report)

# Create workflow with database for HITL persistence
workflow = Workflow(
    name="condition_hitl_demo",
    steps=[analyze_step, analysis_condition, report_step],
    db=SqliteDb(db_file="tmp/condition_hitl.db"),
)

if __name__ == "__main__":
    print("=" * 60)
    print("Condition with User Decision HITL Example")
    print("=" * 60)

    run_output = workflow.run("Q4 sales data")

    # Handle HITL pauses
    while run_output.is_paused:
        # Handle Step requirements (confirmation)
        for requirement in run_output.steps_requiring_confirmation:
            print(f"\n[DECISION POINT] {requirement.step_name}")
            print(f"[HITL] {requirement.confirmation_message}")

            user_choice = input("\nYour choice (yes/no): ").strip().lower()
            if user_choice in ("yes", "y"):
                requirement.confirm()
                print("[HITL] Confirmed - executing 'if' branch")
            else:
                requirement.reject()
                print("[HITL] Rejected - executing 'else' branch")

        run_output = workflow.continue_run(run_output)

    print("\n" + "=" * 60)
    print(f"Status: {run_output.status}")
    print("=" * 60)
    print(run_output.content)
