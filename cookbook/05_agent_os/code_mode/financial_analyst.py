"""
Code Mode — Financial Analyst
==============================
Side-by-side comparison of Code Mode vs Traditional tool_call mode
on a real multi-step financial analysis task.

Based on: cookbook/90_models/anthropic/financial_analyst_thinking.py

Run:
  .venvs/demo/bin/python cookbook/05_agent_os/code_mode/financial_analyst.py
"""

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.calculator import CalculatorTools
from agno.tools.code_mode import CodeModeTool
from agno.tools.yfinance import YFinanceTools

TASK = (
    "I want to invest $50,000 split equally between Apple (AAPL) and Tesla (TSLA). "
    "Get current prices, calculate how many shares of each I can buy, "
    "then calculate what my total portfolio value would be if both stocks increased by 15%. "
    "Also calculate the percentage return on my initial $50,000 investment. "
    "Present results in a markdown table."
)

CODE_MODE_INSTRUCTIONS = (
    "You solve tasks by writing Python code using the run_code tool.\n\n"
    "CRITICAL RULES:\n"
    "1. Write ONE COMPLETE Python program per run_code call. NEVER make multiple small calls.\n"
    "2. Your code should handle the ENTIRE task: fetch data, calculate, format.\n"
    "3. Call functions DIRECTLY: get_current_stock_price(symbol='AAPL'). No prefix.\n"
    "4. json and math are pre-loaded. Do NOT write import statements.\n"
    "5. All functions return JSON strings. Always use json.loads() to parse results.\n"
    "6. Store your final answer in `result` as a formatted markdown string.\n\n"
    "PATTERN:\n"
    "  price_data = json.loads(get_current_stock_price(symbol='AAPL'))\n"
    "  calc = json.loads(divide(a=25000, b=float(price_data['price'])))\n"
    "  result = '| Stock | Shares | Value |\\n' + rows"
)

MODEL = "claude-sonnet-4-20250514"


def run_code_mode():
    yf = YFinanceTools(enable_stock_price=True)
    calc = CalculatorTools()

    agent = Agent(
        name="Code Mode Financial Analyst",
        model=Claude(id=MODEL),
        tools=[CodeModeTool(tools=[yf, calc])],
        tool_call_limit=3,
        instructions=CODE_MODE_INSTRUCTIONS,
        markdown=True,
    )

    return agent.run(TASK)


def run_traditional():
    agent = Agent(
        name="Traditional Financial Analyst",
        model=Claude(id=MODEL),
        tools=[YFinanceTools(enable_stock_price=True), CalculatorTools()],
        instructions=[
            "You are a financial analysis assistant.",
            "Use the calculator tool for all math operations.",
            "Use YFinance tools to get stock prices.",
        ],
        markdown=True,
    )

    return agent.run(TASK)


if __name__ == "__main__":
    print("=" * 70)
    print("CODE MODE")
    print("=" * 70)
    code_response = run_code_mode()
    print(f"\n{code_response.content}\n")
    cm = code_response.metrics

    print("\n" + "=" * 70)
    print("TRADITIONAL MODE")
    print("=" * 70)
    trad_response = run_traditional()
    print(f"\n{trad_response.content}\n")
    tm = trad_response.metrics

    cm_input = cm.input_tokens if cm else 0
    tm_input = tm.input_tokens if tm else 0
    cm_output = cm.output_tokens if cm else 0
    tm_output = tm.output_tokens if tm else 0
    cm_total = cm.total_tokens if cm else 0
    tm_total = tm.total_tokens if tm else 0

    print("\n" + "=" * 70)
    print("TOKEN COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} {'Code Mode':>15} {'Traditional':>15} {'Savings':>15}")
    print("-" * 70)
    print(
        f"{'Input tokens':<25} {cm_input:>15,} {tm_input:>15,} {tm_input - cm_input:>15,}"
    )
    print(
        f"{'Output tokens':<25} {cm_output:>15,} {tm_output:>15,} {tm_output - cm_output:>15,}"
    )
    print(
        f"{'Total tokens':<25} {cm_total:>15,} {tm_total:>15,} {tm_total - cm_total:>15,}"
    )
    if tm_total > 0:
        pct = (1 - cm_total / tm_total) * 100
        print(f"{'Reduction':<25} {pct:>14.0f}%")
