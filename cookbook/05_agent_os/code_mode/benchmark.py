"""
Code Mode — Multi-Model Benchmark
===================================
Runs two benchmark tasks across multiple LLM providers:

1. SIMPLE task (3 parallel tool calls) — shows where Traditional wins
2. SEQUENTIAL task (dependent chain: fetch -> compute -> fetch -> format)
   — shows where Code Mode wins

This gives an honest comparison of when to use each mode.

Requires API keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY

Run:
  .venvs/demo/bin/python cookbook/05_agent_os/code_mode/benchmark.py
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

from agno.agent import Agent
from agno.tools.calculator import CalculatorTools
from agno.tools.code_mode import CodeModeTool
from agno.tools.yfinance import YFinanceTools

# ---------------------------------------------------------------------------
# Task 1: SIMPLE — independent parallel calls, Code Mode should lose
# ---------------------------------------------------------------------------
SIMPLE_TASK = (
    "Get the current stock price for AAPL, MSFT, and GOOGL. "
    "Format as a markdown table: Symbol | Price."
)

# ---------------------------------------------------------------------------
# Task 2: SEQUENTIAL — dependent chain, Code Mode should win
# ---------------------------------------------------------------------------
SEQUENTIAL_TASK = (
    "Perform this multi-step financial analysis:\n"
    "1. Get current stock prices for AAPL, MSFT, NVDA, GOOGL, and META.\n"
    "2. Get analyst recommendations for each stock.\n"
    "3. Get stock fundamentals for each stock to find P/E ratios.\n"
    "4. Calculate the average price across all 5 stocks.\n"
    "5. Calculate the price-to-average ratio for each stock (price / average).\n"
    "6. Rank stocks by their P/E ratio (best value = lowest P/E).\n"
    "7. Build a final markdown report with:\n"
    "   - Table: Symbol | Price | P/E | Recommendation | Price/Avg Ratio\n"
    "   - Summary: cheapest by P/E, most expensive, average P/E\n"
)

CODE_MODE_INSTRUCTIONS = (
    "You solve tasks by writing Python code using the run_code tool.\n\n"
    "CRITICAL RULES:\n"
    "1. Write ONE COMPLETE Python program per run_code call.\n"
    "2. Call functions DIRECTLY: get_current_stock_price(symbol='AAPL').\n"
    "3. json and math are pre-loaded. Do NOT write import statements.\n"
    "4. All functions return JSON strings. Use json.loads() to parse.\n"
    "5. Store your final answer in `result` as a formatted markdown string.\n"
    "6. Handle ALL steps in a single program: fetch, compute, format."
)

TRAD_INSTRUCTIONS = (
    "You are a financial analyst. Use the available tools to answer questions. "
    "Be thorough and complete all requested analysis steps."
)


@dataclass
class BenchmarkResult:
    model_name: str
    mode: str
    task: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model_turns: int = 0
    tool_calls: int = 0
    duration: float = 0.0
    success: bool = True
    error: Optional[str] = None


def count_turns_and_calls(response) -> tuple:
    msgs = response.messages or []
    model_turns = sum(1 for m in msgs if hasattr(m, "role") and m.role == "assistant")
    tool_calls = len(response.tools) if response.tools else 0
    return model_turns, tool_calls


def run_one(
    agent: Agent, task: str, model_name: str, mode: str, task_label: str
) -> BenchmarkResult:
    try:
        t0 = time.time()
        resp = agent.run(task)
        elapsed = time.time() - t0

        m = resp.metrics
        turns, calls = count_turns_and_calls(resp)
        return BenchmarkResult(
            model_name=model_name,
            mode=mode,
            task=task_label,
            input_tokens=m.input_tokens if m else 0,
            output_tokens=m.output_tokens if m else 0,
            total_tokens=m.total_tokens if m else 0,
            model_turns=turns,
            tool_calls=calls,
            duration=elapsed,
        )
    except Exception as e:
        return BenchmarkResult(
            model_name=model_name,
            mode=mode,
            task=task_label,
            success=False,
            error=str(e)[:100],
        )


def run_benchmark(model, model_name: str, tools_simple, tools_sequential) -> list:
    results = []

    for task, task_label, tools in [
        (SIMPLE_TASK, "Simple", tools_simple),
        (SEQUENTIAL_TASK, "Sequential", tools_sequential),
    ]:
        print(f"  {task_label}: Code Mode...", end=" ", flush=True)
        cm = CodeModeTool(tools=tools)
        code_agent = Agent(
            name="Code Mode",
            model=model,
            tools=[cm],
            instructions=CODE_MODE_INSTRUCTIONS,
            markdown=True,
        )
        r = run_one(code_agent, task, model_name, "Code Mode", task_label)
        results.append(r)
        status = (
            f"{r.total_tokens:,} tokens, {r.duration:.0f}s"
            if r.success
            else f"FAILED: {r.error}"
        )
        print(status)

        print(f"  {task_label}: Traditional...", end=" ", flush=True)
        trad_agent = Agent(
            name="Traditional",
            model=model,
            tools=tools,
            instructions=TRAD_INSTRUCTIONS,
            markdown=True,
        )
        r = run_one(trad_agent, task, model_name, "Traditional", task_label)
        results.append(r)
        status = (
            f"{r.total_tokens:,} tokens, {r.duration:.0f}s"
            if r.success
            else f"FAILED: {r.error}"
        )
        print(status)

    return results


def print_results(all_results: list):
    for task_label in ["Simple", "Sequential"]:
        task_results = [r for r in all_results if r.task == task_label]
        if not task_results:
            continue

        print("\n" + "=" * 110)
        expected = "Traditional wins" if task_label == "Simple" else "Code Mode wins"
        print(f"  {task_label.upper()} TASK  (expected: {expected})")
        print("=" * 110)
        header = f"{'Model':<20} {'Mode':<15} {'Input':>10} {'Output':>10} {'Total':>10} {'Turns':>7} {'Calls':>7} {'Time':>8}"
        print(header)
        print("-" * 110)

        models_seen = []
        for r in task_results:
            if r.model_name not in models_seen:
                models_seen.append(r.model_name)

        for model_name in models_seen:
            model_results = [r for r in task_results if r.model_name == model_name]
            for r in model_results:
                if not r.success:
                    print(
                        f"{r.model_name:<20} {r.mode:<15} {'FAILED':>10}  {r.error or ''}"
                    )
                    continue
                print(
                    f"{r.model_name:<20} {r.mode:<15} "
                    f"{r.input_tokens:>10,} {r.output_tokens:>10,} {r.total_tokens:>10,} "
                    f"{r.model_turns:>7} {r.tool_calls:>7} {r.duration:>7.1f}s"
                )

            code = next(
                (r for r in model_results if r.mode == "Code Mode" and r.success), None
            )
            trad = next(
                (r for r in model_results if r.mode == "Traditional" and r.success),
                None,
            )
            if code and trad and trad.total_tokens > 0:
                pct = (1 - code.total_tokens / trad.total_tokens) * 100
                saved = trad.total_tokens - code.total_tokens
                winner = "Code Mode" if saved > 0 else "Traditional"
                print(
                    f"{'':>35} --> {winner} wins: "
                    f"{abs(saved):,} tokens ({abs(pct):.0f}%), "
                    f"{abs(trad.model_turns - code.model_turns)} turn diff, "
                    f"{abs(trad.duration - code.duration):.0f}s time diff"
                )
            print()


if __name__ == "__main__":
    # Simple task: just price tools
    yf_simple = [YFinanceTools(enable_stock_price=True)]

    # Sequential task: price + fundamentals + recommendations + calculator = 17 tools
    yf_full = [
        YFinanceTools(
            enable_stock_price=True,
            enable_stock_fundamentals=True,
            enable_analyst_recommendations=True,
            enable_company_info=True,
            enable_key_financial_ratios=True,
            enable_income_statements=True,
            enable_historical_prices=True,
            enable_technical_indicators=True,
            enable_company_news=True,
        ),
        CalculatorTools(),
    ]

    all_results = []
    models_to_test = []

    if os.getenv("ANTHROPIC_API_KEY"):
        from agno.models.anthropic import Claude

        models_to_test.append(
            (Claude(id="claude-sonnet-4-20250514"), "Claude Sonnet 4")
        )

    if os.getenv("OPENAI_API_KEY"):
        from agno.models.openai import OpenAIChat

        models_to_test.append((OpenAIChat(id="gpt-4o"), "GPT-4o"))

    if os.getenv("GOOGLE_API_KEY"):
        from agno.models.google import Gemini

        models_to_test.append((Gemini(id="gemini-2.5-flash"), "Gemini 2.5 Flash"))

    if not models_to_test:
        print(
            "No API keys found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY."
        )
        sys.exit(1)

    print(f"Models: {', '.join(name for _, name in models_to_test)}")
    print(f"Simple task tools: {sum(len(t.get_functions()) for t in yf_simple)}")
    print(f"Sequential task tools: {sum(len(t.get_functions()) for t in yf_full)}")
    print()

    for model, name in models_to_test:
        print(f"--- {name} ---")
        results = run_benchmark(model, name, yf_simple, yf_full)
        all_results.extend(results)

    print_results(all_results)
