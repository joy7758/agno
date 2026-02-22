"""
Code Mode + GitHub MCP — Repository Analysis
==============================================
Demonstrates CodeModeTool with GitHub MCP server (51 tools).
With 51 tools, discovery mode is essential — injecting all stubs
into the context would consume ~10K tokens on every model turn.
Instead, the LLM uses search_tools to find relevant GitHub API
functions, then writes one code block that chains:
  search_issues -> get_issue -> list_pull_requests -> analyze

Setup:
  export GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...

Run:
  .venvs/demo/bin/python cookbook/05_agent_os/code_mode/github_analyst.py
"""

import asyncio
import os

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.code_mode import CodeModeTool
from agno.tools.mcp import MCPTools

TASK = (
    "Analyze the repository 'agno-agi/agno'. "
    "Find the 5 most recent open issues, and for each issue get: "
    "title, number, labels, creation date, and number of comments. "
    "Then check if there are any open pull requests. "
    "Summarize everything in a markdown report with two sections: "
    "'Recent Issues' (as a table) and 'Open PRs' (count + titles)."
)

CODE_MODE_INSTRUCTIONS = (
    "You solve tasks by writing Python code using the run_code tool.\n\n"
    "WORKFLOW:\n"
    "1. Use search_tools to discover available GitHub functions (e.g., search for 'issue', 'pull').\n"
    "2. Write ONE complete Python program in run_code that performs the entire analysis.\n\n"
    "CRITICAL RULES:\n"
    "- Write ONE complete program per run_code call. Do NOT make multiple small calls.\n"
    "- Call functions DIRECTLY: list_issues(owner='x', repo='y'). No prefix needed.\n"
    "- json is pre-imported. Always use json.loads() to parse tool return values.\n"
    "- Store your final answer in `result` as a formatted markdown string.\n"
    "- Wrap tool calls in try/except to handle API errors gracefully."
)


async def run_github_code_mode():
    token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not token:
        print("Set GITHUB_PERSONAL_ACCESS_TOKEN environment variable.")
        return

    npx_cmd = "npx.cmd" if os.name == "nt" else "npx"

    async with MCPTools(
        f"{npx_cmd} -y @modelcontextprotocol/server-github",
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": token},
    ) as mcp:
        tool_count = len(mcp.functions)
        print(f"GitHub MCP tools loaded: {tool_count}")

        cm = CodeModeTool(tools=[mcp])
        print(f"Discovery mode: {cm._discovery_enabled}")

        agent = Agent(
            name="GitHub Analyst",
            model=Claude(id="claude-sonnet-4-20250514"),
            tools=[cm],
            tool_call_limit=5,
            instructions=CODE_MODE_INSTRUCTIONS,
            markdown=True,
        )

        response = await agent.arun(TASK)
        print(f"\n{response.content}\n")

        m = response.metrics
        if m:
            print(f"Total tokens: {m.total_tokens:,}")
            print(f"Input tokens: {m.input_tokens:,}")
            print(f"Output tokens: {m.output_tokens:,}")


if __name__ == "__main__":
    asyncio.run(run_github_code_mode())
