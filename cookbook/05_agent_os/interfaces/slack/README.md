# Slack Cookbook

Examples for `interfaces/slack` in AgentOS.

## Files
- `agent_with_user_memory.py` — Agent With User Memory.
- `basic.py` — Basic.
- `basic_workflow.py` — Basic Workflow.
- `channel_summarizer.py` — Channel Summarizer.
- `file_analyst.py` — File Analyst.
- `multimodal_team.py` — Multi-agent team with image input (GPT-4o vision) and output (DALL-E).
- `multimodal_workflow.py` — Parallel workflow with visual analysis, web research, and image generation.
- `multiple_instances.py` — Two Slack apps on one server with separate credentials.
- `reasoning_agent.py` — Reasoning Agent.
- `research_assistant.py` — Research Assistant.
- `streaming.py` — Streaming with tool progress cards, thread titles, and suggested prompts.
- `streaming_deep_research.py` — Deep research agent with reasoning model and multiple tools.
- `streaming_research.py` — Streaming research agent with reasoning and rich plan-block cards.
- `streaming_team.py` — Streaming multi-agent team with coordinate mode.
- `support_team.py` — Support Team.
- `test_all.py` — Two apps (workflow + team) on one server for comprehensive testing.
- `test_streaming_events.py` — Switch between agent/team/workflow with TEST_MODE env var.

## Prerequisites
- Load environment variables with `direnv allow` (requires `.envrc`).
- Run examples with `.venvs/demo/bin/python <path-to-file>.py`.
- Some examples require local services (for example Postgres, Redis, Slack, or MCP servers).
